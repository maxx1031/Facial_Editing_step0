#!/usr/bin/env python3
"""
Step 5: Build the I_r reference image pool for expression transfer.

Two sources:
  --source real    Process real face datasets (AffectNet / RAF-DB / generic label dirs)
  --source flux    Generate synthetic I_r images with Flux (diverse persons × expressions)
  --source both    Run both (default)

Real dataset directory layout expected:
  <real_dataset_dir>/
    <dataset_name>/
      <emotion_label>/     # e.g. "happy", "sad", "angry", "neutral", ...
        *.jpg / *.png

Output:
  data/reference_pool/
    happy/
      real_<hash>.png
      flux_<idx>.png
      ...
    sad/
      ...
  data/reference_pool/pool_metadata.jsonl
    {"pool_id": "happy_0042", "path": "...", "expression": "happy",
     "source": "real|flux", "det_score": 0.95, "emotion_conf": 0.88,
     "face_area_frac": 0.35}
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import jsonlines
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    def expand(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            return os.environ.get(var, "")
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        return obj
    return expand(raw)


# ---------------------------------------------------------------------------
# Reuse model loaders from step4 (copy to avoid import dependency)
# ---------------------------------------------------------------------------

def load_insightface_app(det_thresh: float = 0.5):
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
    return app


def load_hsemotion(model_name: str = "enet_b0_8_best_afew"):
    import torch
    from hsemotion.facial_emotions import HSEmotionRecognizer
    original_torch_load = torch.load
    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)
    try:
        torch.load = _torch_load_compat
        recognizer = HSEmotionRecognizer(model_name=model_name)
    finally:
        torch.load = original_torch_load
    return recognizer


def pil_to_bgr_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def detect_faces(app, img: Image.Image):
    bgr = pil_to_bgr_array(img)
    return app.get(bgr)


def crop_face_region(img: Image.Image, bbox, margin: float = 0.3, target_size: int = 512) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    fw, fh = x2 - x1, y2 - y1
    x1 = max(0, x1 - fw * margin)
    y1 = max(0, y1 - fh * margin)
    x2 = min(w, x2 + fw * margin)
    y2 = min(h, y2 + fh * margin)
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    return crop.resize((target_size, target_size), Image.LANCZOS)


def get_emotion(recognizer, img: Image.Image, face=None):
    if face is not None and hasattr(face, "bbox"):
        img = crop_face_region(img, face.bbox, margin=0.3, target_size=224)
    arr = np.array(img.convert("RGB"))
    emotion, scores = recognizer.predict_emotions(arr, logits=False)
    return emotion.lower(), float(np.max(scores))


# Mapping from various dataset emotion labels to our canonical 8 classes
_LABEL_MAP = {
    # neutral
    "neutral": "neutral",
    "no expression": "neutral",
    # happy
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    # sad
    "sad": "sad",
    "sadness": "sad",
    # angry
    "angry": "angry",
    "anger": "angry",
    # surprised
    "surprised": "surprised",
    "surprise": "surprised",
    # fearful
    "fearful": "fearful",
    "fear": "fearful",
    # disgusted
    "disgusted": "disgusted",
    "disgust": "disgusted",
    # contempt
    "contempt": "contempt",
}

VALID_EXPRESSIONS = {
    "neutral", "happy", "sad", "angry",
    "surprised", "fearful", "disgusted", "contempt",
}


def normalize_label(label: str) -> str | None:
    return _LABEL_MAP.get(label.lower().strip())


# ---------------------------------------------------------------------------
# Part A: Process real face datasets
# ---------------------------------------------------------------------------

def process_real_dataset(
    source_dir: Path,
    out_dir: Path,
    app,
    recognizer,
    min_face_area_frac: float,
    emotion_conf_threshold: float,
    target_size: int = 512,
    max_per_expression: int | None = None,
) -> list[dict]:
    """
    Walk <source_dir>/<emotion_label>/*.{jpg,png,...} and process each image.
    Returns list of pool metadata records for accepted images.
    """
    records = []
    dataset_name = source_dir.name

    for label_dir in sorted(source_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        canonical = normalize_label(label_dir.name)
        if canonical is None:
            print(f"  Skipping unknown label dir: {label_dir.name}")
            continue

        expr_out = out_dir / canonical
        expr_out.mkdir(parents=True, exist_ok=True)

        image_files = [
            p for p in label_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        if max_per_expression is not None:
            image_files = image_files[:max_per_expression]

        accepted = 0
        for img_path in tqdm(image_files, desc=f"  {dataset_name}/{label_dir.name}", leave=False):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # Face detection
            faces = detect_faces(app, img)
            if len(faces) == 0:
                continue

            # Use the highest-confidence face
            face = max(faces, key=lambda f: f.det_score)
            if face.det_score < 0.5:
                continue

            # Check face area fraction
            w, h = img.size
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            img_area = w * h
            frac = face_area / img_area if img_area > 0 else 0
            if frac < min_face_area_frac:
                continue

            # Emotion verification
            detected_emotion, conf = get_emotion(recognizer, img, face)
            if conf < emotion_conf_threshold:
                continue
            if detected_emotion != canonical:
                continue  # label mismatch

            # Crop and save
            cropped = crop_face_region(img, face.bbox, margin=0.3, target_size=target_size)
            img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()[:8]
            out_name = f"real_{dataset_name}_{img_hash}.png"
            out_path = expr_out / out_name

            if not out_path.exists():
                cropped.save(out_path)

            pool_id = f"{canonical}_{out_name[:-4]}"
            records.append({
                "pool_id": pool_id,
                "path": str(out_path.resolve()),
                "expression": canonical,
                "source": "real",
                "dataset": dataset_name,
                "det_score": float(face.det_score),
                "emotion_conf": float(conf),
                "face_area_frac": float(frac),
            })
            accepted += 1

        print(f"  {dataset_name}/{label_dir.name}: accepted {accepted}/{len(image_files)}")

    return records


# ---------------------------------------------------------------------------
# Part B: Generate Flux synthetic I_r images
# ---------------------------------------------------------------------------

EMOTION_DESCRIPTIONS = {
    "neutral": "completely neutral expressionless face, relaxed facial muscles, no emotion",
    "happy": "genuinely happy smiling expression, raised cheeks, crinkled eyes, joyful",
    "sad": "sad expression, downturned corners of mouth, slightly furrowed brow, melancholic",
    "angry": "angry expression, furrowed brow, tense jaw, intense stare, hostile",
    "surprised": "surprised expression, raised eyebrows, wide open eyes, slightly open mouth",
    "fearful": "fearful expression, wide eyes, raised eyebrows, slight tension, anxious",
    "disgusted": "disgusted expression, wrinkled nose, raised upper lip, aversion",
    "contempt": "contemptuous expression, slight asymmetric smirk, one raised brow",
}


def generate_flux_reference_pool(
    out_dir: Path,
    cfg: dict,
    app,
    recognizer,
    n_per_expression: int,
    emotion_conf_threshold: float,
    min_face_area_frac: float,
    seed: int = 0,
) -> list[dict]:
    """Generate synthetic I_r images using Flux for each expression class."""
    # Import Flux generation utilities from step2 (same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    from step2_generate_images import load_pipeline, generate_image
    from step1_generate_prompts import generate_template_descriptions, _SCENES, _LIGHTING

    gen_cfg = cfg["generation"]
    pipe = load_pipeline(
        model_id=gen_cfg["model_id"],
        hf_token=gen_cfg.get("hf_token", ""),
        device=gen_cfg.get("device", "cuda"),
    )

    rng = random.Random(seed)
    # Generate a large pool of diverse person descriptions (separate from I_e persons)
    # Use offset seed so we get different persons than step1
    n_descs = max(n_per_expression * 2, 200)
    descs = generate_template_descriptions(n_descs, seed=seed + 9999)

    records = []
    global_idx = 0

    for expression, emo_desc in EMOTION_DESCRIPTIONS.items():
        expr_out = out_dir / expression
        expr_out.mkdir(parents=True, exist_ok=True)

        accepted = 0
        attempts = 0
        pbar = tqdm(total=n_per_expression, desc=f"  Flux {expression}", leave=False)

        while accepted < n_per_expression and attempts < n_per_expression * 5:
            attempts += 1
            person_desc, scene, lighting = rng.choice(descs)
            context = f"{scene}, {lighting}" if scene or lighting else ""
            prompt = (
                f"professional photograph of {person_desc}"
                + (f", {context}" if context else "")
                + f", {emo_desc}, photorealistic, sharp focus, 8k"
            )
            gen_seed = seed + global_idx
            global_idx += 1

            try:
                img = generate_image(
                    pipe,
                    prompt=prompt,
                    width=512,
                    height=512,
                    num_steps=gen_cfg["num_steps"],
                    guidance_scale=gen_cfg["guidance_scale"],
                    seed=gen_seed,
                )
            except Exception as e:
                print(f"  Generation failed: {e}")
                continue

            # Validate: single face, correct expression
            faces = detect_faces(app, img)
            if len(faces) == 0:
                continue
            face = max(faces, key=lambda f: f.det_score)
            if face.det_score < 0.5:
                continue

            w, h = img.size
            bbox = face.bbox
            frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
            if frac < min_face_area_frac:
                continue

            detected_emotion, conf = get_emotion(recognizer, img, face)
            if conf < emotion_conf_threshold or detected_emotion != expression:
                continue

            # Crop and save
            cropped = crop_face_region(img, face.bbox, margin=0.3, target_size=512)
            out_name = f"flux_{expression}_{accepted:05d}.png"
            out_path = expr_out / out_name
            cropped.save(out_path)

            pool_id = f"{expression}_flux_{accepted:05d}"
            records.append({
                "pool_id": pool_id,
                "path": str(out_path.resolve()),
                "expression": expression,
                "source": "flux",
                "dataset": "flux_generated",
                "det_score": float(face.det_score),
                "emotion_conf": float(conf),
                "face_area_frac": float(frac),
                "prompt": prompt,
                "seed": gen_seed,
            })
            accepted += 1
            pbar.update(1)

        pbar.close()
        print(f"  Flux {expression}: accepted {accepted}/{attempts} attempts")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build I_r reference pool for expression transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all real datasets found in data/real_datasets/
  python step5_build_reference_pool.py --source real

  # Generate Flux synthetic I_r (500 per expression class)
  python step5_build_reference_pool.py --source flux

  # Both (default)
  python step5_build_reference_pool.py --source both

  # Limit for quick test
  python step5_build_reference_pool.py --source real --max-per-expression 100
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--source", choices=["real", "flux", "both"], default="both",
        help="Which I_r source to process",
    )
    parser.add_argument(
        "--max-per-expression", type=int, default=None,
        help="Max images per expression class (for quick testing)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool_cfg = cfg.get("reference_pool", {})

    pool_dir = Path(cfg["paths"]["reference_pool_dir"])
    pool_metadata_path = Path(cfg["paths"]["reference_pool_metadata"])
    pool_dir.mkdir(parents=True, exist_ok=True)

    min_face_area = pool_cfg.get("min_face_area_frac", 0.10)
    emotion_conf_thresh = pool_cfg.get("emotion_conf_threshold", 0.55)
    n_flux = args.max_per_expression or pool_cfg.get("flux_per_expression", 500)
    real_dataset_root = Path(pool_cfg.get("real_dataset_dir", "data/real_datasets"))

    # Load existing pool metadata to support incremental append
    existing_ids = set()
    existing_records = []
    if pool_metadata_path.exists():
        with jsonlines.open(pool_metadata_path) as reader:
            for rec in reader:
                existing_ids.add(rec["pool_id"])
                existing_records.append(rec)
    print(f"Existing pool entries: {len(existing_records)}")

    print("Loading face detection + expression models...")
    app = load_insightface_app(det_thresh=0.5)
    recognizer = load_hsemotion(cfg.get("filtering", {}).get("emotion_model", "enet_b0_8_best_afew"))
    print("Models loaded.")

    new_records = []

    # ---- Part A: real datasets ----
    if args.source in ("real", "both"):
        if not real_dataset_root.exists():
            print(f"WARNING: real_dataset_dir not found: {real_dataset_root}")
            print("  Create subdirectories like data/real_datasets/affectnet/<emotion>/*.jpg")
        else:
            dataset_dirs = [d for d in real_dataset_root.iterdir() if d.is_dir()]
            if not dataset_dirs:
                print(f"WARNING: No dataset subdirectories found in {real_dataset_root}")
            for dataset_dir in sorted(dataset_dirs):
                print(f"\nProcessing real dataset: {dataset_dir.name}")
                recs = process_real_dataset(
                    source_dir=dataset_dir,
                    out_dir=pool_dir,
                    app=app,
                    recognizer=recognizer,
                    min_face_area_frac=min_face_area,
                    emotion_conf_threshold=emotion_conf_thresh,
                    target_size=512,
                    max_per_expression=args.max_per_expression,
                )
                # Skip already-indexed entries
                recs = [r for r in recs if r["pool_id"] not in existing_ids]
                new_records.extend(recs)
                print(f"  New entries from {dataset_dir.name}: {len(recs)}")

    # ---- Part B: Flux generation ----
    if args.source in ("flux", "both"):
        print("\nGenerating Flux synthetic I_r images...")
        flux_recs = generate_flux_reference_pool(
            out_dir=pool_dir,
            cfg=cfg,
            app=app,
            recognizer=recognizer,
            n_per_expression=n_flux,
            emotion_conf_threshold=emotion_conf_thresh,
            min_face_area_frac=min_face_area,
            seed=args.seed,
        )
        flux_recs = [r for r in flux_recs if r["pool_id"] not in existing_ids]
        new_records.extend(flux_recs)
        print(f"New Flux entries: {len(flux_recs)}")

    # ---- Write metadata ----
    if new_records:
        with jsonlines.open(pool_metadata_path, mode="a") as writer:
            for rec in new_records:
                writer.write(rec)
        print(f"\nAppended {len(new_records)} new entries to {pool_metadata_path}")
    else:
        print("\nNo new entries to write.")

    # ---- Summary ----
    all_records = existing_records + new_records
    from collections import Counter
    expr_counts = Counter(r["expression"] for r in all_records)
    src_counts = Counter(r["source"] for r in all_records)
    print(f"\nPool summary ({len(all_records)} total):")
    for expr in sorted(VALID_EXPRESSIONS):
        print(f"  {expr}: {expr_counts.get(expr, 0)}")
    print(f"  sources: {dict(src_counts)}")
    print(f"Output: {pool_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 5: Build the I_r reference image pool for expression transfer.

Source: InfiniteYou generation using filtered pairs from step4.
  - Randomly pick left or right image from a filtered pair as ID reference
  - Use InfiniteYou to generate a new face image with the target expression
  - Validate with face detection + emotion recognition

Input:
  data/filtered/filter_metadata.jsonl  (step4 output, passed=true only)

Output:
  data/reference_pool/
    happy/
      infu_<idx>.png
      ...
    sad/
      ...
  data/reference_pool/pool_metadata.jsonl
    {"pool_id": "happy_0042", "path": "...", "expression": "happy",
     "source": "infiniteyou", "det_score": 0.95, "emotion_conf": 0.88,
     "face_area_frac": 0.35}
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import jsonlines
import numpy as np
import torch
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


VALID_EXPRESSIONS = {
    "neutral", "happy", "sad", "angry",
    "surprised", "fearful", "disgusted", "contempt",
}


# ---------------------------------------------------------------------------
# Emotion prompt descriptions (for InfiniteYou generation)
# ---------------------------------------------------------------------------

EMOTION_DESCRIPTIONS = {
    "neutral": "completely neutral expressionless face, relaxed facial muscles, no emotion",
    "happy": "happy expression, genuine smile, bright eyes, joyful",
    "sad": "sad expression, downturned corners of mouth, slightly furrowed brow, melancholic",
    "angry": "angry expression, furrowed brows, intense gaze, tense jaw",
    "surprised": "surprised expression, wide eyes, raised eyebrows, open mouth slightly",
    "fearful": "fearful expression, wide eyes, tense face, worried look",
    "disgusted": "disgusted expression, wrinkled nose, curled lip",
    "contempt": "contemptuous expression, slight smirk, raised eyebrow",
}

# Lighting conditions for diversity
LIGHTING_CONDITIONS = [
    "soft natural daylight, diffused sunlight",
    "golden hour warm sunlight, soft shadows",
    "overcast day soft diffused light, even illumination",
    "morning light, gentle warm tones",
    "afternoon sunlight, balanced exposure",
    "soft studio lighting, professional portrait",
    "warm indoor ambient light, cozy atmosphere",
    "cool fluorescent office lighting",
    "window light from the side, natural indoor",
    "mixed indoor lighting, warm and cool tones",
    "dramatic side lighting, strong contrast",
    "rim lighting from behind, silhouette edges",
    "butterfly lighting, classic portrait style",
    "rembrandt lighting, artistic shadows",
    "split lighting, half face illuminated",
    "soft fill light, minimal shadows",
    "high key lighting, bright and even",
    "low key lighting, moody atmosphere",
    "bounce light, soft reflections",
    "ambient room light, natural feel",
]


# ---------------------------------------------------------------------------
# InfiniteYou pipeline setup
# ---------------------------------------------------------------------------

def setup_infiniteyou_pipeline(args):
    """Initialize InfiniteYou pipeline."""
    infiniteyou_path = args.infiniteyou_path
    sys.path.insert(0, infiniteyou_path)
    from pipelines.pipeline_infu_flux import InfUFluxPipeline

    infu_model_path = os.path.join(
        args.model_dir,
        f'infu_flux_{args.infu_flux_version}',
        args.model_version,
    )
    insightface_root_path = os.path.join(
        args.model_dir,
        'supports',
        'insightface',
    )

    print(f"Loading InfiniteYou pipeline...")
    print(f"  Base model: {args.base_model_path}")
    print(f"  InfuNet: {infu_model_path}")
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )
    print("InfiniteYou pipeline loaded.")
    return pipe


def build_prompt(person_description: str, emotion: str, lighting: str) -> str:
    """Build a full generation prompt."""
    emotion_desc = EMOTION_DESCRIPTIONS.get(emotion, emotion)
    return (
        f"professional portrait photograph, "
        f"{person_description}, "
        f"{emotion_desc}, "
        f"{lighting}, "
        f"high quality, photorealistic, sharp focus, 8k"
    )


# ---------------------------------------------------------------------------
# Load filtered pairs from step4
# ---------------------------------------------------------------------------

def load_passing_pairs(filter_metadata_path: Path) -> list[dict]:
    """Load only pairs that passed all filters from step4."""
    passing = []
    with jsonlines.open(filter_metadata_path) as reader:
        for rec in reader:
            if rec.get("passed", False):
                passing.append(rec)
    return passing


def build_expression_to_pairs(pairs: list[dict]) -> dict[str, list[dict]]:
    """
    Index pairs by expression.
    Each pair has left (emotion_left) and right (emotion_right).
    For each expression, collect pairs that have that expression on either side,
    along with which side ('left' or 'right') to use as ID reference.
    """
    expr_pairs: dict[str, list[tuple[dict, str]]] = {}
    for pair in pairs:
        e_left = pair.get("emotion_left", "")
        e_right = pair.get("emotion_right", "")
        if e_left in VALID_EXPRESSIONS:
            expr_pairs.setdefault(e_left, []).append((pair, "left"))
        if e_right in VALID_EXPRESSIONS:
            expr_pairs.setdefault(e_right, []).append((pair, "right"))
    return expr_pairs


# ---------------------------------------------------------------------------
# Generate reference pool using InfiniteYou
# ---------------------------------------------------------------------------

def generate_infiniteyou_reference_pool(
    pipe,
    out_dir: Path,
    pairs: list[dict],
    app,
    recognizer,
    n_per_expression: int,
    emotion_conf_threshold: float,
    min_face_area_frac: float,
    args,
    seed: int = 0,
) -> list[dict]:
    """
    Generate I_r reference images using InfiniteYou.

    For each expression class:
      1. Collect filtered pairs that have that expression
      2. Randomly pick a pair and randomly choose left or right as ID ref
      3. Generate with InfiniteYou using the target expression prompt
      4. Validate face detection + emotion
      5. Save to pool
    """
    rng = random.Random(seed)
    expr_pairs = build_expression_to_pairs(pairs)

    records = []

    for expression in sorted(VALID_EXPRESSIONS):
        expr_out = out_dir / expression
        expr_out.mkdir(parents=True, exist_ok=True)

        # For this expression, we need pairs where ANY side has this emotion
        # so we can use that side's image as ID reference
        candidates = expr_pairs.get(expression, [])
        if not candidates:
            # Fallback: use any pair, pick random side
            candidates = [(p, rng.choice(["left", "right"])) for p in pairs]

        if not candidates:
            print(f"  {expression}: no candidate pairs available, skipping")
            continue

        accepted = 0
        attempts = 0
        max_attempts = n_per_expression * 5
        pbar = tqdm(total=n_per_expression, desc=f"  InfiniteYou {expression}", leave=False)

        while accepted < n_per_expression and attempts < max_attempts:
            attempts += 1

            # Pick a random pair and side
            pair, side = rng.choice(candidates)

            # Load the ID reference image
            if side == "left":
                id_img_path = pair["left_path"]
            else:
                id_img_path = pair["right_path"]

            id_img_path = Path(id_img_path)
            if not id_img_path.exists():
                continue

            try:
                id_image = Image.open(id_img_path).convert("RGB")
            except Exception:
                continue

            # Build prompt with target expression + random lighting
            person_desc = pair.get("person_description", "a person")
            lighting = rng.choice(LIGHTING_CONDITIONS)
            prompt = build_prompt(person_desc, expression, lighting)
            gen_seed = rng.randint(0, 2**32 - 1)

            # Generate with InfiniteYou
            try:
                img = pipe(
                    id_image=id_image,
                    prompt=prompt,
                    control_image=None,
                    width=args.width,
                    height=args.height,
                    seed=gen_seed,
                    guidance_scale=args.guidance_scale,
                    num_steps=args.num_steps,
                    infusenet_conditioning_scale=args.infusenet_conditioning_scale,
                    infusenet_guidance_start=args.infusenet_guidance_start,
                    infusenet_guidance_end=args.infusenet_guidance_end,
                    cpu_offload=args.cpu_offload,
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
            out_name = f"infu_{expression}_{accepted:05d}.png"
            out_path = expr_out / out_name
            cropped.save(out_path)

            pool_id = f"{expression}_infu_{accepted:05d}"
            records.append({
                "pool_id": pool_id,
                "path": str(out_path.resolve()),
                "expression": expression,
                "source": "infiniteyou",
                "dataset": "infiniteyou_generated",
                "det_score": float(face.det_score),
                "emotion_conf": float(conf),
                "face_area_frac": float(frac),
                "prompt": prompt,
                "seed": gen_seed,
                "id_ref_path": str(id_img_path.resolve()),
                "id_ref_side": side,
                "id_ref_pair_id": pair.get("pair_id", ""),
                "id_ref_person_id": pair.get("person_id", ""),
            })
            accepted += 1
            pbar.update(1)

        pbar.close()
        print(f"  InfiniteYou {expression}: accepted {accepted}/{attempts} attempts")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build I_r reference pool using InfiniteYou generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reference pool (default 500 per expression)
  python step5_build_reference_pool.py

  # Quick test with 10 per expression
  python step5_build_reference_pool.py --max-per-expression 10

  # Custom InfiniteYou model path
  python step5_build_reference_pool.py --model-dir /path/to/InfiniteYou
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--max-per-expression", type=int, default=None,
        help="Max images per expression class (overrides config flux_per_expression)",
    )
    parser.add_argument("--seed", type=int, default=0)

    # InfiniteYou model config
    parser.add_argument("--infiniteyou-path", default="/scratch/GenAI_class/f007yzf/InfiniteYou",
                        help="Path to InfiniteYou repo (added to sys.path)")
    parser.add_argument("--base-model-path", default="black-forest-labs/FLUX.1-dev",
                        help="Base diffusion model path")
    parser.add_argument("--model-dir", default="ByteDance/InfiniteYou",
                        help="InfiniteYou model directory")
    parser.add_argument("--infu-flux-version", default="v1.0")
    parser.add_argument("--model-version", default="aes_stage2")

    # GPU
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index")

    # Generation parameters
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--infusenet-conditioning-scale", type=float, default=1.2,
                        help="Identity control strength (1.2 = strong)")
    parser.add_argument("--infusenet-guidance-start", type=float, default=0.0)
    parser.add_argument("--infusenet-guidance-end", type=float, default=1.0)

    # Memory optimization
    parser.add_argument("--quantize-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload")

    # Dry run
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without generating images")

    args = parser.parse_args()

    # Set CUDA device
    torch.cuda.set_device(args.cuda_device)
    print(f"Using GPU: {args.cuda_device}")

    cfg = load_config(args.config)
    pool_cfg = cfg.get("reference_pool", {})

    pool_dir = Path(cfg["paths"]["reference_pool_dir"])
    pool_metadata_path = Path(cfg["paths"]["reference_pool_metadata"])
    pool_dir.mkdir(parents=True, exist_ok=True)

    min_face_area = pool_cfg.get("min_face_area_frac", 0.10)
    emotion_conf_thresh = pool_cfg.get("emotion_conf_threshold", 0.55)
    n_per_expression = args.max_per_expression or pool_cfg.get("flux_per_expression", 500)

    # Load filtered pairs from step4
    filter_meta = Path(cfg["paths"]["filtered_dir"]) / "filter_metadata.jsonl"
    if not filter_meta.exists():
        print(f"ERROR: {filter_meta} not found. Run step4 first.")
        sys.exit(1)

    print(f"Loading filtered pairs from {filter_meta} ...")
    pairs = load_passing_pairs(filter_meta)
    print(f"  Passing pairs: {len(pairs)}")

    if not pairs:
        print("ERROR: No passing pairs found. Cannot build reference pool.")
        sys.exit(1)

    # Show expression distribution in source pairs
    from collections import Counter
    left_emotions = Counter(p.get("emotion_left", "") for p in pairs)
    right_emotions = Counter(p.get("emotion_right", "") for p in pairs)
    print(f"  Left emotions:  {dict(left_emotions)}")
    print(f"  Right emotions: {dict(right_emotions)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would generate:")
        print(f"  {n_per_expression} images x {len(VALID_EXPRESSIONS)} expressions "
              f"= {n_per_expression * len(VALID_EXPRESSIONS)} total")
        print(f"  Using {len(pairs)} filtered pairs as ID references")
        return

    # Load existing pool metadata to support incremental append
    existing_ids = set()
    existing_records = []
    if pool_metadata_path.exists():
        with jsonlines.open(pool_metadata_path) as reader:
            for rec in reader:
                existing_ids.add(rec["pool_id"])
                existing_records.append(rec)
    print(f"Existing pool entries: {len(existing_records)}")

    print("\nLoading face detection + expression models...")
    face_app = load_insightface_app(det_thresh=0.5)
    recognizer = load_hsemotion(
        cfg.get("filtering", {}).get("emotion_model", "enet_b0_8_best_afew")
    )
    print("Validation models loaded.")

    # Initialize InfiniteYou pipeline
    pipe = setup_infiniteyou_pipeline(args)

    # Generate reference pool
    print(f"\nGenerating InfiniteYou reference pool ({n_per_expression} per expression)...")
    new_records = generate_infiniteyou_reference_pool(
        pipe=pipe,
        out_dir=pool_dir,
        pairs=pairs,
        app=face_app,
        recognizer=recognizer,
        n_per_expression=n_per_expression,
        emotion_conf_threshold=emotion_conf_thresh,
        min_face_area_frac=min_face_area,
        args=args,
        seed=args.seed,
    )

    # Deduplicate against existing entries
    new_records = [r for r in new_records if r["pool_id"] not in existing_ids]
    print(f"\nNew InfiniteYou entries: {len(new_records)}")

    # Write metadata
    if new_records:
        with jsonlines.open(pool_metadata_path, mode="a") as writer:
            for rec in new_records:
                writer.write(rec)
        print(f"Appended {len(new_records)} new entries to {pool_metadata_path}")
    else:
        print("No new entries to write.")

    # Summary
    all_records = existing_records + new_records
    expr_counts = Counter(r["expression"] for r in all_records)
    src_counts = Counter(r["source"] for r in all_records)
    print(f"\nPool summary ({len(all_records)} total):")
    for expr in sorted(VALID_EXPRESSIONS):
        print(f"  {expr}: {expr_counts.get(expr, 0)}")
    print(f"  sources: {dict(src_counts)}")
    print(f"Output: {pool_dir}")


if __name__ == "__main__":
    main()

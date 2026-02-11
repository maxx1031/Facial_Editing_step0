#!/usr/bin/env python3
"""
Step 4: Three-layer filtering pipeline.

Layer 1: Face Detection (RetinaFace via insightface)
  - Each side must have exactly 1 face

Layer 2: Identity Consistency (ArcFace via insightface)
  - Cosine similarity >= arcface_threshold (default 0.5)

Layer 3: Emotion Verification (HSEmotion)
  - Left and right must have DIFFERENT emotions

Input:  data/cropped/crop_metadata.jsonl
Output: data/filtered/ (symlinks or copies of passing pairs)
        data/filter_stats.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import jsonlines
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    def expand(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            return os.environ.get(var, obj)
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        return obj
    return expand(raw)


def load_insightface_app(det_thresh: float = 0.5):
    """Load insightface FaceAnalysis app with RetinaFace + ArcFace."""
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
    return app


def load_hsemotion():
    """Load HSEmotion model."""
    from hsemotion.facial_emotions import HSEmotionRecognizer

    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    return recognizer


def pil_to_bgr_array(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to BGR numpy array (for insightface)."""
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1].copy()  # RGB -> BGR


def detect_and_embed(app, img: Image.Image) -> tuple[list, list]:
    """
    Run insightface on image.
    Returns: (faces_list, embeddings_list)
    Each face has .embedding (ArcFace), .bbox, .det_score
    """
    bgr = pil_to_bgr_array(img)
    faces = app.get(bgr)
    embeddings = [f.embedding for f in faces if f.embedding is not None]
    return faces, embeddings


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (n1 * n2))


def get_emotion(recognizer, img: Image.Image) -> tuple[str, float]:
    """
    Run HSEmotion on a PIL image.
    Returns: (emotion_label, confidence)
    """
    arr = np.array(img.convert("RGB"))
    emotion, scores = recognizer.predict_emotions(arr, logits=False)
    # emotion is a string like "Happy", scores is array
    return emotion.lower(), float(np.max(scores))


def filter_pair(
    app,
    recognizer,
    left_img: Image.Image,
    right_img: Image.Image,
    arcface_threshold: float,
    require_exact_one_face: bool,
) -> dict:
    """
    Run all 3 filter layers on a pair.
    Returns dict with pass/fail status and metrics.
    """
    result = {
        "pass_layer1": False,
        "pass_layer2": False,
        "pass_layer3": False,
        "passed_all": False,
        "reject_reason": None,
        "faces_left": 0,
        "faces_right": 0,
        "arcface_similarity": None,
        "emotion_detected_left": None,
        "emotion_detected_right": None,
        "emotion_conf_left": None,
        "emotion_conf_right": None,
    }

    # ---- Layer 1: Face Detection ----
    try:
        faces_l, embs_l = detect_and_embed(app, left_img)
        faces_r, embs_r = detect_and_embed(app, right_img)
    except Exception as e:
        result["reject_reason"] = f"face_detect_error: {e}"
        return result

    result["faces_left"] = len(faces_l)
    result["faces_right"] = len(faces_r)

    if require_exact_one_face:
        if len(faces_l) != 1 or len(faces_r) != 1:
            result["reject_reason"] = (
                f"face_count: left={len(faces_l)}, right={len(faces_r)}"
            )
            return result
    else:
        if len(faces_l) == 0 or len(faces_r) == 0:
            result["reject_reason"] = "no_face_detected"
            return result

    result["pass_layer1"] = True

    # ---- Layer 2: Identity Consistency (ArcFace) ----
    # Use the first (and only, if exact_one) face
    try:
        emb_l = embs_l[0] if embs_l else None
        emb_r = embs_r[0] if embs_r else None

        if emb_l is None or emb_r is None:
            result["reject_reason"] = "arcface_embedding_missing"
            return result

        sim = cosine_similarity(emb_l, emb_r)
        result["arcface_similarity"] = sim

        if sim < arcface_threshold:
            result["reject_reason"] = f"arcface_sim_low: {sim:.3f} < {arcface_threshold}"
            return result

    except Exception as e:
        result["reject_reason"] = f"arcface_error: {e}"
        return result

    result["pass_layer2"] = True

    # ---- Layer 3: Emotion Verification (HSEmotion) ----
    try:
        emo_l, conf_l = get_emotion(recognizer, left_img)
        emo_r, conf_r = get_emotion(recognizer, right_img)

        result["emotion_detected_left"] = emo_l
        result["emotion_detected_right"] = emo_r
        result["emotion_conf_left"] = conf_l
        result["emotion_conf_right"] = conf_r

        if emo_l == emo_r:
            result["reject_reason"] = f"same_emotion: both={emo_l}"
            return result

    except Exception as e:
        result["reject_reason"] = f"emotion_error: {e}"
        return result

    result["pass_layer3"] = True
    result["passed_all"] = True
    return result


def load_existing_filter_results(filter_meta_file: Path) -> set:
    """Load already-processed image_ids."""
    existing = set()
    if filter_meta_file.exists():
        try:
            with jsonlines.open(filter_meta_file) as reader:
                for obj in reader:
                    existing.add(obj["image_id"])
        except Exception:
            pass
    return existing


def compute_stats(results: list) -> dict:
    """Compute filtering statistics."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    pass1 = sum(1 for r in results if r["pass_layer1"])
    pass2 = sum(1 for r in results if r["pass_layer2"])
    pass3 = sum(1 for r in results if r["pass_layer3"])

    sims = [r["arcface_similarity"] for r in results if r["arcface_similarity"] is not None]
    emotion_pairs = [
        (r["emotion_detected_left"], r["emotion_detected_right"])
        for r in results
        if r["emotion_detected_left"] and r["emotion_detected_right"]
    ]

    reject_reasons = {}
    for r in results:
        if r["reject_reason"]:
            key = r["reject_reason"].split(":")[0]
            reject_reasons[key] = reject_reasons.get(key, 0) + 1

    return {
        "total": total,
        "pass_layer1_face_detect": pass1,
        "pass_layer2_arcface": pass2,
        "pass_layer3_emotion": pass3,
        "pass_rate_layer1": pass1 / total if total else 0,
        "pass_rate_layer2": pass2 / pass1 if pass1 else 0,
        "pass_rate_layer3": pass3 / pass2 if pass2 else 0,
        "overall_pass_rate": pass3 / total if total else 0,
        "arcface_similarity_stats": {
            "mean": float(np.mean(sims)) if sims else None,
            "std": float(np.std(sims)) if sims else None,
            "min": float(np.min(sims)) if sims else None,
            "max": float(np.max(sims)) if sims else None,
            "median": float(np.median(sims)) if sims else None,
        },
        "reject_reasons": reject_reasons,
        "emotion_pair_distribution": {},  # filled below
    }


def main():
    parser = argparse.ArgumentParser(description="Three-layer filtering of cropped pairs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute stats from existing filter_metadata.jsonl, don't run new filtering")
    parser.add_argument("--arcface-threshold", type=float, default=None,
                        help="Override arcface_threshold from config")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy passing pairs to filtered/, just write metadata")
    args = parser.parse_args()

    cfg = load_config(args.config)
    filter_cfg = cfg["filtering"]

    arcface_threshold = args.arcface_threshold or filter_cfg["arcface_threshold"]
    require_exact_one_face = filter_cfg["require_exact_one_face"]

    cropped_dir = Path(cfg["paths"]["cropped_dir"])
    filtered_dir = Path(cfg["paths"]["filtered_dir"])
    stats_file = Path(cfg["paths"]["filter_stats"])

    crop_meta_file = cropped_dir / "crop_metadata.jsonl"
    filter_meta_file = filtered_dir / "filter_metadata.jsonl"

    filtered_dir.mkdir(parents=True, exist_ok=True)

    if not crop_meta_file.exists():
        print(f"ERROR: {crop_meta_file} not found. Run step3 first.")
        sys.exit(1)

    # Load crop metadata
    crop_records = []
    with jsonlines.open(crop_meta_file) as reader:
        for obj in reader:
            crop_records.append(obj)

    print(f"Loaded {len(crop_records)} cropped pairs")

    # Stats-only mode: recompute from existing filter metadata
    if args.stats_only:
        if not filter_meta_file.exists():
            print("No filter_metadata.jsonl found. Run without --stats-only first.")
            sys.exit(1)

        all_results = []
        with jsonlines.open(filter_meta_file) as reader:
            for obj in reader:
                all_results.append(obj)

        stats = compute_stats(all_results)
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print("\nFilter Statistics:")
        print(json.dumps(stats, indent=2))
        print(f"\nStats saved to: {stats_file}")
        return

    # Load models
    print("Loading insightface (RetinaFace + ArcFace)...")
    app = load_insightface_app(
        det_thresh=filter_cfg.get("face_det_score_threshold", 0.5)
    )

    print("Loading HSEmotion...")
    recognizer = load_hsemotion()

    # Load existing filter results for checkpoint resumption
    existing_ids = load_existing_filter_results(filter_meta_file)
    print(f"Already filtered: {len(existing_ids)}")

    remaining = [r for r in crop_records if r["image_id"] not in existing_ids]
    print(f"To filter: {len(remaining)}")

    meta_writer = jsonlines.open(filter_meta_file, mode="a")

    all_results = []
    # Load existing results for stats computation
    if filter_meta_file.exists():
        try:
            with jsonlines.open(filter_meta_file) as reader:
                for obj in reader:
                    all_results.append(obj.get("filter_result", {}))
        except Exception:
            pass

    passed_count = 0
    failed_count = 0
    error_count = 0

    for crop_rec in tqdm(remaining, desc="Filtering"):
        image_id = crop_rec["image_id"]
        left_path = Path(crop_rec["left_path"])
        right_path = Path(crop_rec["right_path"])

        if not left_path.exists() or not right_path.exists():
            print(f"\nWARN: Missing image files for {image_id}")
            error_count += 1
            continue

        try:
            left_img = Image.open(left_path).convert("RGB")
            right_img = Image.open(right_path).convert("RGB")
        except Exception as e:
            print(f"\nERROR loading images for {image_id}: {e}")
            error_count += 1
            continue

        filter_result = filter_pair(
            app, recognizer, left_img, right_img,
            arcface_threshold, require_exact_one_face
        )
        all_results.append(filter_result)

        record = {
            **crop_rec,
            "filter_result": filter_result,
            "passed": filter_result["passed_all"],
        }

        meta_writer.write(record)

        if filter_result["passed_all"]:
            passed_count += 1

            # Copy to filtered/ directory
            if not args.no_copy:
                dst_dir = filtered_dir / image_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(left_path, dst_dir / "left.png")
                shutil.copy2(right_path, dst_dir / "right.png")
        else:
            failed_count += 1

    meta_writer.close()

    # Compute and save stats
    stats = compute_stats(all_results)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nFiltering Complete:")
    print(f"  Passed all layers: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Errors: {error_count}")
    print(f"\nFilter Statistics:")
    print(f"  Layer 1 (face detect) pass rate: {stats.get('pass_rate_layer1', 0):.1%}")
    print(f"  Layer 2 (arcface) pass rate:     {stats.get('pass_rate_layer2', 0):.1%}")
    print(f"  Layer 3 (emotion) pass rate:     {stats.get('pass_rate_layer3', 0):.1%}")
    print(f"  Overall pass rate:               {stats.get('overall_pass_rate', 0):.1%}")

    sim_stats = stats.get("arcface_similarity_stats", {})
    if sim_stats.get("mean") is not None:
        print(f"\nArcFace similarity: mean={sim_stats['mean']:.3f}, "
              f"std={sim_stats['std']:.3f}, "
              f"median={sim_stats['median']:.3f}")

    print(f"\nOutput: {filtered_dir}")
    print(f"Stats:  {stats_file}")


if __name__ == "__main__":
    main()

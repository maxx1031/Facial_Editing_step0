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


def resolve_cfg_path(base_dir: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


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
    import torch
    from hsemotion.facial_emotions import HSEmotionRecognizer

    # PyTorch >=2.6 changed torch.load default weights_only=True, which breaks
    # older hsemotion checkpoints that store model objects.
    # Restrict this override to hsemotion init only.
    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = _torch_load_compat
        recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    finally:
        torch.load = original_torch_load

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


def crop_face_region(img: Image.Image, bbox, margin: float = 0.3) -> Image.Image:
    """
    Crop face region from PIL image using insightface bbox with margin.
    bbox: [x1, y1, x2, y2]
    """
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    fw, fh = x2 - x1, y2 - y1
    # Add margin
    x1 = max(0, x1 - fw * margin)
    y1 = max(0, y1 - fh * margin)
    x2 = min(w, x2 + fw * margin)
    y2 = min(h, y2 + fh * margin)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def get_emotion(recognizer, img: Image.Image, face=None) -> tuple[str, float]:
    """
    Run HSEmotion on a face crop.
    If `face` (insightface face object with .bbox) is provided, crop first.
    Returns: (emotion_label, confidence)
    """
    if face is not None and hasattr(face, "bbox"):
        img = crop_face_region(img, face.bbox, margin=0.3)
    arr = np.array(img.convert("RGB"))
    emotion, scores = recognizer.predict_emotions(arr, logits=False)
    return emotion.lower(), float(np.max(scores))


def filter_pair(
    app,
    recognizer,
    left_img: Image.Image,
    right_img: Image.Image,
    arcface_threshold: float,
    require_exact_one_face: bool,
    expected_emotion_left: str | None = None,
    expected_emotion_right: str | None = None,
    face_det_score_threshold: float = 0.5,
    min_face_area_frac: float = 0.08,
    emotion_conf_threshold: float = 0.55,
    require_emotion_match_metadata: bool = True,
    enable_emotion_layer: bool = True,
    reject_on_emotion_error: bool = False,
) -> dict:
    """
    Run all 3 filter layers on a pair.
    Returns dict with pass/fail status and metrics.
    """
    result = {
        "pass_layer1": False,
        "pass_layer2": False,
        "pass_layer3": False,
        "pass_layer4_naturalness": False,
        "passed_all": False,
        "reject_reason": None,
        "emotion_error_msg": None,
        "faces_left": 0,
        "faces_right": 0,
        "det_score_left": None,
        "det_score_right": None,
        "face_area_frac_left": None,
        "face_area_frac_right": None,
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

    f_l = faces_l[0]
    f_r = faces_r[0]

    try:
        if len(faces_l) > 0:
            result["det_score_left"] = float(getattr(f_l, "det_score", 0.0))
            x1, y1, x2, y2 = [float(v) for v in getattr(f_l, "bbox", [0, 0, 0, 0])]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            result["face_area_frac_left"] = float(area / max(1.0, left_img.size[0] * left_img.size[1]))

        if len(faces_r) > 0:
            result["det_score_right"] = float(getattr(f_r, "det_score", 0.0))
            x1, y1, x2, y2 = [float(v) for v in getattr(f_r, "bbox", [0, 0, 0, 0])]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            result["face_area_frac_right"] = float(area / max(1.0, right_img.size[0] * right_img.size[1]))
    except Exception:
        pass

    if result["det_score_left"] is not None and result["det_score_left"] < face_det_score_threshold:
        result["reject_reason"] = f"face_det_score_low_left: {result['det_score_left']:.3f} < {face_det_score_threshold}"
        return result
    if result["det_score_right"] is not None and result["det_score_right"] < face_det_score_threshold:
        result["reject_reason"] = f"face_det_score_low_right: {result['det_score_right']:.3f} < {face_det_score_threshold}"
        return result

    if result["face_area_frac_left"] is not None and result["face_area_frac_left"] < min_face_area_frac:
        result["reject_reason"] = f"face_too_small_left: {result['face_area_frac_left']:.3f} < {min_face_area_frac}"
        return result
    if result["face_area_frac_right"] is not None and result["face_area_frac_right"] < min_face_area_frac:
        result["reject_reason"] = f"face_too_small_right: {result['face_area_frac_right']:.3f} < {min_face_area_frac}"
        return result

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
    if not enable_emotion_layer or recognizer is None:
        result["pass_layer3"] = True
    else:
        try:
            emo_l, conf_l = get_emotion(recognizer, left_img, face=f_l)
            emo_r, conf_r = get_emotion(recognizer, right_img, face=f_r)

            result["emotion_detected_left"] = emo_l
            result["emotion_detected_right"] = emo_r
            result["emotion_conf_left"] = conf_l
            result["emotion_conf_right"] = conf_r

            if conf_l < emotion_conf_threshold:
                result["reject_reason"] = f"emotion_conf_low_left: {conf_l:.3f} < {emotion_conf_threshold}"
                return result
            if conf_r < emotion_conf_threshold:
                result["reject_reason"] = f"emotion_conf_low_right: {conf_r:.3f} < {emotion_conf_threshold}"
                return result

            if require_emotion_match_metadata and expected_emotion_left is not None:
                if emo_l != expected_emotion_left.lower():
                    result["reject_reason"] = f"emotion_mismatch_left: pred={emo_l} expected={expected_emotion_left}"
                    return result

            if require_emotion_match_metadata and expected_emotion_right is not None:
                if emo_r != expected_emotion_right.lower():
                    result["reject_reason"] = f"emotion_mismatch_right: pred={emo_r} expected={expected_emotion_right}"
                    return result

            if emo_l == emo_r:
                result["reject_reason"] = f"same_emotion: both={emo_l}"
                return result

            result["pass_layer3"] = True

        except Exception as e:
            err_msg = f"emotion_error: {e}"
            result["emotion_error_msg"] = err_msg
            result["reject_reason"] = err_msg
            if reject_on_emotion_error:
                return result
            # Fail-open: continue filtering without emotion layer
            result["reject_reason"] = None
            result["pass_layer3"] = True

    # ---- Layer 4: Naturalness / Quality Check ----
    try:
        det_l = result.get("det_score_left") or 0.0
        det_r = result.get("det_score_right") or 0.0

        # If right image det_score drops significantly vs left, face is likely distorted
        if det_l > 0 and det_r > 0:
            det_ratio = det_r / det_l
            result["det_score_ratio_r_over_l"] = det_ratio
            if det_ratio < 0.75:
                result["reject_reason"] = f"naturalness_det_score_drop: ratio={det_ratio:.3f}"
                return result

        # Landmark plausibility: insightface 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        # Check inter-eye distance vs face width ratio — severely distorted faces have abnormal ratios
        if hasattr(f_r, "kps") and f_r.kps is not None and len(f_r.kps) >= 5:
            kps = np.array(f_r.kps, dtype=float)
            eye_dist = np.linalg.norm(kps[0] - kps[1])  # left_eye - right_eye
            rx1, ry1, rx2, ry2 = [float(v) for v in f_r.bbox]
            face_w = max(1.0, rx2 - rx1)
            eye_face_ratio = eye_dist / face_w
            result["eye_face_ratio_right"] = eye_face_ratio
            # Normal human face: eye distance ~30-45% of face bbox width
            if eye_face_ratio < 0.15 or eye_face_ratio > 0.65:
                result["reject_reason"] = f"landmark_abnormal_right: eye_face_ratio={eye_face_ratio:.3f}"
                return result

            # Nose should be roughly between eyes vertically; big deviation = distorted
            nose = kps[2]
            eye_mid = (kps[0] + kps[1]) / 2.0
            nose_offset_x = abs(nose[0] - eye_mid[0]) / face_w
            result["nose_offset_x_right"] = nose_offset_x
            if nose_offset_x > 0.20:
                result["reject_reason"] = f"landmark_abnormal_right: nose_offset_x={nose_offset_x:.3f}"
                return result

    except Exception as e:
        # Don't block on naturalness check failure, just log
        result["naturalness_check_error"] = str(e)

    result["pass_layer4_naturalness"] = True
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

    pass1 = sum(1 for r in results if r.get("pass_layer1"))
    pass2 = sum(1 for r in results if r.get("pass_layer2"))
    pass3 = sum(1 for r in results if r.get("pass_layer3"))
    pass4 = sum(1 for r in results if r.get("pass_layer4_naturalness"))

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
        "pass_layer4_naturalness": pass4,
        "pass_rate_layer1": pass1 / total if total else 0,
        "pass_rate_layer2": pass2 / pass1 if pass1 else 0,
        "pass_rate_layer3": pass3 / pass2 if pass2 else 0,
        "pass_rate_layer4": pass4 / pass3 if pass3 else 0,
        "overall_pass_rate": pass4 / total if total else 0,
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
    cfg_base_dir = Path(args.config).resolve().parent
    filter_cfg = cfg["filtering"]

    arcface_threshold = args.arcface_threshold or filter_cfg["arcface_threshold"]
    require_exact_one_face = filter_cfg["require_exact_one_face"]

    face_det_score_threshold = float(filter_cfg.get("face_det_score_threshold", 0.5))
    min_face_area_frac = float(filter_cfg.get("min_face_area_frac", 0.08))
    emotion_conf_threshold = float(filter_cfg.get("emotion_conf_threshold", 0.55))
    require_emotion_match_metadata = bool(filter_cfg.get("require_emotion_match_metadata", True))
    enable_emotion_layer = bool(filter_cfg.get("enable_emotion_layer", True))
    reject_on_emotion_error = bool(filter_cfg.get("reject_on_emotion_error", False))

    cropped_dir = resolve_cfg_path(cfg_base_dir, cfg["paths"]["cropped_dir"])
    filtered_dir = resolve_cfg_path(cfg_base_dir, cfg["paths"]["filtered_dir"])
    stats_file = resolve_cfg_path(cfg_base_dir, cfg["paths"]["filter_stats"])

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

    recognizer = None
    if enable_emotion_layer:
        print("Loading HSEmotion...")
        try:
            recognizer = load_hsemotion()
        except Exception as e:
            print(f"WARNING: Failed to load HSEmotion, disabling emotion layer. Error: {e}")
            enable_emotion_layer = False

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
            arcface_threshold,
            require_exact_one_face,
            expected_emotion_left=crop_rec.get("emotion_left"),
            expected_emotion_right=crop_rec.get("emotion_right"),
            face_det_score_threshold=face_det_score_threshold,
            min_face_area_frac=min_face_area_frac,
            emotion_conf_threshold=emotion_conf_threshold,
            require_emotion_match_metadata=require_emotion_match_metadata,
            enable_emotion_layer=enable_emotion_layer,
            reject_on_emotion_error=reject_on_emotion_error,
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

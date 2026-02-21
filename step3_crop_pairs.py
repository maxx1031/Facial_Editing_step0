#!/usr/bin/env python3
"""
Step 3: Crop side-by-side images into left/right 512×512 pairs.

Input:  data/raw/{person_id}/{pair_id}/seed_{n}.png  (1056×528)
Output: data/cropped/{pair_id}_s{n}/left.png  (512×512)
        data/cropped/{pair_id}_s{n}/right.png (512×512)
        data/cropped/crop_metadata.jsonl
"""

import argparse
import os
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
            val = os.environ.get(var, obj)  # keep placeholder if not set
            return val
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


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Center-crop a PIL image to size×size."""
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def _edge_white_run(
    img: Image.Image,
    side: str,
    white_thresh: int = 245,
    white_frac_thresh: float = 0.95,
    max_scan: int = 64,
) -> int:
    """Count consecutive near-white columns from one image edge."""
    arr = np.array(img.convert("RGB"))
    col_white_frac = (
        (arr[:, :, 0] >= white_thresh)
        & (arr[:, :, 1] >= white_thresh)
        & (arr[:, :, 2] >= white_thresh)
    ).mean(axis=0)

    seq = col_white_frac[:max_scan] if side == "left" else col_white_frac[::-1][:max_scan]
    run = 0
    for v in seq:
        if v >= white_frac_thresh:
            run += 1
        else:
            break
    return int(run)


def _trim_inner_white_seam(
    left_half: Image.Image,
    right_half: Image.Image,
    target_size: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Trim near-white seam artifacts on inner edges:
      - right edge of left half
      - left edge of right half
    """
    run_left_inner = _edge_white_run(left_half, side="right")
    run_right_inner = _edge_white_run(right_half, side="left")

    # Add small safety margin to remove anti-aliased separator remnants.
    trim_left = run_left_inner + 2 if run_left_inner > 0 else 0
    trim_right = run_right_inner + 2 if run_right_inner > 0 else 0

    lw, lh = left_half.size
    rw, rh = right_half.size

    # Keep enough width for downstream center crop.
    trim_left = min(trim_left, max(0, lw - target_size))
    trim_right = min(trim_right, max(0, rw - target_size))

    if trim_left > 0:
        left_half = left_half.crop((0, 0, lw - trim_left, lh))
    if trim_right > 0:
        right_half = right_half.crop((trim_right, 0, rw, rh))

    return left_half, right_half


def _edge_white_run_rows(
    img: Image.Image,
    side: str,
    white_thresh: int = 245,
    white_frac_thresh: float = 0.95,
    max_scan: int = 64,
) -> int:
    """Count consecutive near-white rows from top or bottom edge."""
    arr = np.array(img.convert("RGB"))
    row_white_frac = (
        (arr[:, :, 0] >= white_thresh)
        & (arr[:, :, 1] >= white_thresh)
        & (arr[:, :, 2] >= white_thresh)
    ).mean(axis=1)

    seq = row_white_frac[:max_scan] if side == "top" else row_white_frac[::-1][:max_scan]
    run = 0
    for v in seq:
        if v >= white_frac_thresh:
            run += 1
        else:
            break
    return int(run)


def _trim_all_white_borders(
    img: Image.Image,
    target_size: int,
    white_thresh: int = 245,
    white_frac_thresh: float = 0.95,
    max_scan: int = 64,
) -> Image.Image:
    """
    Trim near-white borders from all 4 edges (left, right, top, bottom).
    Guarantees output is at least target_size in each dimension.
    """
    w, h = img.size

    run_left = _edge_white_run(img, side="left", white_thresh=white_thresh,
                               white_frac_thresh=white_frac_thresh, max_scan=max_scan)
    run_right = _edge_white_run(img, side="right", white_thresh=white_thresh,
                                white_frac_thresh=white_frac_thresh, max_scan=max_scan)
    run_top = _edge_white_run_rows(img, side="top", white_thresh=white_thresh,
                                   white_frac_thresh=white_frac_thresh, max_scan=max_scan)
    run_bottom = _edge_white_run_rows(img, side="bottom", white_thresh=white_thresh,
                                      white_frac_thresh=white_frac_thresh, max_scan=max_scan)

    # Add safety margin.
    trim_l = run_left + 2 if run_left > 0 else 0
    trim_r = run_right + 2 if run_right > 0 else 0
    trim_t = run_top + 2 if run_top > 0 else 0
    trim_b = run_bottom + 2 if run_bottom > 0 else 0

    # Clamp so we always keep at least target_size in each dimension.
    max_h_trim = max(0, h - target_size)
    max_w_trim = max(0, w - target_size)
    trim_l = min(trim_l, max(0, max_w_trim - trim_r))
    trim_r = min(trim_r, max(0, max_w_trim - trim_l))
    trim_t = min(trim_t, max(0, max_h_trim - trim_b))
    trim_b = min(trim_b, max(0, max_h_trim - trim_t))

    if trim_l == 0 and trim_r == 0 and trim_t == 0 and trim_b == 0:
        return img
    return img.crop((trim_l, trim_t, w - trim_r if trim_r > 0 else w,
                     h - trim_b if trim_b > 0 else h))


def split_and_crop(
    img_path: Path,
    target_size: int = 512,
    gutter_px: int = 12,
) -> tuple[Image.Image, Image.Image]:
    """
    Split a 1056×528 image into left/right halves (528×528 each),
    then center crop to target_size×target_size.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    mid = w // 2
    max_gutter = min(mid - target_size, w - mid - target_size)
    if max_gutter < 0:
        max_gutter = 0
    gutter_px = max(0, min(int(gutter_px), int(max_gutter)))

    if w != 1056 or h != 528:
        # Try to handle different aspect ratios gracefully
        left_half = img.crop((0, 0, mid - gutter_px, h))
        right_half = img.crop((mid + gutter_px, 0, w, h))
    else:
        left_half = img.crop((0, 0, 528 - gutter_px, 528))
        right_half = img.crop((528 + gutter_px, 0, 1056, 528))

    # Adaptive cleanup for variable center separator artifacts.
    left_half, right_half = _trim_inner_white_seam(left_half, right_half, target_size=target_size)

    # Trim outer white borders (left/right outer edges + top/bottom).
    left_half = _trim_all_white_borders(left_half, target_size)
    right_half = _trim_all_white_borders(right_half, target_size)

    left_crop = center_crop(left_half, target_size)
    right_crop = center_crop(right_half, target_size)

    return left_crop, right_crop


def find_raw_images(raw_dir: Path) -> list[Path]:
    """Find all generated PNG files."""
    return sorted(raw_dir.rglob("seed_*.png"))


def get_existing_crops(cropped_dir: Path) -> set:
    """Get set of already-cropped image IDs."""
    existing = set()
    meta_file = cropped_dir / "crop_metadata.jsonl"
    if meta_file.exists():
        try:
            with jsonlines.open(meta_file) as reader:
                for obj in reader:
                    existing.add(obj["image_id"])
        except Exception:
            pass
    return existing


def main():
    parser = argparse.ArgumentParser(description="Crop side-by-side images into pairs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target-size", type=int, default=512,
                        help="Output crop size in pixels (default: 512)")
    parser.add_argument("--gutter-px", type=int, default=12,
                        help="Skip this many pixels on each side of the center split (default: 12)")
    parser.add_argument("--raw-metadata", default=None,
                        help="Path to raw metadata.jsonl for richer metadata")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_base_dir = Path(args.config).resolve().parent
    raw_dir = resolve_cfg_path(cfg_base_dir, cfg["paths"]["raw_dir"])
    cropped_dir = resolve_cfg_path(cfg_base_dir, cfg["paths"]["cropped_dir"])
    cropped_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found. Run step2 first.")
        sys.exit(1)

    # Load raw metadata for richer info (optional)
    raw_meta = {}
    meta_file = resolve_cfg_path(cfg_base_dir, cfg["paths"]["metadata_file"])
    if meta_file.exists():
        with jsonlines.open(meta_file) as reader:
            for obj in reader:
                raw_meta[obj["image_id"]] = obj

    raw_images = find_raw_images(raw_dir)
    if not raw_images:
        print(f"No images found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(raw_images)} raw images")

    existing_crops = get_existing_crops(cropped_dir)
    print(f"Already cropped: {len(existing_crops)}")

    crop_meta_file = cropped_dir / "crop_metadata.jsonl"
    meta_writer = jsonlines.open(crop_meta_file, mode="a")

    processed = 0
    skipped = 0
    errors = 0

    for img_path in tqdm(raw_images, desc="Cropping"):
        # Parse image_id from path: .../person_id/pair_id/seed_N.png
        parts = img_path.parts
        seed_name = img_path.stem  # seed_0, seed_1, ...
        pair_id = img_path.parent.name
        person_id = img_path.parent.parent.name
        seed_idx = int(seed_name.split("_")[1])
        image_id = f"{pair_id}_s{seed_idx}"

        if image_id in existing_crops:
            skipped += 1
            continue

        out_dir = cropped_dir / image_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            left_crop, right_crop = split_and_crop(img_path, args.target_size, args.gutter_px)

            left_path = out_dir / "left.png"
            right_path = out_dir / "right.png"

            left_crop.save(str(left_path))
            right_crop.save(str(right_path))

            # Build metadata
            meta_entry = {
                "image_id": image_id,
                "person_id": person_id,
                "pair_id": pair_id,
                "seed_idx": seed_idx,
                "left_path": str(left_path),
                "right_path": str(right_path),
                "source_path": str(img_path),
                "crop_size": args.target_size,
            }

            # Enrich with raw metadata if available
            if image_id in raw_meta:
                rm = raw_meta[image_id]
                meta_entry.update({
                    "emotion_left": rm.get("emotion_left"),
                    "emotion_right": rm.get("emotion_right"),
                    "person_description": rm.get("person_description"),
                    "prompt_left": rm.get("prompt_left"),
                    "prompt_right": rm.get("prompt_right"),
                    "seed": rm.get("seed"),
                })

            meta_writer.write(meta_entry)
            processed += 1

        except Exception as e:
            print(f"\nERROR cropping {img_path}: {e}")
            errors += 1

    meta_writer.close()

    print(f"\nDone.")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {cropped_dir}")
    print(f"  Metadata: {crop_meta_file}")


if __name__ == "__main__":
    main()

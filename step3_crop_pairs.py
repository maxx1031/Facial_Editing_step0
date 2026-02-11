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


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Center-crop a PIL image to size×size."""
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def split_and_crop(img_path: Path, target_size: int = 512) -> tuple[Image.Image, Image.Image]:
    """
    Split a 1056×528 image into left/right halves (528×528 each),
    then center crop to target_size×target_size.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    if w != 1056 or h != 528:
        # Try to handle different aspect ratios gracefully
        half_w = w // 2
        left_half = img.crop((0, 0, half_w, h))
        right_half = img.crop((half_w, 0, w, h))
    else:
        left_half = img.crop((0, 0, 528, 528))
        right_half = img.crop((528, 0, 1056, 528))

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
    parser.add_argument("--raw-metadata", default=None,
                        help="Path to raw metadata.jsonl for richer metadata")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    cropped_dir = Path(cfg["paths"]["cropped_dir"])
    cropped_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found. Run step2 first.")
        sys.exit(1)

    # Load raw metadata for richer info (optional)
    raw_meta = {}
    meta_file = Path(cfg["paths"]["metadata_file"])
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
            left_crop, right_crop = split_and_crop(img_path, args.target_size)

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

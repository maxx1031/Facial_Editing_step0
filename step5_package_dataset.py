#!/usr/bin/env python3
"""
Step 5: Package filtered pairs into a HuggingFace datasets-compatible format.

Input:  data/filtered/filter_metadata.jsonl
Output: data/dataset/
  - images/  (or as parquet with embedded images)
  - metadata.parquet
  - dataset_card.md
  - stats.json
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
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


def load_filter_metadata(filter_meta_file: Path) -> list:
    """Load all filter metadata, return only passing records."""
    passing = []
    with jsonlines.open(filter_meta_file) as reader:
        for obj in reader:
            if obj.get("passed", False):
                passing.append(obj)
    return passing


def build_dataset_record(rec: dict, dataset_images_dir: Path, idx: int) -> dict:
    """
    Build a flat record for the dataset.
    Copies images to dataset/images/ with sequential names.
    """
    image_id = rec["image_id"]
    left_src = Path(rec["left_path"])
    right_src = Path(rec["right_path"])

    left_dst = dataset_images_dir / f"{image_id}_left.png"
    right_dst = dataset_images_dir / f"{image_id}_right.png"

    if left_src.exists():
        shutil.copy2(left_src, left_dst)
    if right_src.exists():
        shutil.copy2(right_src, right_dst)

    filter_result = rec.get("filter_result", {})

    return {
        "index": idx,
        "image_id": image_id,
        "person_id": rec.get("person_id", ""),
        "pair_id": rec.get("pair_id", ""),
        "seed_idx": rec.get("seed_idx", -1),
        "emotion_left": rec.get("emotion_left", ""),
        "emotion_right": rec.get("emotion_right", ""),
        "emotion_detected_left": filter_result.get("emotion_detected_left", ""),
        "emotion_detected_right": filter_result.get("emotion_detected_right", ""),
        "arcface_similarity": filter_result.get("arcface_similarity"),
        "person_description": rec.get("person_description", ""),
        "left_image_path": str(left_dst),
        "right_image_path": str(right_dst),
        "crop_size": rec.get("crop_size", 512),
    }


def save_as_parquet(records: list, output_path: Path):
    """Save dataset records as a parquet file using pyarrow."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Convert to column-oriented dict
        if not records:
            print("No records to save as parquet")
            return

        columns = {key: [] for key in records[0].keys()}
        for rec in records:
            for k, v in rec.items():
                columns[k].append(v)

        table = pa.table(columns)
        pq.write_table(table, str(output_path))
        print(f"Saved parquet: {output_path}")

    except ImportError:
        print("pyarrow not available, skipping parquet output")


def save_as_jsonl(records: list, output_path: Path):
    """Save dataset records as JSONL."""
    with jsonlines.open(output_path, mode="w") as writer:
        for rec in records:
            writer.write(rec)
    print(f"Saved JSONL: {output_path}")


def compute_dataset_stats(records: list, filter_records: list) -> dict:
    """Compute comprehensive dataset statistics."""
    if not records:
        return {"total_pairs": 0}

    # Emotion pair distribution
    emotion_pairs = Counter()
    for rec in records:
        pair = f"{rec['emotion_left']}→{rec['emotion_right']}"
        emotion_pairs[pair] += 1

    # ArcFace similarity distribution
    sims = [r["arcface_similarity"] for r in records if r["arcface_similarity"] is not None]

    # Persons
    persons = set(r["person_id"] for r in records)

    # Detected emotion accuracy (if prompt emotions match detected)
    correct_l = sum(
        1 for r in records
        if r["emotion_detected_left"] and r["emotion_left"]
        and r["emotion_detected_left"].lower() in r["emotion_left"].lower()
    )
    correct_r = sum(
        1 for r in records
        if r["emotion_detected_right"] and r["emotion_right"]
        and r["emotion_detected_right"].lower() in r["emotion_right"].lower()
    )

    n = len(records)

    return {
        "total_pairs": n,
        "total_persons": len(persons),
        "emotion_pair_distribution": dict(emotion_pairs.most_common()),
        "arcface_similarity": {
            "mean": float(np.mean(sims)) if sims else None,
            "std": float(np.std(sims)) if sims else None,
            "min": float(np.min(sims)) if sims else None,
            "max": float(np.max(sims)) if sims else None,
            "median": float(np.median(sims)) if sims else None,
            "pct_above_0.5": sum(1 for s in sims if s >= 0.5) / len(sims) if sims else None,
            "pct_above_0.6": sum(1 for s in sims if s >= 0.6) / len(sims) if sims else None,
            "pct_above_0.7": sum(1 for s in sims if s >= 0.7) / len(sims) if sims else None,
        },
        "emotion_detection_accuracy": {
            "left_prompt_match_rate": correct_l / n if n else 0,
            "right_prompt_match_rate": correct_r / n if n else 0,
        },
        "total_filtered_in": n,
        "total_filtered_out": len(filter_records) - n,
        "overall_yield_rate": n / len(filter_records) if filter_records else 0,
    }


def generate_dataset_card(stats: dict, output_path: Path):
    """Generate a markdown dataset card."""
    card = f"""# Face Emotion Paired Image Dataset

## Dataset Description

This dataset contains paired portrait photographs showing the same person with different facial expressions,
generated using FLUX.2-klein.

Each sample consists of two 512×512 images:
- **Left image**: Person with emotion A
- **Right image**: Same person with emotion B

## Statistics

- **Total pairs**: {stats.get('total_pairs', 0)}
- **Total unique persons**: {stats.get('total_persons', 0)}
- **Overall yield rate**: {stats.get('overall_yield_rate', 0):.1%}

## Emotion Pair Distribution

| Emotion Pair | Count |
|-------------|-------|
"""
    for pair, count in stats.get("emotion_pair_distribution", {}).items():
        card += f"| {pair} | {count} |\n"

    sim_stats = stats.get("arcface_similarity", {})
    if sim_stats.get("mean") is not None:
        card += f"""
## Identity Consistency (ArcFace)

- Mean similarity: {sim_stats['mean']:.3f}
- Std deviation: {sim_stats['std']:.3f}
- Median: {sim_stats['median']:.3f}
- Pairs with sim ≥ 0.5: {sim_stats['pct_above_0.5']:.1%}
- Pairs with sim ≥ 0.6: {sim_stats['pct_above_0.6']:.1%}
- Pairs with sim ≥ 0.7: {sim_stats['pct_above_0.7']:.1%}
"""

    card += """
## Generation Pipeline

1. Person descriptions generated with GPT-4o
2. Side-by-side images (1056×528) generated with FLUX.2-klein
3. Cropped to 512×512 per side
4. Filtered by:
   - Face detection (RetinaFace via insightface)
   - Identity consistency (ArcFace, threshold ≥ 0.5)
   - Emotion verification (HSEmotion, must differ between sides)

## License

Images are generated by FLUX.2-klein. Subject to FLUX model license terms.
"""

    with open(output_path, "w") as f:
        f.write(card)
    print(f"Dataset card: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Package filtered dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--format", choices=["parquet", "jsonl", "both"], default="both",
                        help="Output format for metadata")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Don't copy images, just write metadata with original paths")
    args = parser.parse_args()

    cfg = load_config(args.config)

    filtered_dir = Path(cfg["paths"]["filtered_dir"])
    dataset_dir = Path(cfg["paths"]["dataset_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)

    filter_meta_file = filtered_dir / "filter_metadata.jsonl"

    if not filter_meta_file.exists():
        print(f"ERROR: {filter_meta_file} not found. Run step4 first.")
        sys.exit(1)

    # Load all records and all passing records
    all_records = []
    with jsonlines.open(filter_meta_file) as reader:
        for obj in reader:
            all_records.append(obj)

    passing_records = [r for r in all_records if r.get("passed", False)]
    print(f"Total filtered records: {len(all_records)}")
    print(f"Passing pairs: {len(passing_records)}")

    if not passing_records:
        print("No passing pairs found. Check filter thresholds.")
        sys.exit(0)

    # Create images directory
    dataset_images_dir = dataset_dir / "images"
    if not args.no_copy_images:
        dataset_images_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset records
    dataset_records = []
    for idx, rec in enumerate(tqdm(passing_records, desc="Packaging")):
        if not args.no_copy_images:
            ds_rec = build_dataset_record(rec, dataset_images_dir, idx)
        else:
            # Use original paths
            filter_result = rec.get("filter_result", {})
            ds_rec = {
                "index": idx,
                "image_id": rec.get("image_id", ""),
                "person_id": rec.get("person_id", ""),
                "pair_id": rec.get("pair_id", ""),
                "seed_idx": rec.get("seed_idx", -1),
                "emotion_left": rec.get("emotion_left", ""),
                "emotion_right": rec.get("emotion_right", ""),
                "emotion_detected_left": filter_result.get("emotion_detected_left", ""),
                "emotion_detected_right": filter_result.get("emotion_detected_right", ""),
                "arcface_similarity": filter_result.get("arcface_similarity"),
                "person_description": rec.get("person_description", ""),
                "left_image_path": rec.get("left_path", ""),
                "right_image_path": rec.get("right_path", ""),
                "crop_size": rec.get("crop_size", 512),
            }
        dataset_records.append(ds_rec)

    # Save metadata
    if args.format in ("parquet", "both"):
        save_as_parquet(dataset_records, dataset_dir / "metadata.parquet")
    if args.format in ("jsonl", "both"):
        save_as_jsonl(dataset_records, dataset_dir / "metadata.jsonl")

    # Compute and save stats
    stats = compute_dataset_stats(dataset_records, all_records)
    stats_path = dataset_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Generate dataset card
    generate_dataset_card(stats, dataset_dir / "dataset_card.md")

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Unique persons: {stats['total_persons']}")
    print(f"  Yield rate: {stats['overall_yield_rate']:.1%}")
    print(f"\nEmotion pair distribution:")
    for pair, count in stats.get("emotion_pair_distribution", {}).items():
        print(f"  {pair}: {count}")

    sim = stats.get("arcface_similarity", {})
    if sim.get("mean") is not None:
        print(f"\nArcFace similarity: mean={sim['mean']:.3f}, median={sim['median']:.3f}")

    print(f"\nOutput: {dataset_dir}")


if __name__ == "__main__":
    main()

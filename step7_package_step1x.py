#!/usr/bin/env python3
"""
Step 7: Package triplets into Step1X dual-image training format.

Step1X training data format (from train_util.py):
  image_dir/          <- target images (I_e_edit), filename = image_key
  metadata.json       <- {image_key: {tags, ref_image_path, ref_image_path_2,
                                       train_resolution}}

  ref_image_path    = I_e (source image to edit)
  ref_image_path_2  = I_r (expression reference)
  tags              = editing instruction (FACS-based or simple)
  image_key.png     = I_e_edit (ground truth target, lives in image_dir)

Step1X toml config (data_configs/step1x_edit.toml):
  [[datasets.subsets]]
  image_dir = "data/dataset_v2/images"
  metadata_file = "data/dataset_v2/metadata.json"

Input:
  data/triplets/triplet_metadata.jsonl

Output:
  data/dataset_v2/
    images/
      p0001_pair00_s0_happy_real_affectnet_abc12.png  (symlink or copy of I_e_edit)
      ...
    metadata.json
    stats.json
    dataset_card.md
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import jsonlines
import yaml
from tqdm import tqdm


SIMPLE_CAPTION_TEMPLATE = (
    "Keep identity, age, gender, skin tone, hair, clothing, background, "
    "and lighting unchanged. Change only the facial expression to {target_emotion}."
)


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


def load_triplets(triplets_path: Path) -> list[dict]:
    triplets = []
    with jsonlines.open(triplets_path) as reader:
        for rec in reader:
            triplets.append(rec)
    return triplets


def safe_image_key(triplet_id: str) -> str:
    """Convert triplet_id to a filesystem-safe image key (used as filename stem)."""
    return triplet_id.replace("/", "_").replace(" ", "_")


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists():
        return
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build_metadata_entry(
    triplet: dict,
    image_key: str,
    resolution: list[int],
    use_facs_caption: bool,
) -> dict:
    """Build a single Step1X metadata entry from a triplet record."""
    if use_facs_caption and triplet.get("facs_instruction"):
        tags = triplet["facs_instruction"]
    else:
        tags = SIMPLE_CAPTION_TEMPLATE.format(
            target_emotion=triplet.get("target_emotion", "the target expression")
        )

    return {
        "tags": tags,
        "ref_image_path": triplet["I_e_path"],
        "ref_image_path_2": triplet["I_r_path"],
        "train_resolution": resolution,
        # Extra metadata (not used by trainer, useful for debugging)
        "source_emotion": triplet.get("source_emotion", ""),
        "target_emotion": triplet.get("target_emotion", ""),
        "person_id": triplet.get("person_id", ""),
        "pair_id": triplet.get("pair_id", ""),
        "I_r_source": triplet.get("I_r_source", ""),
        "arcface_similarity": triplet.get("arcface_similarity"),
    }


def compute_stats(metadata: dict, triplets: list[dict]) -> dict:
    emotion_pairs = Counter(
        f"{t['source_emotion']}→{t['target_emotion']}" for t in triplets
    )
    ir_sources = Counter(t.get("I_r_source", "unknown") for t in triplets)
    persons = {t.get("person_id", "") for t in triplets}
    sims = [t["arcface_similarity"] for t in triplets if t.get("arcface_similarity") is not None]

    import numpy as np
    return {
        "total_triplets": len(triplets),
        "unique_persons": len(persons),
        "emotion_pair_distribution": dict(emotion_pairs.most_common()),
        "I_r_source_distribution": dict(ir_sources),
        "arcface_similarity": {
            "mean": float(np.mean(sims)) if sims else None,
            "median": float(np.median(sims)) if sims else None,
            "min": float(np.min(sims)) if sims else None,
            "max": float(np.max(sims)) if sims else None,
        },
        "facs_instructions_available": sum(1 for t in triplets if t.get("facs_instruction")),
    }


def generate_dataset_card(stats: dict, output_path: Path) -> None:
    card = f"""# Face Expression Transfer Dataset v2

## Task
Given (I_e, I_r), transfer the facial expression of I_r to I_e while preserving all other
identity attributes (background, lighting, clothing, age, gender, skin tone).

## Training Format (Step1X dual-image conditioning)
- `image_dir/` → I_e_edit (ground truth target)
- `ref_image_path` → I_e (source image to edit)
- `ref_image_path_2` → I_r (expression reference, different person/context)
- `tags` → FACS-based or template editing instruction

## Statistics
- **Total triplets**: {stats['total_triplets']}
- **Unique persons**: {stats['unique_persons']}
- **FACS instructions**: {stats['facs_instructions_available']} / {stats['total_triplets']}

## Emotion Pair Distribution
| Pair | Count |
|------|-------|
"""
    for pair, cnt in stats.get("emotion_pair_distribution", {}).items():
        card += f"| {pair} | {cnt} |\n"

    card += f"""
## I_r Source Distribution
{stats.get('I_r_source_distribution', {})}

## Identity Consistency (ArcFace, I_e ↔ I_e_edit)
{stats.get('arcface_similarity', {})}

## Generation Pipeline
1. (I_e, I_e_edit) pairs: side-by-side Flux generation, step4 identity filter
2. I_r pool: real datasets (AffectNet/RAF-DB) + Flux synthetic, labeled by expression
3. Triplets: K={stats.get('k_ir_per_pair', '?')} I_r per pair, expression-matched
4. Instructions: FACS AU delta (step8) or simple template caption
"""
    output_path.write_text(card)
    print(f"Dataset card: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Package triplets into Step1X dual-image training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step7_package_step1x.py                  # full packaging with copies
  python step7_package_step1x.py --symlink         # symlinks instead of copies (faster)
  python step7_package_step1x.py --use-facs        # use FACS captions (requires step8 first)
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--symlink", action="store_true",
        help="Use symlinks for I_e_edit images instead of copies (saves disk space)",
    )
    parser.add_argument(
        "--use-facs", action="store_true",
        help="Use FACS-based captions from step8 (falls back to template if not available)",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[512, 512],
        metavar=("H", "W"),
        help="Training resolution [H W] (default: 512 512)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    triplets_path = Path(cfg["paths"]["triplets_metadata"])
    dataset_v2_dir = Path(cfg["paths"]["dataset_v2_dir"])

    if not triplets_path.exists():
        print(f"ERROR: {triplets_path} not found. Run step6 (and optionally step8) first.")
        sys.exit(1)

    images_dir = dataset_v2_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading triplets from {triplets_path} ...")
    triplets = load_triplets(triplets_path)
    print(f"  Total triplets: {len(triplets)}")

    resolution = args.resolution  # [H, W]

    metadata = {}
    skipped_missing = 0

    for triplet in tqdm(triplets, desc="Packaging"):
        # Validate all three paths exist
        for key in ("I_e_path", "I_r_path", "I_e_edit_path"):
            if not Path(triplet[key]).exists():
                skipped_missing += 1
                break
        else:
            image_key = safe_image_key(triplet["triplet_id"])
            target_img_path = images_dir / f"{image_key}.png"

            # Link/copy I_e_edit as the target image
            link_or_copy(Path(triplet["I_e_edit_path"]), target_img_path, use_symlink=args.symlink)

            metadata[image_key] = build_metadata_entry(
                triplet, image_key, resolution, use_facs_caption=args.use_facs
            )

    if skipped_missing > 0:
        print(f"WARNING: Skipped {skipped_missing} triplets with missing image files")

    # Write metadata.json
    meta_path = dataset_v2_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata.json: {len(metadata)} entries → {meta_path}")

    # Stats + dataset card
    stats = compute_stats(metadata, triplets)
    stats["k_ir_per_pair"] = cfg.get("reference_pool", {}).get("k_ir_per_pair", 3)
    stats_path = dataset_v2_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats: {stats_path}")

    generate_dataset_card(stats, dataset_v2_dir / "dataset_card.md")

    print(f"\nDataset v2 ready at: {dataset_v2_dir}")
    print(f"  images/: {len(metadata)} target images")
    print(f"  metadata.json: Step1X dual-image format")
    print()
    print("Use in Step1X toml config:")
    print("  [[datasets.subsets]]")
    print(f"  image_dir = \"{images_dir}\"")
    print(f"  metadata_file = \"{meta_path}\"")


if __name__ == "__main__":
    main()

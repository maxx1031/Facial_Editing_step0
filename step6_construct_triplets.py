#!/usr/bin/env python3
"""
Step 6: Construct (I_e, I_r, I_e_edit) training triplets.

For each passing (I_e, I_e_edit) pair from step4:
  - I_e   = left crop  (source expression E1)
  - I_e_edit = right crop (target expression E2, ground truth)
  - Sample K I_r candidates from the reference pool where expression == E2

This turns each filtered pair into K training triplets (K-fold data augmentation).

Input:
  data/filtered/filter_metadata.jsonl     (step4 output, passed=true only)
  data/reference_pool/pool_metadata.jsonl (step5 output)

Output:
  data/triplets/triplet_metadata.jsonl
    {
      "triplet_id": "p0001_pair00_s0_ir042",
      "I_e_path":      "/abs/path/to/left.png",
      "I_r_path":      "/abs/path/to/pool/happy/real_affectnet_abc12.png",
      "I_e_edit_path": "/abs/path/to/right.png",
      "source_emotion":  "neutral",
      "target_emotion":  "happy",
      "person_id":       "p0001",
      "pair_id":         "p0001_pair00",
      "image_id":        "p0001_pair00_s0",
      "arcface_similarity": 0.72,
      "pool_id":         "happy_real_affectnet_abc12",
      "I_r_source":      "real",
      "I_r_emotion_conf": 0.88,
    }
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import jsonlines
import yaml
from tqdm import tqdm


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


def load_passing_pairs(filter_metadata_path: Path) -> list[dict]:
    """Load only pairs that passed all filters from step4."""
    passing = []
    with jsonlines.open(filter_metadata_path) as reader:
        for rec in reader:
            if rec.get("passed", False):
                passing.append(rec)
    return passing


def load_reference_pool(pool_metadata_path: Path) -> dict[str, list[dict]]:
    """Load reference pool, indexed by expression label."""
    pool: dict[str, list[dict]] = defaultdict(list)
    with jsonlines.open(pool_metadata_path) as reader:
        for rec in reader:
            expr = rec.get("expression")
            if expr:
                pool[expr].append(rec)
    return dict(pool)


def load_existing_triplets(triplets_path: Path) -> set[str]:
    """Return set of already-generated triplet_ids for incremental append."""
    if not triplets_path.exists():
        return set()
    existing = set()
    with jsonlines.open(triplets_path) as reader:
        for rec in reader:
            existing.add(rec["triplet_id"])
    return existing


def build_triplets(
    pairs: list[dict],
    pool: dict[str, list[dict]],
    k: int,
    rng: random.Random,
    existing_ids: set[str],
) -> list[dict]:
    """
    For each passing pair, sample K I_r from the pool where expression matches.
    Returns list of new (not already existing) triplet records.
    """
    triplets = []
    skipped_no_pool = 0

    for pair in tqdm(pairs, desc="Constructing triplets"):
        target_expr = pair.get("emotion_right")
        if not target_expr:
            continue

        candidates = pool.get(target_expr, [])
        if not candidates:
            skipped_no_pool += 1
            continue

        # Sample K candidates (with replacement if pool is small)
        sample_size = min(k, len(candidates))
        sampled = rng.sample(candidates, sample_size)
        if sample_size < k:
            # Fill up to K with replacement
            sampled = sampled + rng.choices(candidates, k=k - sample_size)

        filter_result = pair.get("filter_result", {})
        image_id = pair["image_id"]

        for ir_idx, ir_rec in enumerate(sampled):
            triplet_id = f"{image_id}_{ir_rec['pool_id']}"
            if triplet_id in existing_ids:
                continue

            triplets.append({
                "triplet_id": triplet_id,
                "I_e_path": str(Path(pair["left_path"]).resolve()),
                "I_r_path": ir_rec["path"],
                "I_e_edit_path": str(Path(pair["right_path"]).resolve()),
                "source_emotion": pair.get("emotion_left", ""),
                "target_emotion": target_expr,
                "person_id": pair.get("person_id", ""),
                "pair_id": pair.get("pair_id", ""),
                "image_id": image_id,
                "arcface_similarity": filter_result.get("arcface_similarity"),
                "pool_id": ir_rec["pool_id"],
                "I_r_source": ir_rec.get("source", ""),
                "I_r_emotion_conf": ir_rec.get("emotion_conf"),
                "I_r_det_score": ir_rec.get("det_score"),
            })

    if skipped_no_pool > 0:
        print(f"  Skipped {skipped_no_pool} pairs: no I_r candidates in pool for target expression")

    return triplets


def print_summary(triplets: list[dict]) -> None:
    from collections import Counter
    emotion_pairs = Counter(
        f"{t['source_emotion']}→{t['target_emotion']}" for t in triplets
    )
    ir_sources = Counter(t["I_r_source"] for t in triplets)
    persons = {t["person_id"] for t in triplets}

    print(f"\nTriplet summary ({len(triplets)} total):")
    print(f"  Unique persons: {len(persons)}")
    print(f"  I_r sources: {dict(ir_sources)}")
    print("  Emotion pair distribution:")
    for pair, cnt in emotion_pairs.most_common():
        print(f"    {pair}: {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Construct (I_e, I_r, I_e_edit) training triplets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step6_construct_triplets.py              # use config default K=3
  python step6_construct_triplets.py --k 5        # 5 I_r samples per pair
  python step6_construct_triplets.py --k 1        # minimal for quick test
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--k", type=int, default=None,
        help="Number of I_r samples per (I_e, I_e_edit) pair (overrides config)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool_cfg = cfg.get("reference_pool", {})

    k = args.k or pool_cfg.get("k_ir_per_pair", 3)

    filter_meta = Path(cfg["paths"]["filtered_dir"]) / "filter_metadata.jsonl"
    pool_meta = Path(cfg["paths"]["reference_pool_metadata"])
    triplets_dir = Path(cfg["paths"]["triplets_dir"])
    triplets_path = Path(cfg["paths"]["triplets_metadata"])

    triplets_dir.mkdir(parents=True, exist_ok=True)

    if not filter_meta.exists():
        print(f"ERROR: {filter_meta} not found. Run step4 first.")
        sys.exit(1)
    if not pool_meta.exists():
        print(f"ERROR: {pool_meta} not found. Run step5 first.")
        sys.exit(1)

    print(f"Loading filtered pairs from {filter_meta} ...")
    pairs = load_passing_pairs(filter_meta)
    print(f"  Passing pairs: {len(pairs)}")

    print(f"Loading reference pool from {pool_meta} ...")
    pool = load_reference_pool(pool_meta)
    total_pool = sum(len(v) for v in pool.values())
    print(f"  Pool total: {total_pool}")
    for expr in sorted(pool):
        print(f"    {expr}: {len(pool[expr])}")

    print(f"Loading existing triplets (for incremental append) ...")
    existing_ids = load_existing_triplets(triplets_path)
    print(f"  Existing triplets: {len(existing_ids)}")

    rng = random.Random(args.seed)
    print(f"\nBuilding triplets (K={k} per pair) ...")
    triplets = build_triplets(pairs, pool, k=k, rng=rng, existing_ids=existing_ids)
    print(f"  New triplets generated: {len(triplets)}")

    if triplets:
        with jsonlines.open(triplets_path, mode="a") as writer:
            for t in triplets:
                writer.write(t)
        print(f"Appended {len(triplets)} triplets to {triplets_path}")
    else:
        print("No new triplets to write.")

    # Re-read full file for summary
    all_triplets = []
    if triplets_path.exists():
        with jsonlines.open(triplets_path) as reader:
            all_triplets = list(reader)
    print_summary(all_triplets)


if __name__ == "__main__":
    main()

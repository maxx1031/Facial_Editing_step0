#!/usr/bin/env python3
"""
Step 2: Generate side-by-side face emotion images using FLUX.2-klein.

Model: black-forest-labs/FLUX.2-klein-4B  (Apache 2.0, ~8GB VRAM, 4-step inference)
       black-forest-labs/FLUX.2-klein-9B  (Non-commercial, ~16GB VRAM)

Pipeline class: Flux2Pipeline (diffusers >= 0.36), falls back to DiffusionPipeline.

Input:  data/prompts.jsonl
Output: data/raw/{person_id}/{pair_id}/seed_{n}.png
        data/raw/metadata.jsonl

Supports checkpoint resumption: skips already-generated images.
Memory optimizations: bf16, CPU offload.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import jsonlines
import torch
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    def expand(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            return os.environ.get(var, "")  # return empty string if not set (optional vars)
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        return obj
    return expand(raw)


def load_pipeline(model_id: str, hf_token: str, device: str, pipeline_class: str = "Flux2Pipeline"):
    """
    Load FLUX.2-klein pipeline with memory optimizations.

    Supported pipeline_class values:
      - "Flux2Pipeline"       for FLUX.2-klein-4B / 9B (diffusers >= 0.36)
      - "FluxPipeline"        fallback for FLUX.1-dev
      - "DiffusionPipeline"   generic auto-detect fallback
    """
    import diffusers

    PipelineClass = getattr(diffusers, pipeline_class, None)
    if PipelineClass is None:
        # Fallback chain: try DiffusionPipeline as generic loader
        print(f"WARNING: diffusers {diffusers.__version__} does not have '{pipeline_class}'.")
        for fallback in ["DiffusionPipeline"]:
            PipelineClass = getattr(diffusers, fallback, None)
            if PipelineClass is not None:
                print(f"  Using fallback: {fallback}")
                pipeline_class = fallback
                break
        if PipelineClass is None:
            raise ImportError(
                f"diffusers does not have '{pipeline_class}' or any fallback. "
                "Please upgrade: pip install -U diffusers"
            )

    print(f"Loading {pipeline_class} from {model_id}...")
    print("  dtype=bfloat16, device=cuda")

    # HF token is optional for Apache-licensed 4B variant
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False  # Disable meta device to avoid tensor issues
    )
    if hf_token and not hf_token.startswith("${"):
        load_kwargs["token"] = hf_token

    pipe = PipelineClass.from_pretrained(model_id, **load_kwargs)

    # Move to GPU directly (no CPU offload needed with 49GB VRAM)
    pipe = pipe.to("cuda")

    # Additional memory savings if available
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    if torch.cuda.is_available():
        print(f"  Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")
    else:
        print("  Pipeline loaded (CPU mode).")
    return pipe


def generate_image(
    pipe,
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    """Generate a single image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    # Flux2Pipeline expects prompt as a list of strings (batch format)
    # Convert single string to list
    if isinstance(prompt, str):
        prompt = [prompt]

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        max_sequence_length=512,
    )
    return result.images[0]


def load_prompts(prompts_file: Path) -> list:
    """Load all prompt records."""
    records = []
    with jsonlines.open(prompts_file) as reader:
        for obj in reader:
            records.append(obj)
    return records


def get_existing_images(raw_dir: Path) -> set:
    """Scan raw_dir for already-generated images, return set of output paths."""
    existing = set()
    for png in raw_dir.rglob("seed_*.png"):
        existing.add(str(png))
    return existing


def count_total_jobs(records: list, seeds_per_pair: int) -> int:
    total = 0
    for rec in records:
        total += len(rec["pairs"]) * seeds_per_pair
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate images with FLUX.2-klein")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override seeds_per_pair from config")
    parser.add_argument("--person-ids", nargs="+", default=None,
                        help="Only process specific person IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without running")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg["generation"]
    seeds_per_pair = args.seeds or gen_cfg["seeds_per_pair"]

    prompts_file = Path(cfg["paths"]["prompts_file"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    metadata_file = Path(cfg["paths"]["metadata_file"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not prompts_file.exists():
        print(f"ERROR: {prompts_file} not found. Run step1 first.")
        sys.exit(1)

    records = load_prompts(prompts_file)

    # Filter to specific persons if requested
    if args.person_ids:
        records = [r for r in records if r["person_id"] in args.person_ids]
        print(f"Filtered to {len(records)} persons: {args.person_ids}")

    existing_images = get_existing_images(raw_dir)
    total_jobs = count_total_jobs(records, seeds_per_pair)
    print(f"Total jobs: {total_jobs}")
    print(f"Already done: {len(existing_images)}")

    if args.dry_run:
        skipped = 0
        to_run = 0
        for rec in records:
            for pair in rec["pairs"]:
                for seed_idx in range(seeds_per_pair):
                    seed = seed_idx * 1000 + hash(pair["pair_id"]) % 1000
                    out_path = raw_dir / rec["person_id"] / pair["pair_id"] / f"seed_{seed_idx}.png"
                    if str(out_path) in existing_images:
                        skipped += 1
                    else:
                        to_run += 1
        print(f"Dry run: {to_run} to generate, {skipped} to skip")
        return

    # Load pipeline
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Generation will be extremely slow on CPU.")

    pipe = load_pipeline(
        gen_cfg["model_id"],
        gen_cfg.get("hf_token", ""),
        gen_cfg["device"],
        pipeline_class=gen_cfg.get("pipeline_class", "Flux2Pipeline"),
    )

    # Open metadata file in append mode
    metadata_writer = jsonlines.open(metadata_file, mode="a")

    # Load existing metadata pair_ids to avoid duplicate entries
    existing_meta_ids = set()
    if metadata_file.exists():
        try:
            with jsonlines.open(metadata_file) as r:
                for obj in r:
                    existing_meta_ids.add(obj.get("image_id", ""))
        except Exception:
            pass

    generated_count = 0
    skipped_count = 0
    error_count = 0

    with tqdm(total=total_jobs, desc="Generating images") as pbar:
        for rec in records:
            person_id = rec["person_id"]
            person_desc = rec["person_description"]

            for pair in rec["pairs"]:
                pair_id = pair["pair_id"]
                pair_dir = raw_dir / person_id / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)

                for seed_idx in range(seeds_per_pair):
                    # Use deterministic seeds based on pair_id hash + index
                    seed = (seed_idx * 7919 + abs(hash(pair_id))) % (2**31)
                    out_path = pair_dir / f"seed_{seed_idx}.png"
                    image_id = f"{pair_id}_s{seed_idx}"

                    pbar.set_postfix({"person": person_id, "pair": pair_id[-6:], "seed": seed_idx})

                    if str(out_path) in existing_images:
                        skipped_count += 1
                        pbar.update(1)
                        continue

                    try:
                        img = generate_image(
                            pipe,
                            prompt=pair["combined_prompt"],
                            width=gen_cfg["width"],
                            height=gen_cfg["height"],
                            num_steps=gen_cfg["num_steps"],
                            guidance_scale=gen_cfg["guidance_scale"],
                            seed=seed,
                        )
                        img.save(str(out_path), format="PNG", optimize=False)

                        # Write metadata
                        if image_id not in existing_meta_ids:
                            metadata_writer.write({
                                "image_id": image_id,
                                "person_id": person_id,
                                "pair_id": pair_id,
                                "seed_idx": seed_idx,
                                "seed": seed,
                                "emotion_left": pair["emotion_left"],
                                "emotion_right": pair["emotion_right"],
                                "prompt": pair["combined_prompt"],
                                "prompt_left": pair["prompt_left"],
                                "prompt_right": pair["prompt_right"],
                                "person_description": person_desc,
                                "width": gen_cfg["width"],
                                "height": gen_cfg["height"],
                                "path": str(out_path),
                            })
                            existing_meta_ids.add(image_id)

                        generated_count += 1
                        existing_images.add(str(out_path))

                    except Exception as e:
                        import traceback
                        print(f"\nERROR generating {image_id}: {e}")
                        traceback.print_exc()
                        error_count += 1

                    pbar.update(1)

                # Free memory periodically
                if generated_count % 10 == 0 and generated_count > 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    metadata_writer.close()

    print(f"\nDone.")
    print(f"  Generated: {generated_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {raw_dir}")
    print(f"  Metadata: {metadata_file}")


if __name__ == "__main__":
    main()

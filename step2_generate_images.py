#!/usr/bin/env python3
"""
Step 2: Generate side-by-side face emotion images using FLUX.2-klein.

Model: black-forest-labs/FLUX.2-klein-4B  (Apache 2.0, ~8GB VRAM, 4-step inference)
       black-forest-labs/FLUX.2-klein-9B  (Non-commercial, ~16GB VRAM)

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


def _prefer_site_packages_diffusers() -> None:
    """
    Some local environments inject a dev diffusers repo path (e.g.
    repos/diffusers_step1x_v1p2/src) into sys.path, which can break FLUX.2-klein
    inference due to API/template mismatches. Remove it so imports resolve to the
    conda site-packages build.
    """
    bad_markers = ("/repos/diffusers_step1x_v1p2/src",)
    cleaned = []
    removed = []
    for p in sys.path:
        if any(m in p for m in bad_markers):
            removed.append(p)
            continue
        cleaned.append(p)
    if removed:
        sys.path[:] = cleaned
        for name in list(sys.modules.keys()):
            if name == "diffusers" or name.startswith("diffusers."):
                del sys.modules[name]
        print("  Removed injected diffusers dev path(s):")
        for p in removed:
            print(f"    - {p}")


def _patch_flux2_chat_format_input() -> None:
    """
    Compat patch for some Flux2/tokenizer combinations where the tokenizer's
    chat template expects message['content'] to be a plain string (legacy style),
    while diffusers 0.36.0 Flux2 `format_input` builds multimodal-style lists.
    """
    try:
        from diffusers.pipelines.flux2 import pipeline_flux2 as _pflux2
    except Exception as e:
        print(f"  WARNING: Flux2 chat-format patch NOT applied ({e}). Text encoding may be broken.")
        return

    if getattr(_pflux2, "_patched_legacy_text_format_input", False):
        return

    def _legacy_text_only_format_input(prompts, system_message=_pflux2.SYSTEM_MESSAGE, images=None):
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]
        # Text-only generation path used in this project.
        if images is None or len(images) == 0:
            return [
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
                for prompt in cleaned_txt
            ]

        # Fallback for image-conditioned cases: keep text prompt as plain string.
        messages = [[{"role": "system", "content": system_message}] for _ in cleaned_txt]
        for i, (el, imgs) in enumerate(zip(messages, images)):
            if imgs is not None:
                # Keep multimodal payload for images if provided by caller.
                el.append({"role": "user", "content": [{"type": "image", "image": im} for im in imgs]})
            el.append({"role": "user", "content": cleaned_txt[i]})
        return messages

    _pflux2.format_input = _legacy_text_only_format_input
    _pflux2._patched_legacy_text_format_input = True
    print("  Applied Flux2 chat-format compatibility patch (string content).")


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

    Uses DiffusionPipeline.from_pretrained with device_map for FLUX.2-klein
    as per official example: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    """
    _prefer_site_packages_diffusers()
    from diffusers import DiffusionPipeline
    import diffusers as _diffusers_mod
    _patch_flux2_chat_format_input()

    print(f"Loading DiffusionPipeline from {model_id}...")
    print("  dtype=bfloat16, device_map=cuda")
    print(f"  diffusers import path: {_diffusers_mod.__file__}")

    # Try device_map first (memory-efficient path), then fall back to a safer
    # load path if local diffusers/accelerate stack hits meta-tensor dispatch bugs.
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    if hf_token and not hf_token.startswith("${"):
        load_kwargs["token"] = hf_token

    try:
        try:
            # Default path: let diffusers read class from model_index.json
            pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        except AttributeError as e:
            # Compatibility fallback: some diffusers versions expose Flux2Pipeline
            # but not Flux2KleinPipeline referenced by the checkpoint metadata.
            if "Flux2KleinPipeline" not in str(e):
                raise
            from diffusers import Flux2Pipeline
            print("  Falling back to Flux2Pipeline (Flux2KleinPipeline not available in current diffusers).")
            pipe = Flux2Pipeline.from_pretrained(model_id, **load_kwargs)
    except NotImplementedError as e:
        if "meta tensor" not in str(e):
            raise
        print("  Detected meta-tensor dispatch issue; retrying without device_map.")
        retry_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        if hf_token and not hf_token.startswith("${"):
            retry_kwargs["token"] = hf_token
        from diffusers import Flux2Pipeline
        pipe = Flux2Pipeline.from_pretrained(model_id, **retry_kwargs)
        if device and device.startswith("cuda") and torch.cuda.is_available():
            pipe = pipe.to(device)

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
    device: str = 'cuda',
) -> Image.Image:
    """Generate a single image."""
    generator = torch.Generator(device=device).manual_seed(seed)
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
                    seed = (seed_idx * 7919 + abs(hash(pair["pair_id"]))) % (2**31)
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
        pipeline_class=gen_cfg.get("pipeline_class", "Flux2KleinPipeline"),
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
    total_gen_time = 0.0
    gen_times = []
    import time as time_module
    pipeline_start_time = time_module.time()

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
                        gen_start = time_module.time()
                        img = generate_image(
                            pipe,
                            prompt=pair["combined_prompt"],
                            width=gen_cfg["width"],
                            height=gen_cfg["height"],
                            num_steps=gen_cfg["num_steps"],
                            guidance_scale=gen_cfg["guidance_scale"],
                            seed=seed,
                            device=gen_cfg["device"],
                        )
                        gen_end = time_module.time()
                        gen_duration = gen_end - gen_start
                        total_gen_time += gen_duration
                        gen_times.append(gen_duration)

                        # Diagnose black image
                        import numpy as np
                        arr = np.array(img)
                        if arr.max() < 5:
                            print(f"\n  [DIAG] BLACK IMAGE {image_id}: mean={arr.mean():.3f} max={arr.max()} min={arr.min()}")
                        img.save(str(out_path), format="PNG", optimize=False)

                        # Write metadata with timing info
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
                                "generation_time_sec": round(gen_duration, 3),
                                "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
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

    # Calculate timing statistics
    pipeline_end_time = time_module.time()
    total_elapsed = pipeline_end_time - pipeline_start_time

    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"  Generated: {generated_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {raw_dir}")
    print(f"  Metadata: {metadata_file}")

    if generated_count > 0:
        avg_time = total_gen_time / generated_count
        min_time = min(gen_times) if gen_times else 0
        max_time = max(gen_times) if gen_times else 0

        print(f"\n{'='*60}")
        print(f"Timing Statistics:")
        print(f"{'='*60}")
        print(f"  Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  Total generation time: {total_gen_time:.1f}s")
        print(f"  Average per image: {avg_time:.2f}s")
        print(f"  Min/Max per image: {min_time:.2f}s / {max_time:.2f}s")
        print(f"  Throughput: {generated_count / total_elapsed * 60:.1f} images/min")

        # Estimate time for larger datasets
        print(f"\n{'='*60}")
        print(f"Time Estimates for Larger Datasets:")
        print(f"{'='*60}")
        for target in [1000, 5000, 10000, 20000, 50000]:
            est_hours = (target * avg_time) / 3600
            print(f"  {target:>6,} images: {est_hours:.1f} hours ({est_hours/24:.1f} days)")

        # Save timing stats to file
        stats_file = Path("data/generation_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        stats = {
            "generated_count": generated_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "total_elapsed_sec": round(total_elapsed, 2),
            "total_generation_sec": round(total_gen_time, 2),
            "avg_time_per_image_sec": round(avg_time, 3),
            "min_time_per_image_sec": round(min_time, 3),
            "max_time_per_image_sec": round(max_time, 3),
            "throughput_per_min": round(generated_count / total_elapsed * 60, 2),
            "model_id": gen_cfg["model_id"],
            "num_steps": gen_cfg["num_steps"],
            "width": gen_cfg["width"],
            "height": gen_cfg["height"],
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {stats_file}")


if __name__ == "__main__":
    main()

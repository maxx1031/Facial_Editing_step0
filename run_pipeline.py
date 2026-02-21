#!/usr/bin/env python3
"""
End-to-end pipeline runner with checkpoint resumption (v2 flow).

Default flow (recommended):
  1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 8

Where:
  7 = MLLM FACS instruction generation (optional, requires --model-path)
  8 = Step1X packaging

Usage:
  python run_pipeline.py                                  # run default v2 flow
  python run_pipeline.py --steps 1 2 3                    # run specific steps
  python run_pipeline.py --steps 7 8 --model-path ...     # MLLM + package with FACS
  python run_pipeline.py --from-step 5                    # resume from step 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


STEPS = {
    1: "step1_generate_prompts.py",
    2: "step2_generate_images.py",
    3: "step3_crop_pairs.py",
    4: "step4_filter_pairs.py",
    5: "step5_build_reference_pool.py",
    6: "step6_construct_triplets.py",
    7: "step8_mllm_instructions.py",  # conceptual step7
    8: "step7_package_step1x.py",      # conceptual step8
}

STEP_DESCRIPTIONS = {
    1: "Generate prompts (template/GPT/manual)",
    2: "Generate side-by-side images (FLUX.2-klein)",
    3: "Crop pairs (1056x528 -> 512x512)",
    4: "Filter pairs (face + ArcFace + emotion + naturalness)",
    5: "Build I_r reference pool (real/flux)",
    6: "Construct triplets (I_e, I_r, I_e_edit)",
    7: "Generate FACS instructions with Qwen2.5-VL (optional)",
    8: "Package Step1X dual-image dataset",
}


def check_prerequisites() -> list[str]:
    issues = []
    if not Path("config.yaml").exists():
        issues.append("config.yaml not found")
    try:
        import yaml  # noqa: F401
    except ImportError:
        issues.append("pyyaml not installed")
    return issues


def validate_env_for_selected_steps(args, steps_to_run: list[int]) -> list[str]:
    warnings = []
    # Step1 GPT backend needs OPENAI key
    if 1 in steps_to_run and args.backend == "gpt" and not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY is required for step1 --backend gpt")

    # Step7 (MLLM) needs model path
    if 7 in steps_to_run and not args.model_path:
        warnings.append("--model-path is required when running step 7 (MLLM FACS instructions)")

    return warnings


def run_step(step_num: int, extra_args: list[str] | None = None, dry_run: bool = False) -> bool:
    script = STEPS[step_num]
    desc = STEP_DESCRIPTIONS[step_num]

    if not Path(script).exists():
        print(f"ERROR: {script} not found")
        return False

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("(dry-run: skipping execution)")
        return True

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\nStep {step_num} completed in {elapsed:.1f}s")
        return True

    print(f"\nStep {step_num} FAILED (exit code {result.returncode})")
    return False


def load_checkpoint(checkpoint_file: Path) -> dict:
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_steps": []}


def save_checkpoint(checkpoint_file: Path, state: dict):
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2)


def print_status_summary(config_path: str = "config.yaml"):
    del config_path  # kept for CLI compatibility

    print("\nPipeline Status (v2):")
    print("-" * 40)

    files_to_check = {
        "Prompts file": "data/prompts.jsonl",
        "Raw metadata": "data/raw/metadata.jsonl",
        "Crop metadata": "data/cropped/crop_metadata.jsonl",
        "Filter metadata": "data/filtered/filter_metadata.jsonl",
        "Filter stats": "data/filter_stats.json",
        "Reference pool": "data/reference_pool/pool_metadata.jsonl",
        "Triplets": "data/triplets/triplet_metadata.jsonl",
        "Step1X metadata": "data/dataset_v2/metadata.json",
    }

    for label, path in files_to_check.items():
        p = Path(path)
        if not p.exists():
            print(f"  {label}: NOT FOUND")
            continue

        if path.endswith(".jsonl"):
            try:
                with open(p) as f:
                    n_lines = sum(1 for _ in f)
                print(f"  {label}: {n_lines} records")
            except Exception:
                print(f"  {label}: exists ({p.stat().st_size} bytes)")
        elif path.endswith(".json"):
            print(f"  {label}: exists")
        else:
            print(f"  {label}: exists")


def main():
    parser = argparse.ArgumentParser(
        description="Run the face expression transfer dataset pipeline (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --steps 1 2 3 4
  python run_pipeline.py --steps 5 --pool-source both --pool-max-per-expression 100
  python run_pipeline.py --steps 6 --k 5
  python run_pipeline.py --steps 7 8 --model-path /path/to/Qwen2.5-VL-7B-Instruct --use-facs

Notes:
  - Default run excludes step 7 (MLLM) because it requires a local VL model.
  - If you provide --model-path and do not specify --steps, step 7 will be inserted automatically.
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Which steps to run (1-8). Default: 1 2 3 4 5 6 8")
    parser.add_argument("--from-step", type=int, default=None,
                        help="Start from this step number")
    parser.add_argument("--skip-steps", nargs="+", type=int, default=None,
                        help="Skip these step numbers")
    parser.add_argument("--status", action="store_true",
                        help="Show current pipeline status and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")

    # Step 1 args
    parser.add_argument("--num-persons", type=int, default=None,
                        help="(Step 1) Number of persons to generate")
    parser.add_argument("--pairs-per-person", type=int, default=None,
                        help="(Step 1) Emotion pairs per person")
    parser.add_argument("--backend", choices=["template", "gpt", "manual"], default="template",
                        help="(Step 1) Prompt backend")
    parser.add_argument("--manual-file", type=str, default=None,
                        help="(Step 1) Manual descriptions file path")

    # Step 2 args
    parser.add_argument("--seeds", type=int, default=None,
                        help="(Step 2) Seeds per pair")

    # Step 4 args
    parser.add_argument("--stats-only", action="store_true",
                        help="(Step 4) Only compute stats")
    parser.add_argument("--arcface-threshold", type=float, default=None,
                        help="(Step 4) ArcFace threshold override")

    # Step 5 args
    parser.add_argument("--pool-source", choices=["real", "flux", "both"], default=None,
                        help="(Step 5) Reference pool source")
    parser.add_argument("--pool-max-per-expression", type=int, default=None,
                        help="(Step 5) Max images per expression")
    parser.add_argument("--pool-seed", type=int, default=None,
                        help="(Step 5) Random seed")

    # Step 6 args
    parser.add_argument("--k", type=int, default=None,
                        help="(Step 6) I_r samples per pair")
    parser.add_argument("--triplet-seed", type=int, default=None,
                        help="(Step 6) Random seed")

    # Step 7 (MLLM) args
    parser.add_argument("--model-path", type=str, default=None,
                        help="(Step 7) Path to Qwen2.5-VL model")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="(Step 7) MLLM batch size")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="(Step 7) MLLM max_new_tokens")
    parser.add_argument("--limit", type=int, default=None,
                        help="(Step 7) Process at most N triplets")
    parser.add_argument("--overwrite", action="store_true",
                        help="(Step 7) Overwrite existing facs_instruction")

    # Step 8 args
    parser.add_argument("--symlink", action="store_true",
                        help="(Step 8) Use symlinks for target images")
    parser.add_argument("--use-facs", action="store_true",
                        help="(Step 8) Use FACS captions from step 7")
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="(Step 8) Resolution [H W]")

    args = parser.parse_args()

    if args.status:
        print_status_summary(args.config)
        return

    issues = check_prerequisites()
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        sys.exit(1)

    if args.steps:
        steps_to_run = sorted(args.steps)
    elif args.from_step:
        steps_to_run = list(range(args.from_step, 9))
    else:
        steps_to_run = [1, 2, 3, 4, 5, 6, 8]
        if args.model_path:
            steps_to_run.insert(6, 7)  # before packaging

    if args.skip_steps:
        skip = set(args.skip_steps)
        steps_to_run = [s for s in steps_to_run if s not in skip]

    invalid_steps = [s for s in steps_to_run if s not in STEPS]
    if invalid_steps:
        print(f"ERROR: invalid step numbers: {invalid_steps}. Valid range: 1-8")
        sys.exit(1)

    warnings = validate_env_for_selected_steps(args, steps_to_run)
    if warnings:
        for msg in warnings:
            print(f"ERROR: {msg}")
        sys.exit(1)

    print(f"Pipeline steps to run: {steps_to_run}")

    step_args: dict[int, list[str]] = {s: ["--config", args.config] for s in steps_to_run}

    if 1 in step_args:
        if args.num_persons is not None:
            step_args[1].extend(["--num-persons", str(args.num_persons)])
        if args.pairs_per_person is not None:
            step_args[1].extend(["--pairs-per-person", str(args.pairs_per_person)])
        step_args[1].extend(["--backend", args.backend])
        if args.manual_file:
            step_args[1].extend(["--manual-file", args.manual_file])

    if 2 in step_args and args.seeds is not None:
        step_args[2].extend(["--seeds", str(args.seeds)])

    if 4 in step_args:
        if args.stats_only:
            step_args[4].append("--stats-only")
        if args.arcface_threshold is not None:
            step_args[4].extend(["--arcface-threshold", str(args.arcface_threshold)])

    if 5 in step_args:
        if args.pool_source:
            step_args[5].extend(["--source", args.pool_source])
        if args.pool_max_per_expression is not None:
            step_args[5].extend(["--max-per-expression", str(args.pool_max_per_expression)])
        if args.pool_seed is not None:
            step_args[5].extend(["--seed", str(args.pool_seed)])

    if 6 in step_args:
        if args.k is not None:
            step_args[6].extend(["--k", str(args.k)])
        if args.triplet_seed is not None:
            step_args[6].extend(["--seed", str(args.triplet_seed)])

    if 7 in step_args:
        step_args[7].extend(["--model-path", args.model_path])
        if args.batch_size is not None:
            step_args[7].extend(["--batch-size", str(args.batch_size)])
        if args.max_new_tokens is not None:
            step_args[7].extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.limit is not None:
            step_args[7].extend(["--limit", str(args.limit)])
        if args.overwrite:
            step_args[7].append("--overwrite")

    if 8 in step_args:
        if args.symlink:
            step_args[8].append("--symlink")
        if args.use_facs:
            step_args[8].append("--use-facs")
        if args.resolution is not None:
            step_args[8].extend(["--resolution", str(args.resolution[0]), str(args.resolution[1])])

    checkpoint_file = Path("data/.pipeline_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_file)

    failed_steps: list[int] = []
    for step_num in steps_to_run:
        success = run_step(step_num, step_args.get(step_num, []), args.dry_run)

        if success:
            if step_num not in checkpoint["completed_steps"]:
                checkpoint["completed_steps"].append(step_num)
            save_checkpoint(checkpoint_file, checkpoint)
        else:
            failed_steps.append(step_num)
            print(f"\nPipeline stopped at step {step_num}. Fix errors and re-run.")
            print(f"Resume with: python run_pipeline.py --from-step {step_num}")
            break

    print(f"\n{'='*60}")
    if not failed_steps:
        print("Pipeline completed successfully!")
        print_status_summary(args.config)
    else:
        print(f"Pipeline failed at steps: {failed_steps}")
        print("Check logs above for details.")

    return 0 if not failed_steps else 1


if __name__ == "__main__":
    sys.exit(main())

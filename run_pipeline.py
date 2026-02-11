#!/usr/bin/env python3
"""
End-to-end pipeline runner with checkpoint resumption.

Usage:
  python run_pipeline.py                          # run all steps
  python run_pipeline.py --steps 1 2 3            # run specific steps
  python run_pipeline.py --steps 4 --stats-only   # step4 stats only
  python run_pipeline.py --from-step 2            # resume from step 2
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
    5: "step5_package_dataset.py",
}

STEP_DESCRIPTIONS = {
    1: "Generate prompts (GPT-4o)",
    2: "Generate images (FLUX.1-dev)",
    3: "Crop pairs (1056×528 → 512×512)",
    4: "Filter pairs (face detect + ArcFace + emotion)",
    5: "Package dataset",
}


def check_env():
    """Check required environment variables."""
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("HF_TOKEN"):
        missing.append("HF_TOKEN")
    return missing


def check_prerequisites():
    """Check that required files/packages exist."""
    issues = []

    # Check config
    if not Path("config.yaml").exists():
        issues.append("config.yaml not found")

    # Check Python packages
    try:
        import yaml
    except ImportError:
        issues.append("pyyaml not installed")

    return issues


def run_step(step_num: int, extra_args: list = None, dry_run: bool = False) -> bool:
    """Run a single pipeline step. Returns True if successful."""
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
    else:
        print(f"\nStep {step_num} FAILED (exit code {result.returncode})")
        return False


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load pipeline checkpoint state."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_steps": []}


def save_checkpoint(checkpoint_file: Path, state: dict):
    """Save pipeline checkpoint state."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2)


def print_status_summary(config_path: str = "config.yaml"):
    """Print current pipeline status."""
    print("\nPipeline Status:")
    print("-" * 40)

    files_to_check = {
        "Prompts file": "data/prompts.jsonl",
        "Raw metadata": "data/raw/metadata.jsonl",
        "Crop metadata": "data/cropped/crop_metadata.jsonl",
        "Filter metadata": "data/filtered/filter_metadata.jsonl",
        "Filter stats": "data/filter_stats.json",
        "Dataset metadata": "data/dataset/metadata.jsonl",
    }

    for label, path in files_to_check.items():
        p = Path(path)
        if p.exists():
            size = p.stat().st_size
            # Count lines for JSONL files
            if path.endswith(".jsonl"):
                try:
                    with open(p) as f:
                        n_lines = sum(1 for _ in f)
                    print(f"  {label}: {n_lines} records")
                except Exception:
                    print(f"  {label}: exists ({size} bytes)")
            elif path.endswith(".json"):
                print(f"  {label}: exists")
            else:
                print(f"  {label}: exists")
        else:
            print(f"  {label}: NOT FOUND")

    # Show filter stats if available
    stats_file = Path("data/filter_stats.json")
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                stats = json.load(f)
            print(f"\nFilter Stats:")
            print(f"  Total processed: {stats.get('total', 0)}")
            print(f"  Pass layer 1 (face): {stats.get('pass_rate_layer1', 0):.1%}")
            print(f"  Pass layer 2 (arcface): {stats.get('pass_rate_layer2', 0):.1%}")
            print(f"  Pass layer 3 (emotion): {stats.get('pass_rate_layer3', 0):.1%}")
            print(f"  Overall pass rate: {stats.get('overall_pass_rate', 0):.1%}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run the face emotion paired image dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                         # Run all steps
  python run_pipeline.py --steps 1 2             # Run only steps 1 and 2
  python run_pipeline.py --from-step 3           # Resume from step 3
  python run_pipeline.py --status                # Show current status
  python run_pipeline.py --steps 4 --stats-only  # Step 4 stats only mode

Step 4 workflow:
  1. First run with --stats-only to check thresholds
  2. Then run without --stats-only to do actual filtering
        """
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Which steps to run (1-5). Default: all")
    parser.add_argument("--from-step", type=int, default=None,
                        help="Start from this step number")
    parser.add_argument("--skip-steps", nargs="+", type=int, default=None,
                        help="Skip these step numbers")
    parser.add_argument("--status", action="store_true",
                        help="Show current pipeline status and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")

    # Step-specific arguments
    parser.add_argument("--num-persons", type=int, default=None,
                        help="(Step 1) Number of persons to generate")
    parser.add_argument("--pairs-per-person", type=int, default=None,
                        help="(Step 1) Emotion pairs per person")
    parser.add_argument("--backend", choices=["template", "gpt", "manual"], default="template",
                        help="(Step 1) Prompt generation backend: template (default), gpt, manual")
    parser.add_argument("--manual-file", type=str, default=None,
                        help="(Step 1) Path to manual descriptions file (for --backend manual)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="(Step 2) Seeds per pair")
    parser.add_argument("--stats-only", action="store_true",
                        help="(Step 4) Only compute stats, don't copy files")
    parser.add_argument("--arcface-threshold", type=float, default=None,
                        help="(Step 4) ArcFace similarity threshold")

    args = parser.parse_args()

    if args.status:
        print_status_summary(args.config)
        return

    # Check environment
    missing_env = check_env()
    if missing_env:
        print(f"WARNING: Missing environment variables: {', '.join(missing_env)}")
        print("Set them with: export VAR=value")

    issues = check_prerequisites()
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        sys.exit(1)

    # Determine which steps to run
    if args.steps:
        steps_to_run = sorted(args.steps)
    elif args.from_step:
        steps_to_run = list(range(args.from_step, 6))
    else:
        steps_to_run = list(range(1, 6))

    if args.skip_steps:
        steps_to_run = [s for s in steps_to_run if s not in args.skip_steps]

    print(f"Pipeline steps to run: {steps_to_run}")

    # Build per-step extra arguments
    step_args = {s: [] for s in steps_to_run}

    if 1 in step_args:
        if args.num_persons:
            step_args[1].extend(["--num-persons", str(args.num_persons)])
        if args.pairs_per_person:
            step_args[1].extend(["--pairs-per-person", str(args.pairs_per_person)])
        step_args[1].extend(["--backend", args.backend])
        if args.manual_file:
            step_args[1].extend(["--manual-file", args.manual_file])

    if 2 in step_args:
        if args.seeds:
            step_args[2].extend(["--seeds", str(args.seeds)])

    if 4 in step_args:
        if args.stats_only:
            step_args[4].append("--stats-only")
        if args.arcface_threshold:
            step_args[4].extend(["--arcface-threshold", str(args.arcface_threshold)])

    # Run steps
    checkpoint_file = Path("data/.pipeline_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_file)

    failed_steps = []
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

    # Final status
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

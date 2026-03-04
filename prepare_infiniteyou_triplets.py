#!/usr/bin/env python3
"""
Prepare training triplets using InfiniteYou model.

For each sample {id} in cropped/, generates:
  - source.png : InfiniteYou(right.png, seed=S)
  - target.png : InfiniteYou(left.png,  seed=S)   # same seed S
  - reference.png : direct copy of left.png

Usage:
  CUDA_VISIBLE_DEVICES=6 python prepare_infiniteyou_triplets.py
"""

import random
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
BASE_PATH = Path("/scratch3/f007yzf/flux_face_emotion")
dataset_name = "data_emotion_test"

INPUT_DIR = BASE_PATH / dataset_name / "cropped"
OUTPUT_DIR = BASE_PATH / dataset_name / "training_data"

REQUIRED_OUTPUTS = {"source.png", "reference.png", "target.png"}

# InfiniteYou settings
INFINITEYOU_REPO = Path("/scratch3/f007yzf/repos/InfiniteYou")
MODEL_VERSION = "aes_stage2"
CUDA_DEVICE = 0  # logical device (use CUDA_VISIBLE_DEVICES to pick physical GPU)

PROMPT = (
    "Same person as reference, keep exactly the same facial expression "
    "(slightly furrowed brows, neutral closed lips, direct eye contact), "
    "no makeup, soft daylight, green park background, creamy bokeh, "
    "photorealistic portrait, hyper-detailed background, "
    "shot on DSLR, 35mm lens."
)

# Pipeline parameters
GUIDANCE_SCALE = 3.5
NUM_STEPS = 30
INFUSENET_CONDITIONING_SCALE = 1.0
INFUSENET_GUIDANCE_START = 0.0
INFUSENET_GUIDANCE_END = 1.0


# ============================================================
# Load InfiniteYou pipeline (once)
# ============================================================
def load_pipeline():
    """Load the InfiniteYou-FLUX pipeline onto the GPU."""
    # Add InfiniteYou repo to path so its modules can be imported
    sys.path.insert(0, str(INFINITEYOU_REPO))

    from pipelines.pipeline_infu_flux import InfUFluxPipeline

    model_dir = "ByteDance/InfiniteYou"
    infu_model_path = f"{model_dir}/infu_flux_v1.0/{MODEL_VERSION}"
    insightface_root_path = f"{model_dir}/supports/insightface"

    torch.cuda.set_device(CUDA_DEVICE)
    pipe = InfUFluxPipeline(
        base_model_path="black-forest-labs/FLUX.1-dev",
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version="v1.0",
        model_version=MODEL_VERSION,
    )
    pipe.load_loras([])
    print("InfiniteYou pipeline loaded.")
    return pipe


# ============================================================
# InfiniteYou inference
# ============================================================
def run_infinite_you(
    pipe, input_path: Path, output_path: Path, seed: int
) -> None:
    """Call InfiniteYou to generate a face-edited image.

    Args:
        pipe:        The loaded InfUFluxPipeline instance.
        input_path:  Path to the input (id) image.
        output_path: Path where the output image should be saved.
        seed:        Random seed for reproducible generation.
    """
    tqdm.write(f"  [InfiniteYou] {input_path.name} -> {output_path.name}  (seed={seed})")

    id_image = Image.open(input_path).convert("RGB")
    result = pipe(
        id_image=id_image,
        prompt=PROMPT,
        control_image=None,
        seed=seed,
        guidance_scale=GUIDANCE_SCALE,
        num_steps=NUM_STEPS,
        infusenet_conditioning_scale=INFUSENET_CONDITIONING_SCALE,
        infusenet_guidance_start=INFUSENET_GUIDANCE_START,
        infusenet_guidance_end=INFUSENET_GUIDANCE_END,
    )
    result.save(str(output_path))


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    # Collect sample IDs (subdirectories only, skip files like metadata)
    sample_ids = sorted(d.name for d in INPUT_DIR.iterdir() if d.is_dir())

    if not sample_ids:
        print(f"No sample directories found in {INPUT_DIR}")
        return

    print(f"Found {len(sample_ids)} samples in {INPUT_DIR}")

    # Load model once
    pipe = load_pipeline()

    skipped = 0
    processed = 0
    warned = 0

    for sample_id in tqdm(sample_ids, desc="Processing triplets"):
        in_dir = INPUT_DIR / sample_id
        out_dir = OUTPUT_DIR / sample_id

        # --- Skip if already complete ---
        if out_dir.exists():
            existing = {f.name for f in out_dir.iterdir() if f.is_file()}
            if REQUIRED_OUTPUTS <= existing:
                skipped += 1
                tqdm.write(f"[SKIP] {sample_id} — already complete")
                continue

        # --- Validate inputs ---
        right_img = in_dir / "right.png"
        left_img = in_dir / "left.png"

        missing = []
        if not right_img.exists():
            missing.append("right.png")
        if not left_img.exists():
            missing.append("left.png")

        if missing:
            warned += 1
            tqdm.write(
                f"[WARN] {sample_id} — missing {', '.join(missing)}, skipping"
            )
            continue

        # --- Process ---
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = random.randint(0, 2**32 - 1)

        try:
            # 1) source.png  <-  right.png  (InfiniteYou)
            run_infinite_you(pipe, right_img, out_dir / "source.png", seed=seed)

            # 2) target.png  <-  left.png   (InfiniteYou, same seed)
            run_infinite_you(pipe, left_img, out_dir / "target.png", seed=seed)

            # 3) reference.png  <-  left.png  (direct copy)
            shutil.copy2(left_img, out_dir / "reference.png")

            processed += 1
        except Exception as e:
            tqdm.write(f"[ERROR] {sample_id} — {e}")
            warned += 1

    print(
        f"\nDone — processed: {processed}, skipped: {skipped}, warned: {warned}"
    )


if __name__ == "__main__":
    main()

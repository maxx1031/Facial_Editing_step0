#!/usr/bin/env python3
"""
Step 8: Generate FACS-based editing instructions using Qwen2.5-VL.

For each (I_e, I_r) pair in the triplet metadata, call Qwen2.5-VL to:
  1. Analyze source FACS AUs (from I_e)
  2. Analyze target FACS AUs (from I_r)
  3. Compute the AU delta
  4. Produce a structured diffusion-model editing instruction

The generated instruction is written back to
data/triplets/triplet_metadata.jsonl (adds field "facs_instruction").

Re-run step7 with --use-facs after this step to apply the new captions
into data/dataset_v2/metadata.json.

Input:
  data/triplets/triplet_metadata.jsonl

Output:
  data/triplets/triplet_metadata.jsonl  (updated in-place, adds facs_instruction)

Usage:
  python step8_mllm_instructions.py --model-path /path/to/Qwen2.5-VL-7B-Instruct
  python step8_mllm_instructions.py --model-path ... --batch-size 4 --max-new-tokens 512
  python step8_mllm_instructions.py --model-path ... --limit 50   # quick test
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import jsonlines
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FACS prompt template
# ---------------------------------------------------------------------------

FACS_ANALYSIS_PROMPT = """You are analyzing facial expressions for a controlled editing task.

Given two images:
- Image 1: source face to be edited
- Image 2: target expression reference

Please output:
1. Source expression in FACS AUs
2. Target expression in FACS AUs
3. Delta (which AUs to activate/deactivate)
4. A structured editing instruction for a diffusion model that:
   - Only describes expression change
   - Uses muscle-level language
   - Explicitly says "keep identity, age, gender, skin tone, lighting unchanged"
   - Avoids style/aesthetic language

Format your response as:
SOURCE_AUs: <list>
TARGET_AUs: <list>
DELTA_ACTIVATE: <list>
DELTA_DEACTIVATE: <list>
INSTRUCTION: <single paragraph editing instruction>"""


def extract_instruction(response_text: str) -> str:
    """Extract just the INSTRUCTION line from the MLLM response."""
    lines = response_text.strip().splitlines()
    for line in lines:
        if line.startswith("INSTRUCTION:"):
            return line[len("INSTRUCTION:"):].strip()
    # Fallback: return the full response if the expected format isn't found
    return response_text.strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen_model(model_path: str):
    """Load Qwen2.5-VL-7B-Instruct for generative inference."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2.5-VL from {model_path} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    print("Qwen2.5-VL loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def analyze_pair(
    model,
    processor,
    I_e_path: str,
    I_r_path: str,
    max_new_tokens: int = 512,
) -> str:
    """Run FACS analysis on (I_e, I_r) pair and return full response text."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    I_e = Image.open(I_e_path).convert("RGB")
    I_r = Image.open(I_r_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FACS_ANALYSIS_PROMPT},
                {"type": "image", "image": I_e},
                {"type": "image", "image": I_r},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated tokens (not the prompt)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True)


def analyze_batch(
    model,
    processor,
    batch: list[dict],
    max_new_tokens: int,
) -> list[str]:
    """Process a batch of triplets. Returns list of response strings."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    all_texts = []
    all_image_inputs = []

    messages_batch = []
    for triplet in batch:
        I_e = Image.open(triplet["I_e_path"]).convert("RGB")
        I_r = Image.open(triplet["I_r_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FACS_ANALYSIS_PROMPT},
                    {"type": "image", "image": I_e},
                    {"type": "image", "image": I_r},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, _ = process_vision_info(messages)
        all_texts.append(text)
        all_image_inputs.extend(image_inputs)

    inputs = processor(
        text=all_texts,
        images=all_image_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    results = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(batch)):
        generated = output_ids[i][input_len:]
        results.append(processor.decode(generated, skip_special_tokens=True))
    return results


# ---------------------------------------------------------------------------
# Triplet metadata I/O
# ---------------------------------------------------------------------------

def load_triplets(triplets_path: Path) -> list[dict]:
    with jsonlines.open(triplets_path) as reader:
        return list(reader)


def save_triplets(triplets: list[dict], triplets_path: Path) -> None:
    """Write triplets to a temp file then rename (atomic update)."""
    tmp = triplets_path.with_suffix(".jsonl.tmp")
    with jsonlines.open(tmp, mode="w") as writer:
        for rec in triplets:
            writer.write(rec)
    tmp.replace(triplets_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate FACS-based editing instructions with Qwen2.5-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After this step, re-run step7 with --use-facs to apply the FACS captions:
  python step7_package_step1x.py --use-facs

Examples:
  python step8_mllm_instructions.py --model-path /path/to/Qwen2.5-VL-7B-Instruct
  python step8_mllm_instructions.py --model-path ... --batch-size 1  # memory-safe
  python step8_mllm_instructions.py --model-path ... --limit 20      # quick test
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model-path", required=True,
        help="Path to Qwen2.5-VL-7B-Instruct model directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (default 1; increase if VRAM allows)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens to generate per pair",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N triplets (for quick testing)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate instructions even for triplets that already have facs_instruction",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    triplets_path = Path(cfg["paths"]["triplets_metadata"])

    if not triplets_path.exists():
        print(f"ERROR: {triplets_path} not found. Run step6 first.")
        sys.exit(1)

    print(f"Loading triplets from {triplets_path} ...")
    triplets = load_triplets(triplets_path)
    print(f"  Total: {len(triplets)}")

    # Filter to triplets that need processing
    to_process_indices = []
    for i, t in enumerate(triplets):
        if args.overwrite or not t.get("facs_instruction"):
            to_process_indices.append(i)

    if args.limit is not None:
        to_process_indices = to_process_indices[: args.limit]

    print(f"  Need processing: {len(to_process_indices)}")
    if not to_process_indices:
        print("Nothing to do (all triplets already have facs_instruction).")
        return

    model, processor = load_qwen_model(args.model_path)

    # Process in batches
    batch_size = args.batch_size
    processed = 0
    errors = 0

    for batch_start in tqdm(
        range(0, len(to_process_indices), batch_size),
        desc="FACS analysis",
        unit="batch",
    ):
        batch_indices = to_process_indices[batch_start : batch_start + batch_size]
        batch_triplets = [triplets[i] for i in batch_indices]

        # Validate image paths
        valid_batch = []
        valid_indices = []
        for i, t in zip(batch_indices, batch_triplets):
            if Path(t["I_e_path"]).exists() and Path(t["I_r_path"]).exists():
                valid_batch.append(t)
                valid_indices.append(i)
            else:
                errors += 1

        if not valid_batch:
            continue

        try:
            if batch_size == 1:
                responses = [
                    analyze_pair(
                        model, processor,
                        valid_batch[0]["I_e_path"],
                        valid_batch[0]["I_r_path"],
                        max_new_tokens=args.max_new_tokens,
                    )
                ]
            else:
                responses = analyze_batch(
                    model, processor, valid_batch,
                    max_new_tokens=args.max_new_tokens,
                )
        except Exception as e:
            print(f"\nBatch inference failed: {e}")
            errors += len(valid_batch)
            continue

        for idx, response in zip(valid_indices, responses):
            instruction = extract_instruction(response)
            triplets[idx]["facs_instruction"] = instruction
            triplets[idx]["facs_response_full"] = response
            processed += 1

        # Save incrementally every 50 processed items
        if processed % 50 == 0:
            save_triplets(triplets, triplets_path)

    # Final save
    save_triplets(triplets, triplets_path)

    already_have = sum(1 for t in triplets if t.get("facs_instruction"))
    print(f"\nDone.")
    print(f"  Newly processed: {processed}")
    print(f"  Errors/skipped:  {errors}")
    print(f"  Total with FACS instruction: {already_have} / {len(triplets)}")
    print()
    print("Next: re-run step7 with --use-facs to apply FACS captions to metadata.json")
    print("  python step7_package_step1x.py --use-facs")


if __name__ == "__main__":
    main()

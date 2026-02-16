import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil

PROJECT_ROOT = Path("/scratch3/f007yzf/flux_face_emotion")
CROPPED_DIR = PROJECT_ROOT / "data/cropped"
META_PATH = CROPPED_DIR / "crop_metadata.json"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "mvp_step1x_case_runs"
DEFAULT_IMAGE_ID = "p0000_pair00_s2"
MODEL_PATH = '/scratch3/f007yzf/models/step1x_v11'

def build_instruction(target_emotion: str) -> str:
    return f"Keep the same identity. Change only the facial expression to {target_emotion}."


def load_case_metadata(image_id: str) -> dict:
    if not META_PATH.exists():
        return {}
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("image_id") == image_id:
                return obj
    return {}


def make_run_name(image_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{image_id}"


def main():
    parser = argparse.ArgumentParser(description="Prepare one Step1X case experiment without overwriting old runs.")
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID, help="Image ID in data/cropped, e.g. p0000_pair00_s2")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Experiment root directory")
    parser.add_argument("--run-name", default=None, help="Optional run folder name; default is timestamp_imageid")
    parser.add_argument("--instruction", default=None, help="Full custom instruction string")
    parser.add_argument("--target-emotion", default=None, help="Override target emotion")
    parser.add_argument("--source-emotion", default=None, help="Override source emotion")
    parser.add_argument("--copy-target", action="store_true", help="Also copy right.png into refs/ for quick comparison")
    parser.add_argument("--note", default="", help="Optional note stored in run_config.json")
    args = parser.parse_args()

    image_id = args.image_id
    case_dir = CROPPED_DIR / image_id
    left_path = case_dir / "left.png"
    right_path = case_dir / "right.png"
    if not left_path.exists():
        raise FileNotFoundError(f"Case source image not found: {left_path}")

    meta = load_case_metadata(image_id)
    person_id = meta.get("person_id")
    person_description = meta.get("person_description")
    source_emotion = args.source_emotion or meta.get("emotion_left")
    target_emotion = args.target_emotion or meta.get("emotion_right") or "happy"
    instruction = args.instruction or build_instruction(target_emotion)

    out_root = Path(args.out_root)
    run_name = args.run_name or make_run_name(image_id)
    run_dir = out_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    input_dir = run_dir / "inputs"
    refs_dir = run_dir / "refs"
    input_dir.mkdir(parents=True, exist_ok=False)

    in_name = f"{image_id}.png"
    input_path = input_dir / in_name
    # Single-case workflow uses file copy so each run remains self-contained.
    shutil.copy2(left_path, input_path)

    if args.copy_target and right_path.exists():
        refs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(right_path, refs_dir / f"{image_id}_target.png")

    prompts_for_inference = {
        in_name: instruction,
    }

    prompt_detail = {
        in_name: {
            "instruction": instruction,
            "person_description": person_description,
        }
    }

    manifest = [{
        "image_id": image_id,
        "person_id": person_id,
        "input_name": in_name,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "source_emotion": source_emotion,
        "target_emotion": target_emotion,
        "instruction": instruction,
        "person_description": person_description,
        "run_name": run_name,
    }]

    run_config = {
        "image_id": image_id,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "note": args.note,
        "source_emotion": source_emotion,
        "target_emotion": target_emotion,
        "instruction": instruction,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    prompts_path = run_dir / "prompts.json"
    prompt_detail_path = run_dir / "prompt_detail.json"
    manifest_path = run_dir / "manifest.json"
    run_config_path = run_dir / "run_config.json"
    prompts_path.write_text(json.dumps(prompts_for_inference, indent=2, ensure_ascii=False))
    prompt_detail_path.write_text(json.dumps(prompt_detail, indent=2, ensure_ascii=False))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False))

    print("Wrote:")
    print(" -", input_dir)
    print(" -", prompts_path, "(for Step1X inference.py)")
    print(" -", prompt_detail_path)
    print(" -", manifest_path)
    print(" -", run_config_path)
    print("")
    print("Run inference with:")
    print(
        "python /scratch3/f007yzf/repos/Step1X-Edit/inference.py "
        f"--model_path {MODEL_PATH} --input_dir {input_dir} "
        f"--output_dir {run_dir / 'outputs'} --json_path {prompts_path} --task_type edit"
        " --version v1.1"
        " --offload"
    )


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import random

CROPPED_DIR = Path("/scratch3/f007yzf/flux_face_emotion/data/cropped")
META_PATH = CROPPED_DIR / "crop_metadata.json"

OUT_ROOT = Path("/scratch3/f007yzf/flux_face_emotion/mvp_step1x_v11_3")
INPUT_DIR = OUT_ROOT / "inputs"
PROMPTS_JSON = OUT_ROOT / "prompts.json"

N = 200
SEED = 0

def build_instruction(target_emotion: str) -> str:
    return f"Keep the same identity. Change only the facial expression to {target_emotion}."

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in INPUT_DIR.glob("*.png"):
        if p.exists() or p.is_symlink():
            p.unlink()

    rows = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 只取 seed_idx=0，避免同一 pair 多张近似图；你也可以删掉这行
            if obj.get("seed_idx", None) != 0:
                continue
            rows.append(obj)

    per_person = {}
    for obj in rows:
        pid = obj.get("person_id")
        if not pid:
            continue
        if pid not in per_person:
            per_person[pid] = obj

    persons = list(per_person.values())
    random.Random(SEED).shuffle(persons)
    chosen = persons[:N]

    prompts = {}
    manifest = []

    for obj in chosen:
        image_id = obj["image_id"]                       # e.g. p0000_pair00_s0
        person_id = obj.get("person_id")
        left_path = Path(obj["left_path"])              # .../left.png
        right_path = Path(obj["right_path"])            # .../right.png
        source_emotion = obj.get("emotion_left")
        target_emotion = obj["emotion_right"]           # 用 right 的标签做目标
        person_description = obj.get("person_description")

        # Step1X 需要 input_dir 下是一张图文件；我们用 image_id.png 作为文件名
        in_name = f"{image_id}.png"
        link_path = INPUT_DIR / in_name

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        # 软链到 left.png
        link_path.symlink_to(left_path)

        prompts[in_name] = {
            "instruction": build_instruction(target_emotion),
            "person_description": person_description,
        }

        manifest.append({
            "image_id": image_id,
            "person_id": person_id,
            "input_name": in_name,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "source_emotion": source_emotion,
            "target_emotion": target_emotion,
            "instruction": prompts[in_name]["instruction"],
            "person_description": person_description,
        })

    PROMPTS_JSON.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("Wrote:")
    print(" -", INPUT_DIR)
    print(" -", PROMPTS_JSON)
    print(" -", OUT_ROOT / "manifest.json")

if __name__ == "__main__":
    main()
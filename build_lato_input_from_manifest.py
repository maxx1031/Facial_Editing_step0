import json
from pathlib import Path

ROOT = Path("/scratch3/f007yzf/flux_face_emotion/mvp_step1x_v11_3")
MANIFEST = ROOT / "manifest.json"
OUT = ROOT / "lato" / "input.json"

def main():
    items = json.loads(MANIFEST.read_text())
    out = {}
    for it in items:
        image_id = it["image_id"]
        out[image_id] = {
            "caption": it["instruction"],
            "source_image_path": it["left_path"],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
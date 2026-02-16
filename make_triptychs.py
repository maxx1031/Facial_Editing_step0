import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/scratch3/f007yzf/flux_face_emotion/mvp_step1x_v11_3")
MANIFEST = ROOT / "manifest.json"
PROMPTS_JSON = ROOT / "prompts.json"

# 这里按实际 inference 产出的目录名改一下
STEP1X_OUT_DIR = ROOT / "step1x_outputs-offload-512"

VIZ_DIR = ROOT / "viz_triptych"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_centered_header_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    center_x: int,
    header_h: int,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        (center_x - tw // 2, (header_h - th) // 2),
        text,
        fill=color,
        font=font,
    )


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = (text or "").replace("\n", " ").split()
    if not words:
        return [""]

    lines: list[str] = []
    cur = words[0]

    for w in words[1:]:
        candidate = f"{cur} {w}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            cur = candidate
            continue
        lines.append(cur)
        cur = w
    lines.append(cur)
    return lines


def _draw_centered_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        (x0 + (w - tw) // 2, y0 + (h - th) // 2),
        text,
        fill=color,
        font=font,
    )


def _draw_centered_multiline_text_in_box(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int] = (0, 0, 0),
    line_gap: int = 6,
) -> None:
    cleaned = [ln for ln in (lines or []) if ln is not None]
    if not cleaned:
        cleaned = [""]

    bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in cleaned]
    heights = [(bb[3] - bb[1]) for bb in bboxes]
    widths = [(bb[2] - bb[0]) for bb in bboxes]
    total_h = sum(heights) + line_gap * (len(cleaned) - 1)
    cur_y = y0 + (h - total_h) // 2

    for ln, tw, th in zip(cleaned, widths, heights):
        draw.text(
            (x0 + (w - tw) // 2, cur_y),
            ln,
            fill=color,
            font=font,
        )
        cur_y += th + line_gap

def main():
    items = json.loads(MANIFEST.read_text())
    prompts_map = {}
    if PROMPTS_JSON.exists():
        prompts_map = json.loads(PROMPTS_JSON.read_text())

    for it in items:
        in_name = it["input_name"]
        left = load_rgb(Path(it["left_path"]))
        right = load_rgb(Path(it["right_path"]))
        edited_path = STEP1X_OUT_DIR / in_name
        edited = load_rgb(edited_path)

        w, h = left.size
        header_h = 120
        canvas_w = w * 3

        prompt_font = _get_font(34)
        desc_font = _get_font(26)

        person_description = it.get("person_description")
        prompt_obj = prompts_map.get(in_name)
        if isinstance(prompt_obj, dict):
            person_description = person_description or prompt_obj.get("person_description")

        desc_lines: list[str] = []
        if person_description:
            desc_draw = ImageDraw.Draw(Image.new("RGB", (10, 10), (255, 255, 255)))
            desc_lines = _wrap_text_to_width(
                desc_draw,
                str(person_description),
                font=desc_font,
                max_width=canvas_w - 80,
            )

        prompt_h = 150 + (len(desc_lines) * 34 if desc_lines else 0)

        canvas = Image.new("RGB", (canvas_w, h + header_h + prompt_h), (255, 255, 255))

        canvas.paste(left, (0, header_h))
        canvas.paste(edited, (w, header_h))
        canvas.paste(right, (w * 2, header_h))

        draw = ImageDraw.Draw(canvas)
        font = _get_font(44)
        titles = ["Input image", "Step1X", "Target"]
        for i, t in enumerate(titles):
            cx = i * w + w // 2
            _draw_centered_header_text(draw, t, center_x=cx, header_h=header_h, font=font)

        # Prompt area (fixed height): show emotion transition only
        src_emotion = it.get("source_emotion") or it.get("emotion_left") or it.get("original_emotion") or "unknown"

        tgt_emotion = it.get("target_emotion") or it.get("emotion_right")
        if not tgt_emotion:
            instr_obj = prompts_map.get(in_name) or it.get("instruction") or ""
            instr = instr_obj.get("instruction") if isinstance(instr_obj, dict) else instr_obj
            m = re.search(r"\bto\s+([a-zA-Z_-]+)\b", instr)
            tgt_emotion = m.group(1) if m else "unknown"

        prompt = f"{src_emotion} → {tgt_emotion}"
        prompt_top = header_h + h
        draw.line([(0, prompt_top), (canvas_w, prompt_top)], fill=(220, 220, 220), width=2)

        prompt_bbox = draw.textbbox((0, 0), prompt, font=prompt_font)
        prompt_th = prompt_bbox[3] - prompt_bbox[1]
        prompt_y = prompt_top + 18
        _draw_centered_text_in_box(
            draw,
            prompt,
            x0=0,
            y0=prompt_y,
            w=canvas_w,
            h=prompt_th,
            font=prompt_font,
        )

        if desc_lines:
            desc_top = prompt_y + prompt_th + 18
            _draw_centered_multiline_text_in_box(
                draw,
                desc_lines,
                x0=0,
                y0=desc_top,
                w=canvas_w,
                h=(header_h + h + prompt_h) - desc_top,
                font=desc_font,
            )

        out_path = VIZ_DIR / f"{it['image_id']}_triptych.png"
        canvas.save(out_path)
        print("wrote", out_path)

if __name__ == "__main__":
    main()
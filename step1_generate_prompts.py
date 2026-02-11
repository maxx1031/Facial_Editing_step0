#!/usr/bin/env python3
"""
Step 1: Generate person descriptions and emotion pair prompts.

Backends:
  --backend template  (default) — fast, no API, uses built-in diverse template pool
  --backend gpt       — calls GPT-4o for richer, more varied descriptions
  --backend manual    — reads descriptions from a user-provided text file (one per line)

Output: data/prompts.jsonl
Each line: {
  "person_id": "p0001",
  "person_description": "...",
  "pairs": [
    {
      "pair_id": "p0001_pair00",
      "emotion_left": "neutral",
      "emotion_right": "happy",
      "prompt_left": "...",
      "prompt_right": "...",
      "combined_prompt": "..."
    },
    ...
  ]
}
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import jsonlines
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    def expand(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            return os.environ.get(var, "")  # return empty if not set
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        return obj
    return expand(raw)


# ---------------------------------------------------------------------------
# Built-in template pool (used by --backend template)
# Covers diverse age, gender, ethnicity, facial features
# ---------------------------------------------------------------------------

_AGES = [
    "18-year-old", "22-year-old", "28-year-old", "35-year-old",
    "42-year-old", "50-year-old", "58-year-old", "65-year-old",
]

_GENDERS = ["man", "woman", "non-binary person"]

_ETHNICITIES = [
    "East Asian", "South Asian", "Black African", "Nigerian",
    "Afro-Caribbean", "Hispanic Latino", "Mexican", "Colombian",
    "White European", "Irish", "Italian", "German",
    "Middle Eastern", "Iranian", "Turkish", "Arab",
    "Southeast Asian", "Filipino", "Vietnamese", "Thai",
    "Mixed heritage, half Black half White",
    "Mixed heritage, half Asian half Hispanic",
    "Indigenous American", "Scandinavian", "Greek",
]

_HAIR = [
    "short black hair", "long wavy brown hair", "curly red hair",
    "straight blonde hair", "dreadlocks", "tight coils, natural hair",
    "shaved head", "medium-length silver hair", "braided dark hair",
    "spiky platinum hair", "long straight black hair", "short grey hair",
    "afro natural hair", "loose curls chestnut hair",
]

_FEATURES = [
    "sharp jawline and high cheekbones", "soft round face with dimples",
    "prominent Roman nose and strong brow", "almond-shaped eyes and full lips",
    "freckles across the nose and cheeks", "deep-set eyes and angular features",
    "wide-set eyes and button nose", "defined cheekbones and hooded eyes",
    "square jaw and thick eyebrows", "soft oval face with a gentle smile line",
    "striking amber eyes and arched brows", "deep brown eyes and broad nose",
    "light green eyes and a cleft chin", "dark eyes and prominent ears",
    "large expressive eyes and a pointed chin",
]

_SKIN_TONES = [
    "fair skin", "light skin with pink undertones", "medium olive skin",
    "warm tan skin", "medium brown skin", "rich dark brown skin",
    "deep ebony skin", "golden-toned skin", "cool beige skin",
    "warm copper skin",
]


def generate_template_descriptions(n: int, seed: int = 42) -> list[str]:
    """Generate n diverse person descriptions from built-in template pool."""
    rng = random.Random(seed)
    descriptions = []
    seen = set()
    attempts = 0

    while len(descriptions) < n and attempts < n * 20:
        attempts += 1
        age = rng.choice(_AGES)
        gender = rng.choice(_GENDERS)
        ethnicity = rng.choice(_ETHNICITIES)
        hair = rng.choice(_HAIR)
        features = rng.choice(_FEATURES)
        skin = rng.choice(_SKIN_TONES)

        desc = (
            f"A {age} {ethnicity} {gender} with {skin}, {features}. "
            f"They have {hair}."
        )

        if desc not in seen:
            seen.add(desc)
            descriptions.append(desc)

    return descriptions[:n]


# ---------------------------------------------------------------------------
# GPT-4o backend
# ---------------------------------------------------------------------------

PERSON_DESCRIPTION_SYSTEM = """You are a creative writer generating diverse, realistic descriptions of people for AI image generation.
Each description should specify:
- Age (18-70 years old)
- Ethnicity/background (diverse: East Asian, South Asian, Black/African, Hispanic/Latino, White/European, Middle Eastern, Mixed heritage, etc.)
- Gender (man/woman/non-binary person)
- Key facial features (e.g., sharp jawline, soft round face, prominent cheekbones, etc.)
- Hair (color, length, style)
- Any distinctive but natural features (e.g., freckles, a strong brow, dimples)
- Approximate skin tone descriptor

Keep descriptions concise (2-3 sentences). Do NOT mention emotions or expressions.
Output ONLY the description text, nothing else."""

PERSON_DESCRIPTION_USER = """Generate {n} unique, diverse person descriptions for portrait photography.
Output as a JSON array of strings.
Example format: ["Description 1...", "Description 2...", ...]
Generate exactly {n} descriptions."""


def generate_gpt_descriptions(
    api_key: str,
    model: str,
    n: int,
    batch_size: int = 10,
    max_retries: int = 3,
) -> list[str]:
    """Call GPT-4o to generate n person descriptions in batches."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for --backend gpt. Install: pip install openai")

    client = OpenAI(api_key=api_key)
    all_descriptions = []

    while len(all_descriptions) < n:
        batch_n = min(batch_size, n - len(all_descriptions))
        print(f"  Requesting {batch_n} descriptions from {model}...")

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": PERSON_DESCRIPTION_SYSTEM},
                        {"role": "user", "content": PERSON_DESCRIPTION_USER.format(n=batch_n)},
                    ],
                    temperature=0.9,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)

                if isinstance(parsed, list):
                    descriptions = parsed
                elif isinstance(parsed, dict):
                    for key in ["descriptions", "people", "persons", "items", "list"]:
                        if key in parsed:
                            descriptions = parsed[key]
                            break
                    else:
                        descriptions = next(
                            (v for v in parsed.values() if isinstance(v, list)), []
                        )
                else:
                    descriptions = []

                all_descriptions.extend(str(d).strip() for d in descriptions if d)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    return all_descriptions[:n]


# ---------------------------------------------------------------------------
# Manual backend
# ---------------------------------------------------------------------------

def load_manual_descriptions(filepath: str, n: int) -> list[str]:
    """Load descriptions from a plain text file (one per line)."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Manual descriptions file not found: {filepath}")
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    if len(lines) < n:
        print(f"WARNING: Only {len(lines)} descriptions in file, requested {n}. Reusing with repetition.")
        while len(lines) < n:
            lines.extend(lines)
    return lines[:n]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

EMOTION_DESCRIPTIONS = {
    "neutral": "completely neutral expressionless face, relaxed facial muscles, no emotion",
    "happy": "genuinely happy smiling expression, raised cheeks, crinkled eyes, joyful",
    "sad": "sad expression, downturned corners of mouth, slightly furrowed brow, melancholic",
    "angry": "angry expression, furrowed brow, tense jaw, intense stare, hostile",
    "surprised": "surprised expression, raised eyebrows, wide open eyes, slightly open mouth",
    "fearful": "fearful expression, wide eyes, raised eyebrows, slight tension, anxious",
    "disgusted": "disgusted expression, wrinkled nose, raised upper lip, aversion",
    "contempt": "contemptuous expression, slight asymmetric smirk, one raised brow",
}


def build_combined_prompt(person_desc: str, emotion_left: str, emotion_right: str) -> dict:
    """Build left, right, and combined side-by-side prompts."""
    emo_l_desc = EMOTION_DESCRIPTIONS[emotion_left]
    emo_r_desc = EMOTION_DESCRIPTIONS[emotion_right]

    base = f"professional portrait photograph, {person_desc}"
    quality = "high quality, photorealistic, sharp focus, studio lighting, 8k"

    prompt_left = f"{base}, {emo_l_desc}, {quality}"
    prompt_right = f"{base}, {emo_r_desc}, {quality}"

    combined_prompt = (
        f"Two side-by-side portrait photographs of the same person. "
        f"LEFT PHOTO: {base}, {emo_l_desc}. "
        f"RIGHT PHOTO: {base}, {emo_r_desc}. "
        f"Both photos: same identity, same lighting, same background. "
        f"{quality}. Split image, diptych format."
    )

    return {
        "prompt_left": prompt_left,
        "prompt_right": prompt_right,
        "combined_prompt": combined_prompt,
    }


def select_emotion_pairs(available_pairs: list, n: int, rng: random.Random) -> list:
    """Select n emotion pairs from the available pool."""
    if n >= len(available_pairs):
        return list(available_pairs[:n])
    return rng.sample(available_pairs, n)


def load_existing_prompts(output_file: Path) -> set:
    """Load existing person_ids for incremental append."""
    existing_ids = set()
    if output_file.exists():
        with jsonlines.open(output_file) as reader:
            for obj in reader:
                existing_ids.add(obj["person_id"])
    return existing_ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for the face emotion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  template  fast, no API needed, uses built-in diverse templates (default)
  gpt       calls GPT-4o (requires OPENAI_API_KEY)
  manual    reads descriptions from --manual-file (one description per line)

Examples:
  python step1_generate_prompts.py                          # 50 persons, template mode
  python step1_generate_prompts.py --backend gpt            # use GPT-4o
  python step1_generate_prompts.py --num-persons 200        # scale up
  python step1_generate_prompts.py --backend manual --manual-file my_descs.txt
        """
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", choices=["template", "gpt", "manual"], default="template",
                        help="Description generation backend (default: template)")
    parser.add_argument("--num-persons", type=int, default=None,
                        help="Override config num_persons")
    parser.add_argument("--pairs-per-person", type=int, default=None,
                        help="Override config emotion_pairs_per_person")
    parser.add_argument("--manual-file", type=str, default=None,
                        help="Path to plain-text file with descriptions (for --backend manual)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="(GPT only) descriptions per API call")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config(args.config)

    num_persons = args.num_persons or cfg["scale"]["num_persons"]
    pairs_per_person = args.pairs_per_person or cfg["scale"]["emotion_pairs_per_person"]

    output_file = Path(cfg["paths"]["prompts_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    available_emotion_pairs = cfg["emotion_pairs"]

    # Load existing to support incremental append
    existing_ids = load_existing_prompts(output_file)
    already_done = len(existing_ids)
    remaining = num_persons - already_done

    if remaining <= 0:
        print(f"Already have {already_done} persons (target: {num_persons}). Nothing to do.")
        return

    print(f"Backend: {args.backend}")
    print(f"Generating prompts for {remaining} persons (skipping {already_done} existing)")
    print(f"  Emotion pairs per person: {pairs_per_person}")
    print(f"  Total pairs to generate: {remaining * pairs_per_person}")

    # --- Fetch descriptions ---
    if args.backend == "template":
        descriptions = generate_template_descriptions(remaining, seed=args.seed + already_done)

    elif args.backend == "gpt":
        api_key = cfg["openai"]["api_key"]
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Use --backend template or set the env var.")
            sys.exit(1)
        descriptions = generate_gpt_descriptions(
            api_key=api_key,
            model=cfg["openai"]["model"],
            n=remaining,
            batch_size=args.batch_size,
            max_retries=cfg["openai"]["max_retries"],
        )

    elif args.backend == "manual":
        if not args.manual_file:
            print("ERROR: --manual-file required for --backend manual")
            sys.exit(1)
        descriptions = load_manual_descriptions(args.manual_file, remaining)

    else:
        print(f"ERROR: Unknown backend: {args.backend}")
        sys.exit(1)

    # --- Build JSONL records ---
    start_idx = already_done
    with jsonlines.open(output_file, mode="a") as writer:
        for i, desc in enumerate(descriptions[:remaining]):
            person_idx = start_idx + i
            person_id = f"p{person_idx:04d}"

            selected_pairs = select_emotion_pairs(available_emotion_pairs, pairs_per_person, rng)

            pairs = []
            for pair_idx, (emo_l, emo_r) in enumerate(selected_pairs):
                prompts = build_combined_prompt(desc, emo_l, emo_r)
                pairs.append({
                    "pair_id": f"{person_id}_pair{pair_idx:02d}",
                    "emotion_left": emo_l,
                    "emotion_right": emo_r,
                    **prompts,
                })

            record = {
                "person_id": person_id,
                "person_description": desc,
                "backend": args.backend,
                "pairs": pairs,
            }
            writer.write(record)

    # Summary
    total_pairs = 0
    with jsonlines.open(output_file) as reader:
        for obj in reader:
            total_pairs += len(obj["pairs"])

    print(f"\nDone. Output: {output_file}")
    print(f"  Total persons: {num_persons}")
    print(f"  Total pairs: {total_pairs}")


if __name__ == "__main__":
    main()

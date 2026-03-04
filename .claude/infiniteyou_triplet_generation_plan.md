# Plan: InfiniteYou Triplet Generation Script

## Context

You're generating face editing training data. Input: cropped face pairs (`left.png`, `right.png`) — same person, different expressions — from the flux_face_emotion pipeline. Goal: run each through InfiniteYou to produce richer backgrounds/hairstyles/makeup while preserving identity and expression.

Current bash scripts (`run_sample_0222_pair_batch.sh`, `run_sample_0222_v3.sh`) have three problems:
1. **Identity drift** — generated face diverges from input
2. **Expression loss** — expressions not preserved
3. **Scene inconsistency** — left/right outputs don't share the same scene

Root cause: scripts shell out to `test.py` per image (reloading ~20GB pipeline every time), don't use `control_image` for expression guidance, and don't use person-specific prompts.

## Output Triplets (per input pair)

| File | Source | Role |
|------|--------|------|
| `I_r.png` | Copy of `left.png` | Identity reference |
| `target.png` | InfiniteYou(`left.png`) | Target (left expression + rich scene) |
| `I_e.png` | InfiniteYou(`right.png`, same seed/scene) | Edited (right expression + same scene) |

## Solution: `generate_triplets.py`

Single Python script at `/scratch3/f007yzf/repos/InfiniteYou/generate_triplets.py` that imports `InfUFluxPipeline` directly (no subprocess calls).

### Key Improvements Over Current Scripts

1. **`control_image` for expression preservation**: Pass input face as `--control_image` to transfer 5 facial keypoints (eyes, nose, mouth corners) as spatial guidance → preserves expression geometry.

2. **Person-aware prompts with emotion labels**: Use `person_description` + emotion phrases from `crop_metadata.jsonl` in prompts. Scene portion identical between left/right; only emotion phrase differs.

3. **Both model versions + ArcFace selection**: Generate with both `sim_stage1` and `aes_stage2`, score identity similarity via ArcFace cosine distance, pick best per pair.

4. **Single pipeline load**: Load pipeline once per model version, process all 3,000 pairs, then switch. Total: 2 loads instead of 12,000.

### Architecture

```
Phase 1: Load sim_stage1 → generate all pairs (left+right) → save to staging/
Phase 2: Load aes_stage2 → generate all pairs (left+right) → save to staging/
Phase 3: ArcFace scoring → select best model per pair → copy to output/
Phase 4: Generate manifest.csv
```

### Smoke Test First Strategy

Before running the full 3,000 pairs, run a smoke test on 2-3 pairs to validate:
- Expression preservation (control_image working)
- Identity preservation (ArcFace > 0.5)
- Scene consistency (same background between target.png and I_e.png)
- Output quality (rich backgrounds, good aesthetics)

```bash
# Smoke test: 3 pairs, both models
python generate_triplets.py \
  --input_metadata /scratch3/f007yzf/flux_face_emotion/data_seed_2/cropped/crop_metadata.jsonl \
  --output_dir /scratch3/f007yzf/flux_face_emotion/data_seed_2/triplets_test/ \
  --batch_start 0 --batch_end 3
```

Review results visually, then tune parameters if needed before full batch run.

### Prompt Template

```python
EMOTION_PHRASES = {
    "happy": "genuinely happy smiling expression, raised cheeks, crinkled eyes",
    "sad": "sad expression, downturned corners of mouth, slightly furrowed brow",
    "angry": "angry expression, furrowed brow, tense jaw, intense stare",
    "fearful": "fearful expression, wide eyes, raised eyebrows, slight tension",
    "disgusted": "disgusted expression, wrinkled nose, raised upper lip",
    "surprised": "surprised expression, raised eyebrows, wide open eyes, slightly open mouth",
    "neutral": "neutral calm expression, relaxed face, direct eye contact",
    "contempt": "contemptuous expression, slight asymmetric smirk, raised chin",
}

SCENE_TEMPLATES = [
    "soft spring daylight, cherry blossom park, creamy bokeh, 85mm lens",
    "golden hour warm side light, summer beach, ocean haze, 50mm lens",
    "dappled afternoon light, autumn forest, golden leaves, 85mm lens",
    "cool overcast winter light, snow-dusted urban street, 35mm lens",
    "studio softbox, neutral gray seamless backdrop, 85mm lens",
    "dramatic Rembrandt lighting, dark studio, deep shadow, 50mm lens",
    "warm tungsten light, cozy cafe interior, exposed brick, 35mm lens",
    "soft window side light, library bookshelf, warm tones, 50mm lens",
    "neon-lit night cityscape, rain-wet street, vivid reflections, 35mm lens",
    "diffused overcast daylight, modern rooftop, contemporary urban, 85mm lens",
]

def build_prompts(record, scene_idx):
    person = record["person_description"]
    emo_left = EMOTION_PHRASES.get(record["emotion_left"], record["emotion_left"])
    emo_right = EMOTION_PHRASES.get(record["emotion_right"], record["emotion_right"])
    scene = SCENE_TEMPLATES[scene_idx % len(SCENE_TEMPLATES)]

    base = f"professional photograph, {person}, {{emotion}}, {scene}, photorealistic, sharp focus, 8k"
    return base.format(emotion=emo_left), base.format(emotion=emo_right)
```

Scene consistency: same seed + identical scene text → same background. Only the emotion phrase differs.

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_steps` | 28 | Slight speedup, minimal quality loss |
| `guidance_scale` | 3.5 | Default, good balance |
| `infusenet_conditioning_scale` | 1.0 | Max identity injection |
| `infusenet_guidance_start` | 0.0 | Full identity throughout |
| `infusenet_guidance_end` | 1.0 | Full identity throughout |
| `width × height` | 864 × 1152 | Portrait orientation, room for background |
| `quantize_8bit` | True | ~18GB VRAM, marginal quality loss |
| `control_image` | Same as `id_image` | Expression keypoint guidance |

### ArcFace Model Selection

```python
def select_best(sim_target_score, sim_ie_score, aes_target_score, aes_ie_score):
    sim_min = min(sim_target_score, sim_ie_score)
    aes_min = min(aes_target_score, aes_ie_score)
    return "sim_stage1" if sim_min >= aes_min else "aes_stage2"
```

Uses minimum of target + I_e scores — ensures both outputs maintain identity.

### Output Structure

```
output_dir/                          # e.g., /scratch3/f007yzf/flux_face_emotion/data_seed_2/triplets/
  p0000_pair00_s0/
    target.png
    I_e.png
    I_r.png
  p0000_pair01_s0/
    ...
  manifest.csv
  run.log

staging_dir/                         # temporary, can be deleted after
  sim_stage1/p0000_pair00_s0/{target,I_e}.png
  aes_stage2/p0000_pair00_s0/{target,I_e}.png
  progress.json
```

### Multi-GPU Parallelism

Support `--batch_start` / `--batch_end` for slicing across GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 python generate_triplets.py --batch_start 0 --batch_end 750
CUDA_VISIBLE_DEVICES=1 python generate_triplets.py --batch_start 750 --batch_end 1500
# etc.
```

### Resume Support

`progress.json` tracks completed `image_id` × `model_version` pairs. On restart, skip already-done items.

### Error Handling

- No face detected in input/output → log warning, skip pair, mark as `failed` in manifest
- CUDA OOM → `torch.cuda.empty_cache()`, log, skip
- Corrupt image → try/except on `Image.open()`, skip

### Manifest CSV Schema

```
image_id, person_id, pair_id, emotion_left, emotion_right, scene_idx, seed,
selected_model, arcface_sim_target, arcface_sim_ie, arcface_aes_target, arcface_aes_ie,
target_path, ie_path, ir_path, status
```

## Critical Files

- `/scratch3/f007yzf/repos/InfiniteYou/pipelines/pipeline_infu_flux.py` — `InfUFluxPipeline`, `extract_arcface_bgr_embedding` to import
- `/scratch3/f007yzf/repos/InfiniteYou/test.py` — reference for pipeline construction
- `/scratch3/f007yzf/flux_face_emotion/data_seed_2/cropped/crop_metadata.jsonl` — input metadata (3,000 records)

## Implementation Steps

1. Create `generate_triplets.py` with argparse, metadata loading, prompt building
2. Implement generation loop using `InfUFluxPipeline` directly with `control_image`
3. Implement ArcFace scoring (reuse pipeline's `extract_arcface_bgr_embedding`)
4. Implement model selection + finalization (copy winners, write manifest)
5. Add progress tracking (`progress.json`) for resume support
6. Add multi-GPU batch slicing
7. **Smoke test on 2-3 pairs first**, review results, tune parameters, then full batch

## Verification

1. **Smoke test**: Run on 2-3 pairs with `--batch_start 0 --batch_end 3`
2. **Check expression**: Visually compare input left/right expressions vs output target/I_e expressions
3. **Check identity**: Verify ArcFace scores > 0.5 (good) or > 0.6 (excellent)
4. **Check scene consistency**: Visually compare target.png and I_e.png backgrounds — should be nearly identical
5. **Check manifest**: Verify manifest.csv has correct paths and scores

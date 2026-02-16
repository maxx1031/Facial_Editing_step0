#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: InfiniteYou Augmentation Pipeline

使用 InfiniteYou 模型对已有的 cropped 数据进行泛化增强：
- 保持 identity 不变（使用 InfuseNet 控制）
- 保持 emotion 不变（在 prompt 中指定）
- 添加多样化的光照条件
- 保持与原有标签的一一对应

输入: data/cropped/ (来自 step3)
输出: data/augmented/ (新的泛化数据)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

# 添加 InfiniteYou 到 path
INFINITEYOU_PATH = '/scratch/GenAI_class/f007yzf/InfiniteYou'
sys.path.insert(0, INFINITEYOU_PATH)

from pipelines.pipeline_infu_flux import InfUFluxPipeline


# ===================== 光照条件模板 =====================
LIGHTING_CONDITIONS = [
    # 自然光
    "soft natural daylight, diffused sunlight",
    "golden hour warm sunlight, soft shadows",
    "overcast day soft diffused light, even illumination",
    "morning light, gentle warm tones",
    "afternoon sunlight, balanced exposure",

    # 室内光
    "soft studio lighting, professional portrait",
    "warm indoor ambient light, cozy atmosphere",
    "cool fluorescent office lighting",
    "window light from the side, natural indoor",
    "mixed indoor lighting, warm and cool tones",

    # 戏剧性光照
    "dramatic side lighting, strong contrast",
    "rim lighting from behind, silhouette edges",
    "butterfly lighting, classic portrait style",
    "rembrandt lighting, artistic shadows",
    "split lighting, half face illuminated",

    # 环境光
    "soft fill light, minimal shadows",
    "high key lighting, bright and even",
    "low key lighting, moody atmosphere",
    "bounce light, soft reflections",
    "ambient room light, natural feel",
]

# 情感描述模板
EMOTION_PROMPTS = {
    'neutral': 'completely neutral expressionless face, relaxed facial muscles, no emotion',
    'happy': 'happy expression, genuine smile, bright eyes, joyful',
    'sad': 'sad expression, downturned corners of mouth, slightly furrowed brow, melancholic',
    'angry': 'angry expression, furrowed brows, intense gaze, tense jaw',
    'surprised': 'surprised expression, wide eyes, raised eyebrows, open mouth slightly',
    'fear': 'fearful expression, wide eyes, tense face, worried look',
    'disgust': 'disgusted expression, wrinkled nose, curled lip',
    'contempt': 'contemptuous expression, slight smirk, raised eyebrow',
}


def build_prompt(person_description: str, emotion: str, lighting: str) -> str:
    """构建完整的生成 prompt"""
    emotion_desc = EMOTION_PROMPTS.get(emotion, emotion)

    prompt = (
        f"professional portrait photograph, "
        f"{person_description}, "
        f"{emotion_desc}, "
        f"{lighting}, "
        f"high quality, photorealistic, sharp focus, 8k"
    )
    return prompt


def load_metadata(cropped_dir: Path) -> List[Dict]:
    """加载 cropped 数据的 metadata"""
    metadata_path = cropped_dir / 'crop_metadata.jsonl'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata 文件不存在: {metadata_path}")

    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    return metadata


def get_unique_pairs(metadata: List[Dict]) -> Dict[str, Dict]:
    """提取唯一的 person + emotion pair 组合"""
    unique_pairs = {}

    for item in metadata:
        # 使用 pair_id 作为唯一键 (如 p0000_pair00)
        pair_id = item['pair_id']

        if pair_id not in unique_pairs:
            unique_pairs[pair_id] = {
                'person_id': item['person_id'],
                'pair_id': pair_id,
                'emotion_left': item['emotion_left'],
                'emotion_right': item['emotion_right'],
                'person_description': item['person_description'],
                'seeds': []
            }

        # 记录所有可用的 seed 变体
        unique_pairs[pair_id]['seeds'].append({
            'image_id': item['image_id'],
            'seed_idx': item['seed_idx'],
            'left_path': item['left_path'],
            'right_path': item['right_path'],
            'original_seed': item.get('seed', 0)
        })

    return unique_pairs


def setup_pipeline(args) -> InfUFluxPipeline:
    """初始化 InfiniteYou pipeline"""
    print("正在加载 InfiniteYou 模型...")

    infu_model_path = os.path.join(
        args.model_dir,
        f'infu_flux_{args.infu_flux_version}',
        args.model_version
    )
    insightface_root_path = os.path.join(
        args.model_dir,
        'supports',
        'insightface'
    )

    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )

    print("✓ 模型加载完成")
    return pipe


def generate_augmented_image(
    pipe: InfUFluxPipeline,
    id_image: Image.Image,
    prompt: str,
    seed: int,
    args
) -> Image.Image:
    """生成单张增强图像"""

    image = pipe(
        id_image=id_image,
        prompt=prompt,
        control_image=None,  # 不使用 control image，让模型自由生成姿态
        width=args.width,
        height=args.height,
        seed=seed,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        infusenet_conditioning_scale=args.infusenet_conditioning_scale,
        infusenet_guidance_start=args.infusenet_guidance_start,
        infusenet_guidance_end=args.infusenet_guidance_end,
        cpu_offload=args.cpu_offload,
    )

    return image


def process_pair(
    pipe: InfUFluxPipeline,
    pair_info: Dict,
    cropped_dir: Path,
    output_dir: Path,
    args,
    lighting_conditions: List[str],
) -> List[Dict]:
    """处理一个 person+emotion pair，生成多个光照变体"""

    results = []
    pair_id = pair_info['pair_id']
    person_description = pair_info['person_description']
    emotion_left = pair_info['emotion_left']
    emotion_right = pair_info['emotion_right']

    # 选择一个 seed 变体作为 ID 参考（选第一个或随机）
    seed_info = pair_info['seeds'][0]

    # 加载左右两边的 ID 图像
    left_id_path = cropped_dir.parent.parent / seed_info['left_path']
    right_id_path = cropped_dir.parent.parent / seed_info['right_path']

    if not left_id_path.exists() or not right_id_path.exists():
        print(f"  ⚠ 跳过 {pair_id}: 源图像不存在")
        return results

    left_id_image = Image.open(left_id_path).convert('RGB')
    right_id_image = Image.open(right_id_path).convert('RGB')

    # 为每个光照条件生成变体
    for light_idx, lighting in enumerate(lighting_conditions):
        # 生成随机种子
        seed = random.randint(0, 2**32 - 1)

        # 创建输出目录
        variant_id = f"{pair_id}_light{light_idx:02d}"
        variant_dir = output_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)

        # 生成左边图像（emotion_left）
        prompt_left = build_prompt(person_description, emotion_left, lighting)
        try:
            img_left = generate_augmented_image(
                pipe, left_id_image, prompt_left, seed, args
            )
            left_out_path = variant_dir / 'left.png'
            img_left.save(left_out_path)
        except Exception as e:
            print(f"  ⚠ 生成左图失败 {variant_id}: {e}")
            continue

        # 生成右边图像（emotion_right）使用相同种子
        prompt_right = build_prompt(person_description, emotion_right, lighting)
        try:
            img_right = generate_augmented_image(
                pipe, right_id_image, prompt_right, seed, args
            )
            right_out_path = variant_dir / 'right.png'
            img_right.save(right_out_path)
        except Exception as e:
            print(f"  ⚠ 生成右图失败 {variant_id}: {e}")
            continue

        # 记录 metadata
        result = {
            'image_id': variant_id,
            'person_id': pair_info['person_id'],
            'pair_id': pair_id,
            'lighting_idx': light_idx,
            'lighting_condition': lighting,
            'left_path': str(variant_dir / 'left.png'),
            'right_path': str(variant_dir / 'right.png'),
            'source_left_path': str(left_id_path),
            'source_right_path': str(right_id_path),
            'emotion_left': emotion_left,
            'emotion_right': emotion_right,
            'person_description': person_description,
            'prompt_left': prompt_left,
            'prompt_right': prompt_right,
            'seed': seed,
            'generation_params': {
                'width': args.width,
                'height': args.height,
                'guidance_scale': args.guidance_scale,
                'num_steps': args.num_steps,
                'infusenet_conditioning_scale': args.infusenet_conditioning_scale,
                'infusenet_guidance_start': args.infusenet_guidance_start,
                'infusenet_guidance_end': args.infusenet_guidance_end,
            }
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Step 6: InfiniteYou Augmentation - 生成多样化光照变体'
    )

    # 数据路径
    parser.add_argument('--cropped_dir',
                        default='/scratch3/f007yzf/flux_face_emotion/data/cropped',
                        help='cropped 数据目录')
    parser.add_argument('--output_dir',
                        default='/scratch3/f007yzf/flux_face_emotion/data/augmented',
                        help='输出目录')

    # 模型配置
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='ByteDance/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0')
    parser.add_argument('--model_version', default='aes_stage2')

    # GPU 设置
    parser.add_argument('--cuda_device', default=0, type=int,
                        help='CUDA 设备编号')

    # 生成参数
    parser.add_argument('--width', default=512, type=int,
                        help='输出宽度（与 cropped 一致）')
    parser.add_argument('--height', default=512, type=int,
                        help='输出高度（与 cropped 一致）')
    parser.add_argument('--guidance_scale', default=3.5, type=float,
                        help='CFG 强度')
    parser.add_argument('--num_steps', default=30, type=int,
                        help='去噪步数')

    # InfuseNet 参数
    parser.add_argument('--infusenet_conditioning_scale', default=1.2, type=float,
                        help='ID 控制强度（1.2 = 强控制保持身份）')
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float,
                        help='ID 控制开始时间')
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float,
                        help='ID 控制结束时间')

    # 内存优化
    parser.add_argument('--quantize_8bit', action='store_true',
                        help='启用 8-bit 量化')
    parser.add_argument('--cpu_offload', action='store_true',
                        help='启用 CPU 卸载')

    # 光照变体设置
    parser.add_argument('--num_lighting_variants', default=5, type=int,
                        help='每个 pair 生成的光照变体数量')
    parser.add_argument('--random_lighting', action='store_true',
                        help='随机选择光照条件（否则按顺序）')

    # 范围控制（用于并行或恢复）
    parser.add_argument('--start_idx', default=0, type=int,
                        help='起始 pair 索引')
    parser.add_argument('--end_idx', default=-1, type=int,
                        help='结束 pair 索引（-1 表示全部）')

    # 其他
    parser.add_argument('--dry_run', action='store_true',
                        help='仅显示将要执行的操作，不实际运行')

    args = parser.parse_args()

    # 设置 CUDA 设备
    torch.cuda.set_device(args.cuda_device)
    print(f"使用 GPU: {args.cuda_device}")

    # 路径设置
    cropped_dir = Path(args.cropped_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 metadata
    print(f"\n加载 metadata: {cropped_dir}")
    metadata = load_metadata(cropped_dir)
    print(f"  总记录数: {len(metadata)}")

    # 提取唯一 pairs
    unique_pairs = get_unique_pairs(metadata)
    pair_list = list(unique_pairs.values())
    print(f"  唯一 pair 数: {len(pair_list)}")

    # 范围控制
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(pair_list)
    pair_list = pair_list[start_idx:end_idx]
    print(f"  处理范围: [{start_idx}, {end_idx}) = {len(pair_list)} pairs")

    # 选择光照条件
    if args.random_lighting:
        lighting_conditions = random.sample(
            LIGHTING_CONDITIONS,
            min(args.num_lighting_variants, len(LIGHTING_CONDITIONS))
        )
    else:
        lighting_conditions = LIGHTING_CONDITIONS[:args.num_lighting_variants]

    print(f"\n光照条件 ({len(lighting_conditions)} 种):")
    for i, lc in enumerate(lighting_conditions):
        print(f"  {i}: {lc}")

    if args.dry_run:
        print("\n[DRY RUN] 将生成:")
        print(f"  - {len(pair_list)} pairs × {len(lighting_conditions)} 光照 = {len(pair_list) * len(lighting_conditions)} 组图像")
        print(f"  - 每组 2 张 (left + right) = {len(pair_list) * len(lighting_conditions) * 2} 张图像")
        return

    # 初始化 pipeline
    pipe = setup_pipeline(args)

    # 处理所有 pairs
    print(f"\n开始生成增强图像...")
    all_results = []

    for pair_info in tqdm(pair_list, desc="Processing pairs"):
        results = process_pair(
            pipe=pipe,
            pair_info=pair_info,
            cropped_dir=cropped_dir,
            output_dir=output_dir,
            args=args,
            lighting_conditions=lighting_conditions,
        )
        all_results.extend(results)

        # 定期保存 metadata
        if len(all_results) % 50 == 0:
            metadata_path = output_dir / 'augmented_metadata.jsonl'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 最终保存 metadata
    metadata_path = output_dir / 'augmented_metadata.jsonl'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 保存统计信息
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_pairs': len(pair_list),
        'lighting_variants': len(lighting_conditions),
        'total_generated': len(all_results),
        'total_images': len(all_results) * 2,
        'lighting_conditions': lighting_conditions,
        'generation_params': {
            'width': args.width,
            'height': args.height,
            'guidance_scale': args.guidance_scale,
            'num_steps': args.num_steps,
            'infusenet_conditioning_scale': args.infusenet_conditioning_scale,
            'infusenet_guidance_start': args.infusenet_guidance_start,
            'infusenet_guidance_end': args.infusenet_guidance_end,
        }
    }

    stats_path = output_dir / 'augmentation_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✓ 增强完成！")
    print(f"  生成 pair 数: {len(all_results)}")
    print(f"  生成图像数: {len(all_results) * 2}")
    print(f"  输出目录: {output_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  统计信息: {stats_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

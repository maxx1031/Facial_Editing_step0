#!/bin/bash
# InfiniteYou Augmentation Script
# 使用 InfiniteYou 对已有数据进行光照多样化增强

# 默认配置
CUDA_DEVICE=${1:-0}
NUM_LIGHTING=${2:-5}
START_IDX=${3:-0}
END_IDX=${4:--1}

# 激活环境
source /scratch3/f007yzf/conda/conda/etc/profile.d/conda.sh
conda activate infiniteyou

# 切换到项目目录
cd /scratch3/f007yzf/flux_face_emotion

echo "=========================================="
echo "InfiniteYou Augmentation Pipeline"
echo "=========================================="
echo "CUDA Device: ${CUDA_DEVICE}"
echo "Lighting Variants: ${NUM_LIGHTING}"
echo "Range: [${START_IDX}, ${END_IDX})"
echo "=========================================="

# 运行增强脚本
python step6_infiniteyou_augmentation.py \
    --cuda_device ${CUDA_DEVICE} \
    --num_lighting_variants ${NUM_LIGHTING} \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --infusenet_conditioning_scale 1.2 \
    --guidance_scale 3.5 \
    --num_steps 30 \
    --width 512 \
    --height 512

echo "=========================================="
echo "Augmentation completed!"
echo "=========================================="

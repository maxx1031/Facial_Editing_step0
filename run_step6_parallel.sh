#!/bin/bash
# Multi-GPU Parallel InfiniteYou Augmentation
# 使用多个 GPU 并行处理不同范围的数据

# 配置
NUM_LIGHTING=${1:-5}
GPUS=${2:-"0,3,4,5,6,7"}  # 可用的 GPU 列表，逗号分隔

# 解析 GPU 列表
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================="
echo "Multi-GPU Parallel Augmentation"
echo "=========================================="
echo "Available GPUs: ${GPUS} (${NUM_GPUS} GPUs)"
echo "Lighting Variants per pair: ${NUM_LIGHTING}"
echo "=========================================="

# 激活环境
source /scratch3/f007yzf/conda/conda/etc/profile.d/conda.sh
conda activate infiniteyou

cd /scratch3/f007yzf/flux_face_emotion

# 获取总 pair 数量
TOTAL_PAIRS=$(python -c "
import json
from pathlib import Path
metadata_path = Path('/scratch3/f007yzf/flux_face_emotion/data/cropped/crop_metadata.jsonl')
pairs = set()
with open(metadata_path) as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            pairs.add(item['pair_id'])
print(len(pairs))
")

echo "Total pairs to process: ${TOTAL_PAIRS}"

# 计算每个 GPU 处理的范围
PAIRS_PER_GPU=$((($TOTAL_PAIRS + $NUM_GPUS - 1) / $NUM_GPUS))

echo "Pairs per GPU: ~${PAIRS_PER_GPU}"
echo "=========================================="

# 启动并行任务
PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    GPU=${GPU_ARRAY[$i]}
    START_IDX=$((i * PAIRS_PER_GPU))
    END_IDX=$(((i + 1) * PAIRS_PER_GPU))

    # 最后一个 GPU 处理剩余的所有
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END_IDX=-1
    fi

    echo "Starting GPU ${GPU}: pairs [${START_IDX}, ${END_IDX})"

    CUDA_VISIBLE_DEVICES=${GPU} python step6_infiniteyou_augmentation.py \
        --cuda_device 0 \
        --num_lighting_variants ${NUM_LIGHTING} \
        --start_idx ${START_IDX} \
        --end_idx ${END_IDX} \
        --infusenet_conditioning_scale 1.2 \
        --guidance_scale 3.5 \
        --num_steps 30 \
        --width 512 \
        --height 512 \
        > logs/augmentation_gpu${GPU}.log 2>&1 &

    PIDS+=($!)
done

echo "=========================================="
echo "All ${NUM_GPUS} jobs started."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/augmentation_gpu*.log"
echo ""
echo "Waiting for all jobs to complete..."
echo "=========================================="

# 等待所有任务完成
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Job $pid completed"
done

echo "=========================================="
echo "All jobs completed!"
echo ""
echo "Merging results..."

# 合并所有输出 metadata (如果分开保存的话)
python -c "
import json
from pathlib import Path

output_dir = Path('/scratch3/f007yzf/flux_face_emotion/data/augmented')
all_results = []

# 收集所有生成的结果
metadata_path = output_dir / 'augmented_metadata.jsonl'
if metadata_path.exists():
    with open(metadata_path) as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

print(f'Total augmented pairs: {len(all_results)}')
print(f'Total images: {len(all_results) * 2}')
"

echo "=========================================="
echo "Done!"
echo "=========================================="

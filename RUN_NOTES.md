# Flux Face Emotion 运行笔记

## 0) 环境准备
```bash
cd /scratch3/f007yzf/flux_face_emotion
source ~/.bashrc
conda activate step1x
```

## 1) 检查可用 CUDA
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

筛选空闲 GPU（`util=0` 且 `memory.used < 200MB`）:
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
| tr -d ' ' | awk -F',' '$2<200 && $3==0 {print "GPU",$1,"is available"}'
```

## 2) 手动跑一个 seed（示例：seed=5）
```bash
cd /scratch3/f007yzf/flux_face_emotion
tmux new -s face_test_seed5
source ~/.bashrc
conda activate step1x

export RUN_SEED=5
python step1_generate_prompts.py --config config.yaml --seed 5 --num-persons 600 --pairs-per-person 5

export RUN_SEED=5
CUDA_VISIBLE_DEVICES=7 python step2_generate_images.py --config config.yaml
```

输出目录:
```bash
data_seed_5/
```

## 3) 自动跑 seed 4 和 5（tmux + 自动等待空闲 GPU）
脚本:
```bash
./run_seeds_4_5_tmux.sh
```

这个脚本会:
- 创建 `face_test_seed4` 和 `face_test_seed5`
- 执行 `source ~/.bashrc`
- 执行 `conda activate step1x`
- 先跑 step1，再跑 step2
- step2 启动前自动等待空闲 GPU

## 4) tmux 常用命令
查看会话:
```bash
tmux ls
```

进入会话:
```bash
tmux attach -t face_test_seed4
tmux attach -t face_test_seed5
```

从会话中退出（不停止任务）:
```bash
Ctrl+b d
```

新建窗口:
```bash
Ctrl+b c
```

## 5) 当前代码版本已固定内容
- Step1 的 emotion pairs 已固定顺序:
  1. `[happy, sad]`
  2. `[happy, angry]`
  3. `[happy, fearful]`
  4. `[happy, disgusted]`
  5. `[happy, surprised]`
- `config.yaml` 的 paths 使用 `data_seed_${RUN_SEED}` 格式。
- Step1/Step2 的配置读取支持字符串内嵌变量（如 `${RUN_SEED}`）。
- Step2 的统计文件写入各自 seed 目录（不再固定写到 `data/`）。

## 6) 可复制修改模板
```bash
cd /scratch3/f007yzf/flux_face_emotion
tmux new -s face_test_seedX
source ~/.bashrc
conda activate step1x

export RUN_SEED=X
python step1_generate_prompts.py --config config.yaml --seed X --num-persons 600 --pairs-per-person 5

export RUN_SEED=X
CUDA_VISIBLE_DEVICES=Y python step2_generate_images.py --config config.yaml
```

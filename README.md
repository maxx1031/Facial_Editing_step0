# FLUX Face Emotion Dataset Pipeline

一个用于生成面部情感配对图像数据集的完整流水线，使用 FLUX.2-klein 模型和面部识别技术。

## 环境配置检查

### 1. 系统要求

**已确认配置:**
- ✅ Python: 3.10.19
- ✅ CUDA: 12.4 (Driver: 570.195.03)
- ✅ GPU: 8x NVIDIA RTX 6000 Ada (每个 49GB VRAM)
- ✅ PyTorch: 2.5.1+cu124 (CUDA 可用)

**已安装的核心包:**
- ✅ diffusers: 0.37.0.dev0
- ✅ transformers: 4.55.0
- ✅ accelerate: 0.34.2
- ✅ insightface: 0.7.3
- ✅ hsemotion: 0.3.0

### 2. 需要设置的环境变量

**必需 (如果使用 GPT-4o 生成 prompts):**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

**可选 (如果使用 FLUX.2-klein-9B):**
```bash
export HF_TOKEN="your-huggingface-token-here"
```

**注意:** 当前配置使用 `FLUX.2-klein-4B` (Apache 2.0)，不需要 HF_TOKEN。

### 3. 完整安装步骤

如果需要重新安装环境:

```bash
# 1. 创建虚拟环境 (推荐)
conda create -n flux_emotion python=3.10
conda activate flux_emotion

# 2. 安装依赖包
cd /scratch3/f007yzf/flux_face_emotion
pip install -r requirements.txt

# 3. 设置环境变量 (添加到 ~/.bashrc 以永久保存)
export OPENAI_API_KEY="sk-..."  # 仅在使用 GPT 时需要
export HF_TOKEN="hf_..."        # 仅在使用 FLUX.2-klein-9B 时需要
```

## 项目结构

```
flux_face_emotion/
├── config.yaml                   # 配置文件
├── requirements.txt              # Python 依赖
├── run_pipeline.py              # 主流水线运行器
├── step1_generate_prompts.py    # 步骤1: 生成 prompts
├── step2_generate_images.py     # 步骤2: 生成图像
├── step3_crop_pairs.py          # 步骤3: 裁剪图像对
├── step4_filter_pairs.py        # 步骤4: 过滤图像对
├── step5_package_dataset.py     # 步骤5: 打包数据集
└── data/                        # 数据目录 (自动创建)
    ├── prompts.jsonl            # 生成的 prompts
    ├── raw/                     # 原始生成图像
    ├── cropped/                 # 裁剪后的图像对
    ├── filtered/                # 过滤后的图像对
    └── dataset/                 # 最终数据集
```

## 流水线步骤说明

### Step 1: 生成 Prompts (GPT-4o 或模板)
生成人物描述和情感转换 prompts。

**支持三种模式:**
- `template`: 使用内置模板 (无需 API)
- `gpt`: 使用 GPT-4o 生成多样化描述 (需要 OPENAI_API_KEY)
- `manual`: 使用手动提供的描述文件

**输出:** `data/prompts.jsonl`

### Step 2: 生成图像 (FLUX.2-klein)
使用 FLUX.2-klein 模型生成 1056×528 分辨率的并排情感配对图像。

**配置:**
- 模型: FLUX.2-klein-4B (8GB VRAM)
- 分辨率: 1056×528
- 推理步数: 4 steps
- 每对生成 5 个种子

**输出:** `data/raw/` + `data/raw/metadata.jsonl`

### Step 3: 裁剪图像对
将 1056×528 的并排图像分割为两张 512×512 的独立图像。

**输出:** `data/cropped/` + `data/cropped/crop_metadata.jsonl`

### Step 4: 过滤图像对
三层过滤确保数据质量:

1. **Layer 1 - 面部检测:** 确保每张图像恰好有一张人脸
2. **Layer 2 - ArcFace 相似度:** 确认两张图像是同一个人 (threshold: 0.5)
3. **Layer 3 - 情感识别:** 验证情感标签正确性 (使用 HSEmotion)

**输出:** `data/filtered/` + `data/filter_stats.json`

### Step 5: 打包数据集
将过滤后的图像打包为 Hugging Face datasets 格式。

**输出:** `data/dataset/` (包含 .arrow 文件和 metadata.jsonl)

## 使用方法

### 快速开始 (验证模式)

```bash
cd /scratch3/f007yzf/flux_face_emotion

# 运行完整流水线 (使用模板模式，50人 x 3对 = 150对)
python run_pipeline.py --backend template

# 或者分步运行
python run_pipeline.py --steps 1 2 3 4 5
```

### 常用命令

```bash
# 1. 查看当前状态
python run_pipeline.py --status

# 2. 使用 GPT-4o 生成更多样化的 prompts
python run_pipeline.py --steps 1 --backend gpt --num-persons 50 --pairs-per-person 3

# 3. 仅运行图像生成 (如果已有 prompts)
python run_pipeline.py --steps 2 --seeds 5

# 4. 先运行 Step 4 统计模式 (检查过滤阈值)
python run_pipeline.py --steps 4 --stats-only

# 5. 调整 ArcFace 阈值后实际过滤
python run_pipeline.py --steps 4 --arcface-threshold 0.6

# 6. 从某步恢复 (如果中断)
python run_pipeline.py --from-step 3

# 7. 运行特定步骤
python run_pipeline.py --steps 4 5

# 8. 跳过某些步骤
python run_pipeline.py --skip-steps 1 2

# 9. 试运行 (查看会执行什么，但不实际运行)
python run_pipeline.py --dry-run
```

### 生产规模运行

编辑 `config.yaml` 扩大规模:

```yaml
scale:
  num_persons: 2000        # 从 50 增加到 2000-3000
  emotion_pairs_per_person: 5  # 从 3 增加到 5-8
```

然后运行:

```bash
python run_pipeline.py --backend gpt --num-persons 2000 --pairs-per-person 5
```

### 单步骤详细使用

```bash
# Step 1: 生成 prompts
python step1_generate_prompts.py --backend template --num-persons 50 --pairs-per-person 3

# Step 2: 生成图像
python step2_generate_images.py --seeds 5

# Step 3: 裁剪
python step3_crop_pairs.py

# Step 4: 过滤 (先统计)
python step4_filter_pairs.py --stats-only

# Step 4: 过滤 (实际执行)
python step4_filter_pairs.py --arcface-threshold 0.5

# Step 5: 打包
python step5_package_dataset.py
```

## 配置文件说明 (config.yaml)

### 关键配置项

```yaml
# OpenAI 配置 (仅 GPT 模式需要)
openai:
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}

# 图像生成配置
generation:
  model_id: black-forest-labs/FLUX.2-klein-4B  # 或 FLUX.2-klein-9B
  width: 1056
  height: 528
  num_steps: 4              # FLUX.2-klein 推荐 4 步
  guidance_scale: 1.0
  seeds_per_pair: 5         # 每对生成 5 个随机种子
  device: cuda

# 规模配置
scale:
  num_persons: 50           # 验证: 50, 生产: 2000-3000
  emotion_pairs_per_person: 3  # 验证: 3, 生产: 5-8

# 过滤配置
filtering:
  arcface_threshold: 0.5    # ArcFace 相似度阈值 (0-1)
  require_exact_one_face: true
  face_det_score_threshold: 0.5
  emotion_model: enet_b0_8_best_afew

# 情感对
emotion_pairs:
  - [neutral, happy]
  - [neutral, sad]
  - [neutral, angry]
  - [happy, surprised]
  # ... 更多情感转换对
```

## 监控和调试

### 查看进度

```bash
# 查看当前状态
python run_pipeline.py --status

# 查看文件统计
ls -lh data/raw/*.jpg | wc -l      # 生成的图像数
cat data/prompts.jsonl | wc -l     # prompts 数量
cat data/filtered/filter_metadata.jsonl | wc -l  # 通过过滤的图像对数
```

### 查看过滤统计

```bash
# 查看 filter_stats.json
cat data/filter_stats.json | python -m json.tool
```

### GPU 使用监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看当前使用情况
nvidia-smi
```

## 预期输出

### 验证阶段 (默认配置)
- **输入:** 50 人 × 3 对 = 150 个情感对
- **生成:** 150 × 5 种子 = 750 张图像
- **裁剪:** 750 × 2 = 1,500 张单独图像
- **过滤后:** 约 300-600 张图像对 (取决于质量)

### 生产阶段 (扩大规模)
- **输入:** 2000 人 × 5 对 = 10,000 个情感对
- **生成:** 10,000 × 5 种子 = 50,000 张图像
- **裁剪:** 50,000 × 2 = 100,000 张单独图像
- **过滤后:** 约 20,000-40,000 张图像对

## 资源需求

### GPU 内存
- **FLUX.2-klein-4B:** ~8GB VRAM
- **FLUX.2-klein-9B:** ~16GB VRAM
- **InsightFace + HSEmotion:** ~2GB VRAM

**推荐:** 使用 GPU 0, 3-7 (当前 GPU 1-2 正在使用中)

### 存储空间
- **验证阶段:** ~5-10GB
- **生产阶段:** ~200-500GB

### 运行时间估算
不提供具体时间估算，取决于:
- GPU 型号和数量
- 批处理大小
- 网络速度 (下载模型)
- 过滤通过率

## 常见问题

### Q1: CUDA out of memory
**解决方案:**
- 使用 FLUX.2-klein-4B 而不是 9B
- 减少批处理大小
- 使用空闲的 GPU (如 GPU 3-7)

### Q2: 环境变量未设置
**解决方案:**
```bash
export OPENAI_API_KEY="your-key"
# 或添加到 ~/.bashrc:
echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### Q3: 过滤通过率太低
**解决方案:**
- 先运行 `--stats-only` 查看分层统计
- 调整 `arcface_threshold` (降低到 0.4-0.5)
- 检查 `face_det_score_threshold`
- 增加 `seeds_per_pair` 以生成更多候选

### Q4: 中断后如何恢复
**解决方案:**
```bash
# 流水线会自动保存检查点到 data/.pipeline_checkpoint.json
# 从中断的步骤恢复:
python run_pipeline.py --from-step 3
```

## 高级用法

### 使用手动描述文件

```bash
# 1. 创建手动描述文件 descriptions.txt (每行一个描述)
cat > descriptions.txt << EOF
A young woman with long brown hair and green eyes
An elderly man with gray beard and glasses
A teenager with curly black hair
EOF

# 2. 运行 Step 1 使用手动模式
python step1_generate_prompts.py --backend manual --manual-file descriptions.txt
```

### 并行运行多个 GPU

```bash
# GPU 0: 生成前 25 人
CUDA_VISIBLE_DEVICES=0 python step2_generate_images.py --start-idx 0 --end-idx 25 &

# GPU 3: 生成后 25 人
CUDA_VISIBLE_DEVICES=3 python step2_generate_images.py --start-idx 25 --end-idx 50 &

wait  # 等待所有任务完成
```

### 自定义情感对

编辑 `config.yaml`:

```yaml
emotion_pairs:
  - [neutral, happy]
  - [happy, sad]
  - [sad, angry]
  # 添加您自己的情感转换对
```

## 许可证

- **FLUX.2-klein-4B:** Apache 2.0 (商用友好)
- **FLUX.2-klein-9B:** 非商业许可证

## 技术栈

- **图像生成:** FLUX.2-klein (Black Forest Labs)
- **面部识别:** InsightFace (ArcFace)
- **情感识别:** HSEmotion
- **Prompt 生成:** GPT-4o (可选) 或模板
- **数据处理:** datasets, PyTorch, diffusers

## 支持

如遇问题:
1. 运行 `python run_pipeline.py --status` 检查状态
2. 查看 `data/filter_stats.json` 了解过滤详情
3. 使用 `--dry-run` 预览将要执行的命令
4. 检查 GPU 使用情况: `nvidia-smi`

---

**祝您使用愉快！**

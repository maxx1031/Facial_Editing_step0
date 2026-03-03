# FLUX Face Emotion Dataset Pipeline

一个用于生成面部情感配对图像数据集的完整流水线，使用 FLUX.2-klein 模型和面部识别技术。

## 当前主流程（v2，推荐）

当前代码主流程已经升级为 Step1X 双图训练格式，建议按下面顺序运行：

1. `step1_generate_prompts.py`
2. `step2_generate_images.py`
3. `step3_crop_pairs.py`
4. `step4_filter_pairs.py`
5. `step5_build_reference_pool.py`
6. `step6_construct_triplets.py`
7. `step8_mllm_instructions.py`（可选，生成 FACS 指令）
8. `step7_package_step1x.py`（输出 Step1X 格式）

一键运行（默认不含 MLLM 步骤）：

```bash
python run_pipeline.py
```

若要包含 FACS 指令生成：

```bash
python run_pipeline.py --steps 7 8 --model-path /path/to/Qwen2.5-VL-7B-Instruct --use-facs
```

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
├── step5_build_reference_pool.py # 步骤5: 构建 I_r 参考池
├── step6_construct_triplets.py   # 步骤6: 构建三元组
├── step8_mllm_instructions.py    # 步骤7: 生成 FACS 指令（可选）
├── step7_package_step1x.py       # 步骤8: 打包 Step1X 数据
├── step5_package_dataset.py      # 旧版 v1 打包脚本（兼容保留）
└── data/                        # 数据目录 (自动创建)
    ├── prompts.jsonl            # 生成的 prompts
    ├── raw/                     # 原始生成图像
    ├── cropped/                 # 裁剪后的图像对
    ├── filtered/                # 过滤后的图像对
    ├── reference_pool/          # I_r 参考池
    ├── triplets/                # (I_e, I_r, I_e_edit) 三元组
    └── dataset_v2/              # Step1X 双图训练格式
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
- 分辨率: 1058×528
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

### Step 5: 构建 I_r 参考池
从真实数据与 Flux 合成数据构建目标表情参考池（I_r）。

**输出:** `data/reference_pool/` + `data/reference_pool/pool_metadata.jsonl`

### Step 6: 构建三元组
按目标表情把 `(I_e, I_e_edit)` 与 `I_r` 组合为 `(I_e, I_r, I_e_edit)`，支持每对采样 K 个 `I_r`。

**输出:** `data/triplets/triplet_metadata.jsonl`

### Step 7: MLLM 生成 FACS 指令（可选）
使用 Qwen2.5-VL 分析 `(I_e, I_r)`，写入 `facs_instruction` 到 triplets 元数据。

**输出:** 更新 `data/triplets/triplet_metadata.jsonl`

### Step 8: 打包 Step1X 双图训练格式
将 triplets 打包为 Step1X 所需 `images/ + metadata.json`。

**输出:** `data/dataset_v2/`

## 使用方法

### 快速开始 (验证模式)

```bash
cd /scratch3/f007yzf/flux_face_emotion

# 运行完整流水线 (使用模板模式，50人 x 3对 = 150对)
python run_pipeline.py --backend template

# 或者分步运行
python run_pipeline.py --steps 1 2 3 4 5 6 8
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

# 7. 构建参考池 + 三元组
python run_pipeline.py --steps 5 6 --pool-source both --k 3

# 8. 生成 FACS 指令并打包（需要本地 Qwen2.5-VL）
python run_pipeline.py --steps 7 8 --model-path /path/to/Qwen2.5-VL-7B-Instruct --use-facs

# 9. 跳过某些步骤
python run_pipeline.py --skip-steps 1 2

# 10. 试运行 (查看会执行什么，但不实际运行)
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

# Step 5: 构建 I_r 参考池
python step5_build_reference_pool.py --source both

# Step 6: 构建三元组
python step6_construct_triplets.py --k 3

# Step 7 (可选): MLLM FACS 指令
python step8_mllm_instructions.py --model-path /path/to/Qwen2.5-VL-7B-Instruct

# Step 8: Step1X 打包
python step7_package_step1x.py --use-facs
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

---

## Step 6: InfiniteYou 光照增强 (新功能)

使用 InfiniteYou 模型对已有的 cropped 数据进行泛化增强：
- **保持 identity 不变**: 使用 InfuseNet 控制身份特征
- **保持 emotion 不变**: 在 prompt 中指定相同的情感
- **添加多样化光照**: 20 种预定义光照条件可选
- **一一对应**: 每个原始 pair 生成多个光照变体

### 光照条件示例

```
- soft natural daylight, diffused sunlight
- golden hour warm sunlight, soft shadows
- dramatic side lighting, strong contrast
- rim lighting from behind, silhouette edges
- soft studio lighting, professional portrait
- ... (共 20 种)
```

### 使用方法

```bash
# 激活 InfiniteYou 环境
conda activate infiniteyou

# 单 GPU 运行 (GPU 0, 每个 pair 5 种光照变体)
./run_step6_augmentation.sh 0 5

# 指定范围 (GPU 0, 5 种光照, pair 0-50)
./run_step6_augmentation.sh 0 5 0 50

# 多 GPU 并行运行 (6 个 GPU)
./run_step6_parallel.sh 5 "0,3,4,5,6,7"
```

### 详细参数

```bash
python step6_infiniteyou_augmentation.py \
    --cuda_device 0 \
    --num_lighting_variants 5 \
    --infusenet_conditioning_scale 1.2 \
    --guidance_scale 3.5 \
    --num_steps 30 \
    --width 512 \
    --height 512 \
    --start_idx 0 \
    --end_idx 100
```

**关键参数说明:**
- `infusenet_conditioning_scale`: ID 控制强度 (推荐 1.0-1.3)
- `num_lighting_variants`: 每个 pair 生成的光照变体数量
- `start_idx/end_idx`: 用于并行处理或恢复

### 输出结构

```
data/augmented/
├── p0000_pair00_light00/
│   ├── left.png
│   └── right.png
├── p0000_pair00_light01/
│   ├── left.png
│   └── right.png
├── ...
├── augmented_metadata.jsonl
└── augmentation_stats.json
```

### 预期输出

以 150 pairs × 5 光照变体为例:
- **输入:** 150 个原始 emotion pairs
- **输出:** 150 × 5 = 750 个增强 pairs
- **图像数:** 750 × 2 = 1,500 张图像

---

## 技术栈

- **图像生成:** FLUX.2-klein (Black Forest Labs)
- **ID 保持增强:** InfiniteYou (ByteDance) - 基于 FLUX.2 系列工作流
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

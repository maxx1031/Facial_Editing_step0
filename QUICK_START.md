# FLUX 面部情感数据集流水线 - 快速开始指南

## 当前状态

✅ **环境已配置完成**
- Python 3.10.19
- PyTorch 2.5.1+cu124 (CUDA 12.4)
- diffusers 0.37.0.dev0
- 所有依赖包已安装
- 代码已修复并可以运行

⚠️ **需要您完成的最后一步**
- 设置 Hugging Face Token (HF_TOKEN)

## 执行步骤

### 步骤 1: 设置 Hugging Face Token

```bash
# 1. 访问 https://huggingface.co/settings/tokens
#    创建一个新的 token (选择 Read 权限即可)

# 2. 接受 FLUX 模型的许可协议
#    访问: https://huggingface.co/black-forest-labs/FLUX.1-schnell
#    点击 "Agree and access repository"

# 3. 设置环境变量
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 可选：永久保存到 ~/.bashrc
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 2: 验证环境

```bash
cd /scratch3/f007yzf/flux_face_emotion

# 运行环境检查脚本
python check_environment.py
```

### 步骤 3: 运行流水线

```bash
# 查看当前状态（prompts 已生成）
python run_pipeline.py --status

# 从 Step 2 开始运行（生成图像）
# 使用 1 个种子进行快速测试
python run_pipeline.py --from-step 2 --seeds 1

# 或者运行完整的 Step 2-5
python run_pipeline.py --from-step 2
```

### 步骤 4: 监控进度

```bash
# 在另一个终端窗口中监控 GPU 使用
watch -n 1 nvidia-smi

# 查看生成的图像数量
ls -lh data/raw/*.png | wc -l

# 查看当前流水线状态
python run_pipeline.py --status
```

## 当前配置说明

### 模型配置 (config.yaml)

```yaml
model_id: black-forest-labs/FLUX.1-schnell
pipeline_class: FluxPipeline
width: 1056
height: 528
num_steps: 4              # 快速模式，4 步推理
guidance_scale: 0.0       # schnell 不使用引导
```

### 数据规模（验证阶段）

- 人数: 50
- 每人情感对: 3
- 每对种子数: 5 (可通过 --seeds 参数修改)
- 预计生成图像: 750 张 (150 对 × 5 种子)

### GPU 资源需求

- FLUX.1-schnell: ~8-12 GB VRAM
- 当前可用: 8x RTX 6000 Ada (每个 49 GB)
- 推荐使用: GPU 0, 3-7 (GPU 1-2 正在使用中)

## 常用命令

```bash
# 1. 仅生成 1 个种子进行快速测试
python run_pipeline.py --steps 2 --seeds 1

# 2. 生成全部 5 个种子
python run_pipeline.py --steps 2 --seeds 5

# 3. 运行完整流水线（Step 2-5）
python run_pipeline.py --from-step 2

# 4. 运行特定步骤
python run_pipeline.py --steps 3 4 5  # 裁剪、过滤、打包

# 5. 查看过滤统计（Step 4 统计模式）
python run_pipeline.py --steps 4 --stats-only

# 6. 调整 ArcFace 阈值后过滤
python run_pipeline.py --steps 4 --arcface-threshold 0.6
```

## 预期输出

### Step 2: 生成图像
- 输入: 150 个情感对（来自 data/prompts.jsonl）
- 输出: data/raw/pXXXX/pairXX/seed_X.png
- 元数据: data/raw/metadata.jsonl

### Step 3: 裁剪图像对
- 将 1056×528 图像分割为两张 512×512 图像
- 输出: data/cropped/
- 元数据: data/cropped/crop_metadata.jsonl

### Step 4: 过滤
- 三层过滤: 面部检测 → ArcFace 同一性 → 情感验证
- 输出: data/filtered/
- 统计: data/filter_stats.json

### Step 5: 打包数据集
- 输出: data/dataset/ (Hugging Face datasets 格式)

## 故障排除

### 问题 1: HF_TOKEN 未设置
```bash
# 错误信息: "401 Client Error: Unauthorized"
# 解决方案: 设置 HF_TOKEN 环境变量
export HF_TOKEN="hf_xxxxx"
```

### 问题 2: CUDA out of memory
```bash
# 解决方案: 使用其他空闲的 GPU
CUDA_VISIBLE_DEVICES=3 python run_pipeline.py --from-step 2
```

### 问题 3: 中断后恢复
```bash
# 流水线会自动保存检查点
# 直接从中断的步骤恢复即可
python run_pipeline.py --from-step 2
```

### 问题 4: 查看详细错误
```bash
# 直接运行单个步骤脚本查看完整输出
python step2_generate_images.py --seeds 1
```

## 性能优化建议

### 1. 并行生成（使用多个 GPU）

如果要加快生成速度，可以手动分配任务到不同 GPU：

```bash
# 暂不推荐，因为代码需要修改才能支持批量分割
# 目前建议单 GPU 顺序执行
```

### 2. 调整批处理大小

当前配置为单张生成。如果 GPU 内存充足，可以修改代码支持批处理。

### 3. 使用更快的模型

```yaml
# FLUX.1-schnell: 4 steps (当前使用，最快)
# FLUX.1-dev: 28-50 steps (质量更高但更慢)
```

## 下一步

1. ✅ 设置 HF_TOKEN
2. ✅ 运行环境检查: `python check_environment.py`
3. ✅ 测试生成: `python run_pipeline.py --steps 2 --seeds 1`
4. ✅ 完整运行: `python run_pipeline.py --from-step 2`

## 技术支持

如遇问题，检查以下内容：
1. `python check_environment.py` - 环境配置
2. `python run_pipeline.py --status` - 流水线状态
3. `nvidia-smi` - GPU 使用情况
4. `data/filter_stats.json` - 过滤统计（Step 4 后）

---

**准备好了吗？设置 HF_TOKEN 后就可以开始了！** 🚀

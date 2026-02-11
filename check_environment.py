#!/usr/bin/env python3
"""
Environment check script for FLUX Face Emotion Pipeline.
Verifies all dependencies and configurations are ready.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (需要 Python 3.9+)")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA {cuda_version} 可用")
            print(f"   - {device_count} 个 GPU 设备")
            print(f"   - 主设备: {device_name}")

            # 显示所有 GPU
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024**3)
                print(f"   - GPU {i}: {name} ({total_mem_gb:.1f} GB)")
            return True
        else:
            print("❌ CUDA 不可用")
            return False
    except ImportError:
        print("❌ PyTorch 未安装")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False


def check_env_vars():
    """Check environment variables."""
    results = {}

    # OPENAI_API_KEY (optional for GPT mode)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        masked = openai_key[:7] + "..." + openai_key[-4:] if len(openai_key) > 11 else "***"
        print(f"✅ OPENAI_API_KEY: {masked}")
        results['openai'] = True
    else:
        print("⚠️  OPENAI_API_KEY: 未设置 (使用 --backend gpt 时需要)")
        results['openai'] = False

    # HF_TOKEN (optional for FLUX.2-klein-9B)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        masked = "hf_" + "..." + hf_token[-4:] if len(hf_token) > 7 else "***"
        print(f"✅ HF_TOKEN: {masked}")
        results['hf'] = True
    else:
        print("⚠️  HF_TOKEN: 未设置 (使用 FLUX.2-klein-9B 时需要)")
        results['hf'] = False

    return results


def check_config():
    """Check if config.yaml exists and is valid."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ config.yaml 不存在")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check key sections
        required_sections = ['generation', 'scale', 'filtering', 'paths', 'emotions', 'emotion_pairs']
        missing = [s for s in required_sections if s not in config]

        if missing:
            print(f"❌ config.yaml 缺少部分: {', '.join(missing)}")
            return False

        print(f"✅ config.yaml 有效")

        # Show key settings
        gen = config.get('generation', {})
        scale = config.get('scale', {})
        print(f"   - 模型: {gen.get('model_id', 'N/A')}")
        print(f"   - 设备: {gen.get('device', 'N/A')}")
        print(f"   - 规模: {scale.get('num_persons', 'N/A')} 人 × {scale.get('emotion_pairs_per_person', 'N/A')} 对")

        return True
    except Exception as e:
        print(f"❌ config.yaml 解析错误: {e}")
        return False


def check_files():
    """Check if all pipeline scripts exist."""
    scripts = [
        "run_pipeline.py",
        "step1_generate_prompts.py",
        "step2_generate_images.py",
        "step3_crop_pairs.py",
        "step4_filter_pairs.py",
        "step5_package_dataset.py",
    ]

    all_exist = True
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} 不存在")
            all_exist = False

    return all_exist


def main():
    print("="*60)
    print("FLUX Face Emotion Pipeline - 环境检查")
    print("="*60)

    results = {}

    print("\n📌 Python 版本:")
    results['python'] = check_python_version()

    print("\n📌 CUDA 和 GPU:")
    results['cuda'] = check_cuda()

    print("\n📌 核心依赖包:")
    core_packages = [
        ('torch', None),
        ('diffusers', None),
        ('transformers', None),
        ('accelerate', None),
    ]

    for pkg, import_name in core_packages:
        results[pkg] = check_package(pkg, import_name)

    print("\n📌 面部和情感识别:")
    face_packages = [
        ('insightface', None),
        ('onnxruntime', None),
        ('cv2', 'cv2'),
        ('hsemotion', None),
    ]

    for pkg, import_name in face_packages:
        results[pkg] = check_package(pkg, import_name)

    print("\n📌 图像和数据处理:")
    data_packages = [
        ('PIL', 'PIL'),
        ('numpy', None),
        ('datasets', None),
        ('tqdm', None),
        ('jsonlines', None),
        ('yaml', 'yaml'),
    ]

    for pkg, import_name in data_packages:
        results[pkg] = check_package(pkg, import_name)

    print("\n📌 环境变量:")
    env_results = check_env_vars()
    results.update(env_results)

    print("\n📌 配置文件:")
    results['config'] = check_config()

    print("\n📌 流水线脚本:")
    results['scripts'] = check_files()

    # Summary
    print("\n" + "="*60)
    print("检查总结:")
    print("="*60)

    critical_checks = ['python', 'cuda', 'torch', 'diffusers', 'transformers',
                      'insightface', 'hsemotion', 'config', 'scripts']
    critical_passed = all(results.get(check, False) for check in critical_checks)

    optional_checks = ['openai', 'hf']

    if critical_passed:
        print("✅ 所有关键组件已就绪！")
        print("\n您可以开始运行流水线:")
        print("  python run_pipeline.py --backend template")
        print("\n或查看状态:")
        print("  python run_pipeline.py --status")

        if not results.get('openai', False):
            print("\n注意: 如需使用 GPT-4o 生成 prompts，请设置 OPENAI_API_KEY:")
            print("  export OPENAI_API_KEY='your-key-here'")

        if not results.get('hf', False):
            print("\n注意: 当前使用 FLUX.2-klein-4B (无需 HF token)")
            print("  如需使用 FLUX.2-klein-9B，请设置 HF_TOKEN")

        return 0
    else:
        print("❌ 存在缺失的关键组件，请先安装:")
        print("  pip install -r requirements.txt")

        missing = [check for check in critical_checks if not results.get(check, False)]
        print(f"\n缺失的组件: {', '.join(missing)}")

        return 1


if __name__ == "__main__":
    sys.exit(main())

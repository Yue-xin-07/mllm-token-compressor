# 架构与目录说明 (Architecture & File Manifest)

## 项目规则与版本控制

- `AGENTS.md`: AI 开发规则、Memory Bank 更新要求、AutoDL 远程连接说明。
- `.gitignore`: 排除 Python 缓存、虚拟环境、Notebook 检查点、模型权重、数据集、运行产物和本地临时文件。
- `.venv/`: 项目隔离 Python 3.11 环境；PyTorch 2.5.1+cu121、Transformers 4.51.3、vLLM 0.7.3、FlashAttention-2 2.8.3.post2、Accelerate 1.15.0、setuptools 80.10.2。

## 记忆库

- `memory-bank/architecture.md`: 本文件，记录项目文件和目录职责。
- `memory-bank/implementation-plan.md`: Step 0–4 的实现任务和验收标准。
- `memory-bank/progress.md`: 当前阶段、已完成事项、下一步与阻碍。
- `memory-bank/tech-stack.md`: 目标模型、技术栈、性能目标和硬件约束。

## 源码与验证目录

- `compressor/`: Token 压缩核心算法源码。
- `scripts/check_env.py`: 检查 Python、PyTorch、CUDA/GPU、Transformers、FlashAttention-2、vLLM 与 Accelerate；可运行 Qwen2.5-VL 短文本生成冒烟验证。
- `scripts/download_model.py`: 使用 Hugging Face Hub 下载或续传模型快照，支持自定义 endpoint 与禁用 Xet，默认保存到 AutoDL 数据盘。
- `scripts/`: 环境检查、模型下载和后续压测工具。
- `tests/`: Pytest 单元测试和正确性回归。

# 进度追踪 (Progress Tracker)

- **当前状态**: Step 0 已完成；Step 1 进行中，评分与预算子步骤已完成。
- **已完成任务**:
  - 项目脚手架与记忆库初始化；在 AutoDL 项目目录初始化 `main` 分支并推送首次提交。
  - 新增 `scripts/check_env.py` 和 `scripts/download_model.py`。
  - 配置 uv 0.13.8 与隔离 Python 3.11.18；安装 Hugging Face Hub 0.36.2、Transformers 4.51.3、PyTorch 2.5.1+cu121、TorchVision 0.20.1+cu121、TorchAudio 2.5.1+cu121、vLLM 0.7.3、FlashAttention-2 2.8.3.post2、Accelerate 1.15.0 和 setuptools 80.10.2。
  - 下载 Qwen2.5-VL-7B-Instruct 至 `/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`；5 个权重分片完整，共 16,584,414,560 字节。
  - `uv pip check` 通过；脚本语法、`--help` 与无模型诊断检查通过。
  - 运行 `python scripts/check_env.py` 通过：检测到 CUDA 12.1 与 RTX 4090；5 个权重分片以 BF16 加载，短文本生成返回 `OK`。
- **当前正在进行的任务**:
  - Step 1：实现序列剪枝、M-RoPE 对齐与完整验收。
- **下一步**:
  - 完成 TokenPruner 与位置对齐，新增 CPU/GPU 测试并验收。
- **已知阻碍/验证记录**:
  - AutoDL 驱动 595.71.05 报告 CUDA 13.2；PyTorch CUDA 12.1 runtime 已验证正常。
  - 首次模型冒烟检查因隔离环境缺少 setuptools 失败；安装 setuptools 80.10.2 后重跑成功。
  - AutoDL 无法连接 `huggingface.co:443`；模型下载通过兼容镜像完成。
  - vLLM 二进制 wheel 解包受容器半核 CPU 限制，安装完成后依赖检查通过。

- **Step 1 子步骤记录**:
  - 评分与预算：FP32 按有效注意力行归一化，以行 Shannon 熵均值驱动默认 25%–75% 保留比例；无证据保留全部。不是谱熵，也尚未接入模型注意力提取。
  - 验证：.venv/bin/python -m pytest tests/test_scoring.py -q，16 passed。

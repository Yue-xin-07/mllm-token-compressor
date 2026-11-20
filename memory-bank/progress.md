# 进度追踪 (Progress Tracker)

- **当前状态**: Step 0、Step 1 已完成；Step 2 尚未开始。
- **已完成任务**:
  - 项目脚手架与记忆库初始化；在 AutoDL 项目目录初始化 `main` 分支并推送首次提交。
  - 新增 `scripts/check_env.py` 和 `scripts/download_model.py`。
  - 配置 uv 0.13.8 与隔离 Python 3.11.18；安装 Hugging Face Hub 0.36.2、Transformers 4.51.3、PyTorch 2.5.1+cu121、TorchVision 0.20.1+cu121、TorchAudio 2.5.1+cu121、vLLM 0.7.3、FlashAttention-2 2.8.3.post2、Accelerate 1.15.0 和 setuptools 80.10.2。
  - 下载 Qwen2.5-VL-7B-Instruct 至 `/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`；5 个权重分片完整，共 16,584,414,560 字节。
  - `uv pip check` 通过；脚本语法、`--help` 与无模型诊断检查通过。
  - 运行 `python scripts/check_env.py` 通过：检测到 CUDA 12.1 与 RTX 4090；5 个权重分片以 BF16 加载，短文本生成返回 `OK`。
- **当前正在进行的任务**:
  - 无。
- **下一步**:
  - 开始 Step 2：在 vLLM Prefill 中接入注意力提取与剪枝，处理 M-RoPE 后续解码、KV Cache 与 Block Table；先验证当前 vLLM 版本可行的接入位置。
- **已知阻碍/验证记录**:
  - AutoDL 驱动 595.71.05 报告 CUDA 13.2；PyTorch CUDA 12.1 runtime 已验证正常。
  - 首次模型冒烟检查因隔离环境缺少 setuptools 失败；安装 setuptools 80.10.2 后重跑成功。
  - AutoDL 无法连接 `huggingface.co:443`；模型下载通过兼容镜像完成。
  - vLLM 二进制 wheel 解包受容器半核 CPU 限制，安装完成后依赖检查通过。

- **Step 1 子步骤记录**:
  - 评分与预算：FP32 按有效注意力行归一化，以行 Shannon 熵均值驱动默认 25%–75% 保留比例；无证据保留全部。不是谱熵，也尚未接入模型注意力提取。
  - 验证：.venv/bin/python -m pytest tests/test_scoring.py -q，16 passed。
  - 剪枝与位置对齐：新增单序列 TokenPruner/PruningResult，稳定排序、保留所有非视觉 Token、原样 gather 三轴位置，输出原序列下一位置起点；无 padding/KV Cache 管理。
  - 验证：.venv/bin/python -m pytest tests/test_scoring.py tests/test_pruner.py -q，38 passed。
  - 完整验收：.venv/bin/python -m pytest tests/ -q -s，48 passed（含 RTX 4090 FP16/BF16 CPU/GPU 一致性、混合设备拒绝、多视觉段、非连续张量与被删最大坐标测试）。
  - 显存回归：10 次预热后运行 5×30 次；每轮释放输出并同步，memory_allocated 均为 673,792 字节，memory_reserved 均为 2,097,152 字节；该用例未观察到活跃张量占用增长，不代表所有负载均无泄漏。git diff --check 通过；所有新增文件少于 300 行。
  - 阶段边界：单条无 padding 序列的可测试 PyTorch 基线；上游提供已掩码 [H,Q,V] 注意力，仅将视觉内容标入掩码。尚未完成真实模型注意力提取、vLLM 集成或最终 TTFT/显存/精度指标评测。

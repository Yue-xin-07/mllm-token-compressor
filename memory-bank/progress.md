# 进度追踪 (Progress Tracker)

- **当前状态**: Step 0、Step 1 已完成；Step 2 进行中：注意力探针已完成，准备请求与引擎适配。
- **已完成任务**:
  - 项目脚手架与记忆库初始化；在 AutoDL 项目目录初始化 `main` 分支并推送首次提交。
  - 新增 `scripts/check_env.py` 和 `scripts/download_model.py`。
  - 配置 uv 0.13.8 与隔离 Python 3.11.18；安装 Hugging Face Hub 0.36.2、Transformers 4.51.3、PyTorch 2.5.1+cu121、TorchVision 0.20.1+cu121、TorchAudio 2.5.1+cu121、vLLM 0.7.3、FlashAttention-2 2.8.3.post2、Accelerate 1.15.0 和 setuptools 80.10.2。
  - 下载 Qwen2.5-VL-7B-Instruct 至 `/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`；5 个权重分片完整，共 16,584,414,560 字节。
  - `uv pip check` 通过；脚本语法、`--help` 与无模型诊断检查通过。
  - 运行 `python scripts/check_env.py` 通过：检测到 CUDA 12.1 与 RTX 4090；5 个权重分片以 BF16 加载，短文本生成返回 `OK`。
  - Step 1 基线已完成：新增注意力评分、熵驱动预算、确定性视觉 Token 剪枝与原始 M-RoPE 坐标对齐；完整单测 48 passed。
  - Step 1 算法及验收已提交：afc6292（评分预算）、98c0980（剪枝与位置对齐）、d57842c（GPU 验收）。
- **当前正在进行的任务**:
  - 无。
- **下一步**:
  - **Step 2：vLLM 推理引擎源码级植入与 PagedAttention 适配（尚未开始）**。
  - 检查 vLLM 0.7.3 的 Qwen2.5-VL 前向、Prefill 和注意力元数据路径，确定注意力提取与剪枝接入位置。
  - 接入 Step 1 基线并同步序列长度、保留索引和原始 M-RoPE 坐标，正确衔接后续解码位置。
  - 适配 KV Cache、slot mapping 与 Block Table 的分配和回收，补充对应正确性测试。
  - 验收：单元测试通过，并在 vLLM 离线推理模式下成功跑通长文本多模态请求。
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

- **Step 1 交付复核**:
  - 逐项检查本阶段 8 个新增/修改文件，补全 architecture.md 的文件职责和接口约定；本次交付整理仅修改记忆文档，沿用已通过的 48 项测试记录。

- **Step 2.1**：新增分块 GQA 视觉注意力探针，应用原序列因果掩码；16 项测试通过（含 GPU FP16/BF16）。尚未接入真实模型。

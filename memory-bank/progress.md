# 进度追踪 (Progress Tracker)

- **当前状态**: Step 0–3 已完成，Step 2 为单卡单请求首版；下一阶段为 Step 4。
- **已完成任务**:
  - 项目脚手架与记忆库初始化；在 AutoDL 项目目录初始化 `main` 分支并推送首次提交。
  - 新增 `scripts/check_env.py` 和 `scripts/download_model.py`。
  - 配置 uv 0.13.8 与隔离 Python 3.11.18；安装 Hugging Face Hub 0.36.2、Transformers 4.51.3、PyTorch 2.5.1+cu121、TorchVision 0.20.1+cu121、TorchAudio 2.5.1+cu121、vLLM 0.7.3、FlashAttention-2 2.8.3.post2、Accelerate 1.15.0 和 setuptools 80.10.2。
  - 下载 Qwen2.5-VL-7B-Instruct 至 `/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`；5 个权重分片完整，共 16,584,414,560 字节。
  - `uv pip check` 通过；脚本语法、`--help` 与无模型诊断检查通过。
  - 运行 `python scripts/check_env.py` 通过：检测到 CUDA 12.1 与 RTX 4090；5 个权重分片以 BF16 加载，短文本生成返回 `OK`。
  - Step 1 基线已完成：新增注意力评分、熵驱动预算、确定性视觉 Token 剪枝与原始 M-RoPE 坐标对齐；完整单测 48 passed。
  - Step 1 算法及验收已提交：afc6292（评分预算）、98c0980（剪枝与位置对齐）、d57842c（GPU 验收）。
  - Step 2 已完成并提交：3123725（注意力探针）、8af46fa（请求准备）、6a63a76（vLLM 接入与缓存生命周期）；常规测试 72 passed，真实模型测试 6 passed。
  - Step 3 已完成：7a84a3f（CUDA 基准）、01fb438（Triton 评分融合与接入）。RTX 4090 四组合成用例的评分耗时下降 69.7%–94.8%，均达到至少 20% 的验收标准。
- **当前正在进行的任务**:
  - 无。
- **下一步**:
  - **Step 4：自动化压测、显存监控与指标产出**。建立 32K、混合并发与原生模型对照，记录 TTFT、峰值显存、吞吐和下游精度，输出 `evaluation_report.md`。
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

- **Step 2.2**：PreparedRequest 与第一层 QK 探针请求准备已实现；3 项测试通过，覆盖坐标/embeddings/索引对齐、无文本证据及全保留。准备结果区分物理长度与原始 next_position。


- **Step 2.3 引擎集成与验收**：
  - 已实现请求压缩后再注册原生调度器，按压缩长度分配 KV blocks；原始 M-RoPE 与 cache offset 独立。
  - 可恢复 runner/model 钩子、decode 偏移、失败 abort/资源回收及 close；无 site-packages 文件改动。
  - 发现 torch 2.5.1 缺少 flash-attn 视觉 rotary 所需 wrap_triton；采用 vLLM 自带 PyTorch rotary 兼容，未更换依赖。
  - 常规测试 72 passed、6 skipped；真实模型显式验收 6 passed，含原生/全保留一致、长输入、纯文本、双图、视频和异常恢复。
  - 长输入 5687→5671，视觉 Token 64→48；40 步生成，块峰值 358→357（16 Token/块），空闲块前后均 3736。
  - docs/step2-integration.md 与 docs/step2-validation.json 记录运行方式、边界和验收；git diff --check 与文件行数检查通过。
  - 未证明 TTFT、32K、全局 KV pool 预留显存缩减或下游准确率达标。


- **Step 2 交付复核（2026-11-23）**：
  - 对照 8a590c6..6a63a76 核查全部 16 个新增/修改文件，覆盖算法、请求准备、runner、兼容层、离线脚本、测试及文档。
  - 修正 architecture.md 开头遗留的 Step 1/Step 2 阶段描述，逐项记录本阶段文件职责；当前为 Step 2 受限首版完成，下一步 Step 3 尚未开始。
  - 本次仅更新 progress.md 与 architecture.md，沿用此前 72 项常规测试和 6 项真实模型验收记录，未重新运行模型或测试。
  - 提交信息采用 Conventional Commits；交付文档提交后，将连同 3 个 Step 2 实现提交推送至 origin/main。

- **Step 3.1 基线测量**：在 RTX 4090 上以 Qwen2.5-VL 形状的 BF16 合成 Q/K 和 FP32 注意力计时。20 次同步测量的中位数显示，5623 文本 query、64 视觉 Token 时探针/评分分别为 13.2482/0.9623 ms；2048 query、1024 视觉 Token 时分别为 5.0503/7.8544 ms。先融合 GPU 评分中的行归一化、熵和列汇总，保留探针和稳定排序。

- **Step 3.2 Triton 评分实现及 RTX 4090 验收（2026-11-23）**：
  - 融合行归一化、熵、有效行计数和视觉列部分归约，接入 Qwen2.5-VL 请求准备。保留稳定排序和 PyTorch 参考。
  - 以 10 次预热、50 次 CUDA 同步墙钟计时的中位数比较两个后端。四组的视觉保留预算与保留索引一致。

  | 文本 query / 视觉 Token | 探针 ms | PyTorch 评分 ms | Triton 评分 ms | 含评分剪枝 ms（PyTorch → Triton） |
  | --- | ---: | ---: | ---: | ---: |
  | 512 / 64 | 1.3691 | 0.3111 | 0.0944 | 0.5754 → 0.3398 |
  | 2048 / 256 | 4.4825 | 1.6939 | 0.1379 | 1.9703 → 0.4430 |
  | 5623 / 64 | 12.4808 | 0.9625 | 0.1083 | 1.3144 → 0.5224 |
  | 2048 / 1024 | 5.2197 | 7.8515 | 0.4079 | 8.1938 → 0.7527 |

  - 真实模型 5687→5671 Token，3 次热请求的中位数为请求准备 80.53 ms、Prefill 步 530.38 ms、后续 7 步 Decode 合计 136.77 ms。该记录用于定位阶段耗时；端到端前后对照归 Step 4。
  - 复现：`.venv/bin/python scripts/benchmark_compression.py --warmup 10 --repeats 50 --model /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`。
  - `MLLM_RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/ -q`：91 passed，含 13 项新增评分/接入测试和 6 项真实模型回归。

- **Step 3 交付复核（2026-11-23）**：
  - 对照 ac93aa0..01fb438 逐项核查 10 个新增或修改文件，补全 architecture.md 文件清单，确认文件行数与 Git diff 格式检查通过。
  - 在 AutoDL 重新运行 `MLLM_RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/ -q`，91 项全部通过，包含 6 项真实模型回归。

# 架构与目录说明 (Architecture & File Manifest)

## 当前阶段与项目规则

- 当前已完成 Step 0–4 的实现与验收流程。Step 2 交付单请求首版，Step 3 完成 Triton 评分，Step 4 增加离线批量对照和评估报告。
- `AGENTS.md`：Memory Bank 优先、单文件 300 行上限、算法 pytest 闭环、子步骤更新文档并提交 Git，以及 AutoDL SSH 说明。
- `.gitignore`：排除缓存、虚拟环境、Notebook 检查点、模型权重、数据集和运行产物。
- `.venv/`：Python 3.11；PyTorch 2.5.1+cu121、Transformers 4.51.3、vLLM 0.7.3、FlashAttention-2 2.8.3.post2 等隔离依赖。

## 记忆库与阶段文档

- `memory-bank/implementation-plan.md`：Step 0–4 的任务和验收标准；Step 0–4 已完成阶段验收，Step 4 记录量化目标的实测结果。
- `memory-bank/progress.md`：当前状态、历史子步骤、测试依据、已知问题与 Step 4 的实测记录。
- `memory-bank/architecture.md`：记录源码、测试、工具及各阶段变更的文件职责。
- `memory-bank/tech-stack.md`：模型、推理引擎、算子工具、硬件约束及最终 TTFT/显存/准确率目标。
- `memory-bank/.ipynb_checkpoints/`：Notebook 历史自动保存副本；当前状态以根目录正式文档为准，不同步改写历史副本。

## Step 0 环境工具

- `scripts/check_env.py`：检查 Python、CUDA/GPU 和推理依赖，支持模型加载及短文本生成冒烟验证。
- `scripts/download_model.py`：下载/续传模型快照，支持镜像 endpoint、禁用 Xet 和数据盘存储路径。

## Step 1 压缩算法基线

- `compressor/__init__.py`：声明算法包，具体接口从子模块导入。
- `compressor/scoring.py`：AttentionBudget、score_attention；对有效 [H,Q,V] 权重行归一化，计算 FP32 重要性、归一化行 Shannon 熵均值和视觉保留数量；无证据时保留全部。
- `compressor/token_pruner.py`：TokenPruner、PruningResult；稳定筛选视觉 Token，保留非视觉 Token，原序 gather 隐藏状态与三轴位置，返回保留索引及原序列 next_position。
- `tests/test_scoring.py`：预算、熵、归一化、空证据/非法输入和数值稳定性测试。
- `tests/test_pruner.py`：确定性、索引/位置对齐、多视觉段、非连续张量、GPU 半精度对照及重复调用活跃显存回归。
- Step 1 历史验收为 48 passed；真实注意力提取与引擎接入由 Step 2 完成。

## Step 2 注意力与请求准备

- `compressor/attention_probe.py`（新增）：旋转后的 Q/K 到 [H,Q,V] 条件视觉注意力；支持 GQA 头映射、按 query 分块、原序列因果掩码和空因果行归零。
- `compressor/prepared.py`（新增并补充预算校验）：PreparedRequest 与 prepare_request；生成多模态 embeddings，取第一层归一化/QKV/M-RoPE 探针，构造文本 query/视觉掩码并复用剪枝；区分物理长度与解码位置偏移。
- `tests/test_attention_probe.py`（新增）：完整注意力参考对照、GQA/分块组合、空证据、非法索引及 CUDA FP16/BF16 一致性。
- `tests/test_prepared.py`（新增并补充非法预算回归）：保留索引、embeddings、三轴位置及 rope_delta 对齐；全保留、无文本证据、纯文本，以及模型访问前拒绝非法预算。

## Step 2 引擎与缓存接入

- `compressor/vllm_engine.py`（新增）：版本限定的 CompressedEngine；原生预处理、原始 M-RoPE、请求准备、压缩序列注册、解码偏移、原生块管理、诊断统计、异常 abort 及 close。
- `compressor/vllm_bridge.py`（新增）：RunnerBridge 上下文钩子；Prefill 注入保留 embeddings/位置，保留原生注意力缓存元数据，采集每步位置/slot/Block Table，退出或异常时恢复钩子。
- `compressor/vllm_compat.py`（新增）：torch 2.5 缺失 wrap_triton 时，使用 vLLM 自带视觉 rotary PyTorch 参考实现，返回恢复函数。
- `scripts/run_compressed.py`（新增）：合成图像和长文本离线入口；支持全保留对照、输入/生成长度选项及 JSON 诊断输出，退出时关闭引擎。
- `tests/test_vllm_bridge.py`（新增）：Prefill 注入、decode 路径、缓存元数据保持及异常钩子恢复。
- `tests/test_vllm_compat.py`（新增）：视觉旋转参考实现数值一致性与兼容函数恢复。
- `tests/test_vllm_integration.py`（新增）：显式启用的真实模型测试；覆盖原生/全保留对照、长输入跨块解码、纯文本、双图、视频、异常回收及恢复。
- `docs/step2-integration.md`（新增）：适用范围、运行命令、接入流程、兼容原因、验收数据及未验证目标。
- `docs/step2-validation.json`（新增）：可追溯验收摘要，包括输入/保留数量、缓存块对照、首末 decode 位置与槽位、测试结果。

## 算法接口、集成边界与验收依据

- 单条无 padding 序列输入：隐藏状态 [S,D]、int64 原始位置 [3,S]、bool 视觉内容掩码 [S]、有效/因果掩码后的跨模态权重 [H,Q,V]；V 列按原视觉顺序，特殊 Token 不属于视觉内容。
- 默认预算为 ceil(V × (0.25 + 0.50 × entropy))；视觉非空时至少保留一个，无证据保留全部；采用行 Shannon 熵均值，同分优先原索引，不是谱熵。
- next_position 为原始三轴位置最大值加一，空序列为零；decode rope_delta = next_position − 压缩长度；物理 cache offset 独立增长。
- Step 2 使用第一层 QKV 预探针，在调度前确定预算，所有层从保留 embeddings 开始 Prefill；以可恢复运行时钩子接入，不改写 site-packages。
- 首版限 vLLM 0.7.3 V0、单 GPU、单活动请求、n=1/best_of=1；不支持 prompt logprobs、多请求及中间层剪枝。
- 关闭 prefix caching、chunked prefill、CUDA Graph、异步输出和 swap；先压缩再注册请求，由原生调度器分配、追加和回收 KV blocks。
- 已有验收：常规测试 72 passed、6 skipped；单独启用真实模型测试 6 passed。5687→5671 Token、视觉 64→48、40 步解码、块峰值 358→357。
- 未验证 32K、最终 TTFT/显存比例/下游精度；全局 KV pool 预留量与请求实际占用块数不同，诊断 trace 和注意力探针仍有额外开销。

## Step 3 评分加速与验证

- `compressor/triton_scoring.py`（新增）：融合 FP32 行归一化、Shannon 熵和视觉列部分归约，避免完整的归一化和熵中间张量。GPU 汇总部分归约，CPU 计算整数保留预算。动态预算接近取整边界时回退到 PyTorch。
- `compressor/scoring.py`（修改）：保留 PyTorch 参考实现，共用预算计算。对 CUDA sm80+ 上连续 FP32、64–2048 视觉 Token 的探针输出选择 Triton，其余输入使用 PyTorch。
- `compressor/prepared.py`（修改）：在真实模型请求准备路径中将探针输出交给评分后端，并传入剪枝流程。
- `compressor/token_pruner.py`（修改）：两个评分后端共用输入校验和剪枝逻辑，按原视觉索引打破同分，并依保留索引提取 embeddings 与 M-RoPE 坐标。
- `tests/test_triton_scoring.py`（新增）：覆盖 CUDA 数值与预算对照、空行、同分、非法权重、边界回退及请求准备路径。
- `scripts/benchmark_compression.py`（新增）：比较一次注意力探针、PyTorch/Triton 评分和含评分剪枝。可通过 `--model` 添加真实模型请求准备、Prefill 与 Decode 的分段耗时。
- `scripts/run_compressed.py`（修改）：提取 `make_prompt`，供离线运行入口和真实模型分段基准复用同一合成图文输入。
- `memory-bank/implementation-plan.md`（修改）：勾选 Step 3，记录评分速度与回归验收结果。
- `memory-bank/progress.md`（修改）：记录 Step 3 基准、验证和交付，并列出 Step 4 工作。
- `memory-bank/architecture.md`（修改）：更新当前阶段，登记 Step 3 全部变更文件及其职责。
- Step 3 验收为 91 passed，其中包含 6 项真实模型回归。RTX 4090 四组合成输入的评分耗时下降 69.7%–94.8%。端到端对照、32K、多请求、显存和精度评估归 Step 4。


## Step 4 自动化压测

- `compressor/vllm_engine.py`：增加原生/压缩批量入口，共用同配置的 vLLM 0.7.3 V0 引擎。批量入口接受累积输出模式，单请求入口默认行为保留。
- `compressor/vllm_bridge.py`：按序列组顺序拼接压缩 Prefill 的 embeddings 与 M-RoPE 位置，保留原生缓存元数据，完成 Prefill 后释放准备数据。仅接入无 chunked prefill 的默认 V0 调度。
- `compressor/vllm_workload.py`：提交整批、记录首输出 Token、在调度阶段采集 KV 峰值、记录实际批大小。发生抢占时中止该轮并恢复调度钩子与缓存。
- `compressor/vllm_compat.py`：对图像限额配置临时修正启动探针中的额外图像占位符，原生和压缩使用该同一兼容路径。
- `scripts/benchmark_e2e.py` 与 `scripts/benchmark_workloads.py`：构造精确的预处理长度，独立进程运行双后端，写逐轮结果及错误，分别采样 NVML 设备与进程显存、PyTorch 分配和保留峰值。
- `tests/test_vllm_workload.py` 及桥接、兼容与真实模型回归：覆盖首 Token 计时、KV 采样时点、批内对齐、启动探针、全保留输出与异常恢复。
- `scripts/evaluate_screenspot.py` 与 `scripts/screenspot_metrics.py`：加载 SeeClick 原版三组标注与图像，使用统一的归一化坐标提示和确定性生成，按点击是否落入标注框判分。批次可从同配置的输出恢复。
- `scripts/report_evaluation.py`：校验双端配置，汇总逐轮性能与逐样本精度，计算目标达成状态并生成 Markdown 报告。
- `tests/test_screenspot_metrics.py`：验证坐标变换、边界、越界及无效输出判分。
- 32K 对照给两端配置相同的 2,114 个 16 Token KV 块，PyTorch 分配器启用可扩展 CUDA 段。
- `evaluation_report.md`：正式评估报告，含目标判定、双端性能表、ScreenSpot 分组准确率和复现命令。原始 JSON 位于 AutoDL 的 `outputs/step4/`。

## GUI Agent 工作集

- `compressor/working_set.py`：结构化文本和截图、确定性排序、依赖闭包与预算回退。
- `compressor/context_submission.py`：估算视觉/文本成本，在计时区间执行筛选，验证真实预处理预算，再用原生 Qwen M-RoPE 和缓存分配。
- `scripts/benchmark_e2e.py`：保留原始 32K 和混合用例，额外测量工作集策略。
- `scripts/gui_protocol.py` 和 `scripts/evaluate_gui.py`：共享工具坐标前缀，以模型缩放后的像素空间解码、映射至原图，写出可续跑的 ScreenSpot 逐样本结果。
- `scripts/gui_replay.py` 和 `scripts/evaluate_gui_replay.py`：真实截图加模拟任务记录，按截图分离开发集与留出集，六种策略共用证据与点击判分。

# 架构与目录说明 (Architecture & File Manifest)

## 项目规则与环境

- `AGENTS.md`：开发规则、Memory Bank 优先读取、单文件 300 行上限、算法单测、子步骤文档更新与提交要求，以及 AutoDL SSH 连接说明。
- `.gitignore`：排除缓存、虚拟环境、Notebook 检查点、模型权重、数据集及运行产物。
- `.venv/`：隔离 Python 3.11 环境；PyTorch 2.5.1+cu121、Transformers 4.51.3、vLLM 0.7.3、FlashAttention-2 2.8.3.post2 等依赖。

## 记忆库与阶段状态文件

- `memory-bank/implementation-plan.md`：Step 0–4 的任务与验收标准；本阶段勾选 Step 1，记录 48 项测试通过及基线边界。
- `memory-bank/progress.md`：当前进展、完成事项、测试和问题记录；本阶段记录 Step 1 完成与下一步 Step 2 的引擎集成任务。
- `memory-bank/architecture.md`：文件功能清单、算法接口及后续集成约定；覆盖本阶段全部新增和修改文件。
- `memory-bank/tech-stack.md`：目标模型、技术栈、硬件约束，以及最终延迟、显存和准确率目标。
- `memory-bank/.ipynb_checkpoints/`：Notebook 自动保存的历史副本；阶段状态以目录根部正式文档为准。

## Step 0 环境工具

- `scripts/check_env.py`：检查 Python、CUDA/GPU 和推理依赖，支持 Qwen2.5-VL 短文本生成冒烟验证。
- `scripts/download_model.py`：下载或续传 Hugging Face 模型快照，支持自定义 endpoint 与禁用 Xet，默认保存至 AutoDL 数据盘。

## Step 1 算法源码（新增）

- `compressor/__init__.py`：声明压缩算法 Python 包；具体接口从子模块导入。
- `compressor/scoring.py`：定义 AttentionBudget 和 score_attention；接收浮点 [H,Q,V] 非负有限权重，按有效行归一化，输出 FP32 重要性、归一化行 Shannon 熵均值与动态保留数量；零行不参与统计，无有效证据时保留全部。
- `compressor/token_pruner.py`：定义 TokenPruner 和 PruningResult；校验输入后稳定筛选视觉 Token、保留全部非视觉 Token，按原序列顺序 gather 隐藏状态与原始三轴 M-RoPE 坐标，返回保留索引、实际视觉预算、熵及原序列下一位置起点。

## Step 1 单元测试（新增）

- `tests/test_scoring.py`：验证熵与预算关系、逐行归一化、零行忽略、空输入/无证据回退、非法权重/比例，以及大数值稳定性和推理模式。
- `tests/test_pruner.py`：验证确定性排序、文本与索引/三轴坐标对齐、输入不变、空序列与非法输入、多视觉段及非连续张量、删除最大坐标后的生成起点；包含 RTX 4090 FP16/BF16 CPU/GPU 对照、混合设备拒绝和重复调用活跃显存回归。
- 完整验收命令：`.venv/bin/python -m pytest tests/ -q -s`，48 passed；每个新增 Python 文件均少于 300 行。

## 算法接口与 Step 2 集成约定

- 单条无 padding 序列输入：隐藏状态 [S,D]、int64 原始位置 [3,S]、bool 视觉内容掩码 [S]、已应用有效位置/因果掩码的跨模态注意力 [H,Q,V]。V 列按原视觉 Token 顺序排列，特殊 Token 不得标入视觉掩码。
- 默认视觉预算为 ceil(V × (0.25 + 0.50 × entropy))；有视觉 Token 时至少保留一个，无有效注意力则保留全部。此基线采用行 Shannon 熵均值，不是谱熵。
- 推理模式计算，不保留请求张量；评分采用 FP32，同分时原索引较小者优先，输出保留原序列顺序。
- next_position 为原始三轴位置最大值加一，空序列返回零；它不是物理 cache_position 或 Transformers rope_delta，不能按压缩长度重新生成视觉网格坐标。
- Step 2 负责真实模型注意力提取、vLLM Prefill 接入、批次与序列元数据、后续解码位置、KV Cache 及 Block Table 适配。本阶段未验证长上下文端到端性能或下游任务精度。

## Step 2 集成

- `compressor/attention_probe.py`：旋转后的 Q/K 到 [H,Q,V] 条件视觉注意力，分块计算、GQA 头映射、空因果行归零。
- `tests/test_attention_probe.py`：完整注意力参考对照、空证据、非法索引及 GPU 精度。

- `compressor/prepared.py`：PreparedRequest、真实模型第一层 QKV 探针、视觉与文本 query 掩码、复用 Step 1 剪枝。
- `tests/test_prepared.py`：准备结果对齐、rope_delta、全保留与纯文本。

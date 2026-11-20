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

## Step 1 算法基线

- compressor/scoring.py：AttentionBudget 与 score_attention，接收 [H,Q,V] 非负注意力权重，输出 FP32 重要性、归一化行熵与视觉预算；忽略全零行。
- compressor/__init__.py：压缩算法包入口。
- tests/test_scoring.py：评分、预算、无证据回退及非法输入测试。
- compressor/token_pruner.py：TokenPruner 接收 [S,D] 隐藏状态、int64 [3,S] 原始位置、bool [S] 视觉内容掩码、[H,Q,V] 注意力；返回 PruningResult（压缩张量、保留索引、视觉预算、熵、原始 next_position）。特殊 Token 不得标为视觉内容；next_position 不是 cache_position/rope_delta。
- tests/test_pruner.py：确定性选择、文本/坐标/索引对齐、空输入、回退与参数验证。

## Step 1 验收与后续集成约定

- 默认视觉预算：ceil(V × (0.25 + 0.50 × H))，H 为有效注意力行归一化 Shannon 熵均值；至少保留一个视觉 Token。无有效证据保留全部，零视觉输入直通。不是谱熵。
- 推理模式、不存储请求张量、FP32 评分、稳定降序排序；输出按原始顺序 gather。单序列接口，不负责批量 padding、注意力提取或 KV Cache。
- tests/test_pruner.py：额外包含 CUDA FP16/BF16 对照、混合设备、活跃显存回归、多视觉段和删除最大坐标测试；完整 tests/ 共 48 项通过。
- 未来 vLLM 适配须区分原始三轴位置、next_position、物理 cache_position 和 rope_delta；不得用压缩长度重新生成视觉网格坐标。

# 工业级实现计划 (Implementation Plan)

- [x] **Step 0: 环境准备、依赖安装与 Qwen2.5-VL 模型权重下载**
  - **任务**: 检查 RTX 4090 算力环境，安装 FlashAttention-2 与 vLLM，编写下载脚本将 `Qwen2.5-VL-7B-Instruct` 权重拉取至本地。
  - **验证标准**: 运行 `python scripts/check_env.py`，成功加载模型并完成一次前向推理不报错。**已通过**：模型在 RTX 4090 上以 BF16 加载并生成 `OK`。

- [x] **Step 1: 跨模态 Token 动态剪枝与预算分配核心算法**
  - **任务**: 在 `compressor/token_pruner.py` 实现基于跨模态注意力得分及矩阵熵的动态裁剪模块，并完美对齐 M-RoPE 空间位置编码。
  - **验证标准**: 运行 `pytest tests/test_pruner.py`，确保张量 Shape 变换正确且无内存泄漏。**基线已通过**：.venv/bin/python -m pytest tests/ -q -s，48 passed；含 RTX 4090 FP16/BF16 一致性及重复调用活跃显存稳定性测试。矩阵熵采用有效注意力行的归一化 Shannon 熵均值；真实模型注意力提取与引擎集成归 Step 2。

- [x] **Step 2: vLLM 推理引擎源码级植入与 PagedAttention 适配**
  - **任务**: 将剪枝模块嵌入 vLLM 的 Prefill 阶段前向管线，处理好稀疏 Token 带来的 Block Table 显存碎片回收。
  - **验证标准**: 在 vLLM 离线推理模式下成功跑通长文本多模态请求。**受限首版已通过**：单卡单请求 V0、5687 Token 输入和 40 步解码；72 项常规测试、6 项真实模型测试。图像/双图/视频、全保留对照、缓存回收及异常恢复通过；详见 docs/step2-integration.md。未验收 32K 或最终性能/精度指标。

- [x] **Step 3: Triton 自定义算子加速核心瓶颈**
  - **任务**: 针对 Token 评分筛选编写 `@triton.jit` 融合算子，减少 GPU HBM 访存。
  - **验证标准**: 算子单元测试通过，且计算耗时相比 PyTorch 原生实现提速 $\ge 20\%$。 **已通过**：RTX 4090 上四个合成用例的评分耗时下降 69.7%–94.8%，剪枝保留索引与 PyTorch 一致。启用真实模型后的全套测试 91 passed。

- [ ] **Step 4: 自动化压测、显存监控与指标产出**
  - **任务**: 编写混合并发压测脚本，输出对比表格（TTFT、峰值显存、吞吐量）。
  - **验证标准**: 生成包含最终量化指标的 `evaluation_report.md`。

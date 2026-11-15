# 技术栈与硬核约束 (Tech Stack)

- **目标项目**：MLLM-Token-Compressor (面向长上下文 Agent 场景的多模态 Token 动态压缩与推理加速引擎)
- **量化业务目标（面试硬指标）**：
  - **首字延迟 (TTFT)**：在 32K 超长上下文输入下，相比原生 Qwen2.5-VL-7B 降低 $\ge 35\%$。
  - **显存优化**：峰值 KV Cache 显存占用压缩 $\ge 40\%$，支持单张 RTX 4090 (24GB) 稳定运行长序列多模态交互。
  - **精度无损**：在核心下游任务（如 ScreenSpot GUI 评测或 Video-MME）上的准确率跌幅控制在 $\le 1\%$ 以内。
- **核心技术栈**：
  - Base Model: `Qwen2.5-VL-7B-Instruct` (HuggingFace Transformers / vLLM)
  - 推理引擎: `vLLM` (定制修改 Prefill 阶段调度与 PagedAttention 块内存管理)
  - 加速算子: `@triton.jit` (用于替代原生 PyTorch 的高频注意力打分与 Top-K 算子)
  - 基建环境: PyTorch 2.4+, CUDA 12.1, FlashAttention-2, RTX 4090 (AutoDL 云端)

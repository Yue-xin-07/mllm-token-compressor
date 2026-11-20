# Step 2 单请求离线集成

适用环境：vLLM 0.7.3 V0、Qwen2.5-VL-7B-Instruct、BF16、单 RTX 4090。
仅支持一个活动请求、n=1、best_of=1；不支持 prompt logprobs。
关闭 prefix caching、chunked prefill、CUDA Graph、异步输出、交换及推测解码。
原始输入长度也必须不超过 max_model_len（默认 8192）。

## 使用

在项目根目录运行：

```bash
.venv/bin/python scripts/run_compressed.py
.venv/bin/python scripts/run_compressed.py --keep-all --output /tmp/keep-all.json
.venv/bin/python -m pytest tests/ -q
MLLM_RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/test_vllm_integration.py -q -s
```

默认演示为 224×224 合成红色图像与 800 次上下文重复；生成 40 个 Token。
为验证跨块解码而启用 ignore_eos，可能在自然回答结束后继续生成无关文本。
输出 JSON 包含 Token 数、缓存块峰值、前后空闲块、每步坐标/槽位/Block Table。
输出中的 prompt token IDs 和用量对应压缩后实际执行的序列；原始长度在统计中单独记录。

## 接入方式

1. 原生输入预处理生成多模态参数，原模型生成视觉 embeddings。
2. 根据未剪枝的视觉网格生成原始 M-RoPE；第一层 input norm/QKV/旋转构造探针。
3. 非视觉、非特殊文本位置为 query；视觉内容为 key。按 GQA 分组、因果掩码、
   FP32 和 query 分块计算视觉条件注意力，复用 Step 1 的熵预算和确定性剪枝。
   不构造完整 S×S 注意力矩阵，但保留 H×Q×V 输出，视觉 Token 很多时仍有开销。
4. 先压缩再注册调度请求。调度器按压缩长度原生分配、追加和回收所有层的 KV blocks，
   无需先分配完整请求再搬运稀疏 KV。预先保留的 embeddings 直接送入模型。
5. Prefill 使用 gather 后的原始三轴坐标；decode 的 rope_delta =
   原始 next_position − 压缩长度。逻辑 cache offset 和 M-RoPE 坐标保持独立。
6. 请求完成或失败时 abort、恢复 runner/model 钩子并释放准备结果；
   CompressedEngine.close() 释放引擎并恢复视觉旋转兼容函数。

适配代码全部保存在仓库：通过版本检查和可恢复运行时钩子接入已安装 vLLM，
不需要手工编辑 site-packages；重新实例化适配器即可重复应用。
程序直接持有单进程 driver worker，不能用于分布式服务或多实例共享进程。

当前 FlashAttention 2.8.3 的视觉旋转 Triton 包装需要 torch.library.wrap_triton，
而 PyTorch 2.5.1 不提供它。vllm_compat 在缺失此 API 时使用 vLLM 自带
apply_rotary_emb_torch 参考实现。兼容函数的数值和恢复行为有独立测试。

## 验收记录

- 常规测试：72 passed、6 skipped；6 项真实模型测试通过环境变量单独启用。
- 真实模型测试：6 passed；覆盖全保留与原生 Token 输出一致、长文本跨块解码、
  纯文本对照、双图、视频、注入前向异常后的块回收与后续请求恢复。
- 长输入：5687 → 5671 Token；视觉内容 64 → 48；连续生成 40 Token。
- 同一长输入全保留/压缩的缓存块峰值：358 → 357，块大小 16。
- 长输入请求前后空闲 GPU blocks：3736 → 3736。
- decode 每步校验原始 next_position 递增、逻辑序列长度、物理 slot 与 Block Table 一致。
- 详细摘要见同目录 step2-validation.json；完整运行日志及 trace 位于 /tmp，
  可能随机器清理而消失。

这是 Step 2 的正确性基线。5687 Token 验收不代表 32K 验收；仅减少视觉 Token，
文本为主的样例压缩收益有限。第一层注意力探针的额外计算、CPU 同步和诊断 trace
开销尚未优化，未测最终 TTFT、峰值 KV 显存比例或下游精度，不能宣称已达业务指标。
vLLM 全局 KV pool 的预留大小仍由初始化配置决定；本阶段验证的是请求实际占用块。

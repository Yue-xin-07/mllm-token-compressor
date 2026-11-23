# Step 4 evaluation report

| Project target | Outcome |
| --- | --- |
| 32K TTFT p50 reduction ≥35% | Missed (-1.90%) |
| 32K peak occupied KV reduction ≥40% | Missed (+0.20%) |
| ScreenSpot accuracy delta ≥−1 pp | Met (-0.63 pp) |

## Performance

TTFT uses burst submission to first returned token. Throughput divides total output tokens by total measured wall time. GPU device and process peaks are sampled by NVML every 10 ms. PyTorch peaks are process-local allocator metrics. KV occupancy comes from the scheduler after block allocation and before generated results can free blocks.

NVIDIA GeForce RTX 4090, PyTorch 2.5.1+cu121, vLLM 0.7.3 V0, bfloat16, GPU memory utilization 0.95. Each backend ran in a fresh process with 3 warm-up rounds and 20 measured rounds per workload. Image cap 200,704 pixels. Each response was fixed to 32 tokens.
The two backends share a 2114-block KV pool and allocator configuration `expandable_segments:True`.
The synthetic long inputs use repeated text and one 448×448 image. The mixed workload submits a 2K text request and an 8K image request in the same scheduling round.

### single_32k

Measured rounds, native 20, compressed 20. Failed rounds, native 0, compressed 0.

| Metric | Native | Compressed | Change |
| --- | ---: | ---: | ---: |
| TTFT p50 (ms) | 4198.974 | 4278.682 | +1.90% |
| TTFT p95 (ms) | 4224.539 | 4332.488 | +2.56% |
| Output tokens/s | 6.591 | 6.479 | -1.70% |
| Peak device memory (GiB) | 23.285 | 23.405 | +0.51% |
| Peak process memory (GiB) | 22.807 | 22.926 | +0.52% |
| Peak PyTorch reserved (GiB) | 22.350 | 22.469 | +0.53% |
| Peak PyTorch allocated (GiB) | 21.736 | 21.719 | -0.08% |
| KV pool (GiB) | 1.806 | 1.806 | +0.00% |
| Peak occupied KV (GiB) | 1.752 | 1.748 | -0.20% |

Input tokens by request [32768], retained [32704], visual [256]. Peak KV blocks 2,050 → 2,046. The 25% minimum visual retention caps input token removal at 192 of 32,768 (0.59%) before generated tokens and cache block rounding.

### mixed_2k_8k

Measured rounds, native 20, compressed 20. Failed rounds, native 0, compressed 0.

| Metric | Native | Compressed | Change |
| --- | ---: | ---: | ---: |
| TTFT p50 (ms) | 1006.208 | 1029.991 | +2.36% |
| TTFT p95 (ms) | 1015.039 | 1034.745 | +1.94% |
| Output tokens/s | 39.208 | 38.832 | -0.96% |
| Peak device memory (GiB) | 23.285 | 23.405 | +0.51% |
| Peak process memory (GiB) | 22.807 | 22.926 | +0.52% |
| Peak PyTorch reserved (GiB) | 22.350 | 22.469 | +0.53% |
| Peak PyTorch allocated (GiB) | 18.749 | 18.800 | +0.27% |
| KV pool (GiB) | 1.806 | 1.806 | +0.00% |
| Peak occupied KV (GiB) | 0.550 | 0.547 | -0.62% |

Input tokens by request [2048, 8192], retained [2048, 8128], visual [0, 256]. Peak KV blocks 644 → 640. The 25% minimum visual retention caps input token removal at 192 of 10,240 (1.88%) before generated tokens and cache block rounding.

## ScreenSpot click accuracy

Original [SeeClick test set](https://github.com/njucckevin/SeeClick#gui-grounding-benchmark-screenspot). A click on or inside the target box is correct. Missing or invalid points count as errors. Both backends use the same normalized-coordinate prompt and deterministic decoding.

Native 1,272/1,272, compressed 1,272/1,272, paired 1,272.

- `screenspot_desktop.json` SHA-256 `d101500a238f2f7d7f00684279c0de4d8e046ea5fc410b609d86c8d15a600a0d`
- `screenspot_mobile.json` SHA-256 `7ea52a90feb79c039560232e24ba0209824d21ec7adc3755c8a629343f54c0cd`
- `screenspot_web.json` SHA-256 `0df1f5a55a2255f32e26314a8fa299a38449ccbac7c04457e7810923eba97352`
- `screenspot_imgs.tar` SHA-256 `76aeb06a56d86da2bd404bda31b3d240608da08064cdb2c945f03b562d0d9c69`

| Group | Count | Native | Compressed | Delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| desktop / icon | 140 | 12.86% | 12.86% | +0.00 |
| desktop / text | 194 | 13.40% | 12.37% | -1.03 |
| mobile / icon | 229 | 3.06% | 3.49% | +0.44 |
| mobile / text | 273 | 6.59% | 4.40% | -2.20 |
| web / text | 230 | 3.48% | 3.48% | +0.00 |
| web / icon | 206 | 2.43% | 1.94% | -0.49 |
| Overall | 1,272 | 6.45% | 5.82% | -0.63 |

Invalid outputs on paired samples, native 435, compressed 442. The prompt used an image cap of 802,816 pixels and a 24-token budget. Generation stops at the first closing parenthesis, which can cut off verbose answers before the coordinates.

The measured delta applies to this prompt and decoding protocol. The low absolute scores and invalid-output counts limit broader grounding conclusions. Qwen's [computer-use cookbook](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb) formats grounding answers as tool calls.

## Reproduction

Run from the repository root with the existing AutoDL environment. Download the original [ScreenSpot release](https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/) into `data/screenspot`.

```bash
mkdir -p data/screenspot
base=https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/files/
for file in screenspot_desktop.json screenspot_mobile.json screenspot_web.json screenspot_imgs.tar; do
  curl -fL --retry 2 "$base?p=%2F$file&dl=1" -o "data/screenspot/$file"
done
tar -xf data/screenspot/screenspot_imgs.tar -C data/screenspot

for mode in native compressed; do
  .venv/bin/python scripts/benchmark_e2e.py --mode "$mode" --output "outputs/step4/$mode.json"
  .venv/bin/python scripts/evaluate_screenspot.py --mode "$mode" --dataset data/screenspot \
    --output "outputs/step4/screenspot-$mode.json"
done
.venv/bin/python scripts/report_evaluation.py
```

Raw round and per-sample results are in `outputs/step4` on AutoDL.

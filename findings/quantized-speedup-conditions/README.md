# A8 speed-ups depend on chain depth and weights; the E4B QAT MLP gets none

The [positive control](../../docs/MEASUREMENTS.md#128-layer-positive-controls) shows Core AI W8A8 running 1.86–1.87× faster than FP16 on a 128-layer synthetic chain. The same measurements on shorter chains, other weights and a released QAT checkpoint show how much of that belongs to the format and how much to the fixture.

Unless stated otherwise, each synthetic layer is a 512 → 512 1×1 convolution over 4,096 positions. Times are awaited `function.run` calls, so they are API latencies, not continuous throughput.

## Depth

The sixteen-code weights described under [Weights](#weights), 73.63% exact zeros, with unequal per-output-channel scales. The A8W4 graph has a QDQ at both ends and between every layer; two processes per depth.

| Layers | W4A16 (ms) | A8W4 (ms) | A8W4 speed vs W4A16 |
|---:|---:|---:|---:|
| 1 | 0.35 | 0.40 | 0.86× |
| 2 | 0.45 | 0.48 | 0.92× |
| 8 | 0.83 | 0.74 | 1.13× |
| 32 | 2.25 | 1.78 | 1.27× |
| 128 | 7.80 | 5.87 | 1.33× |

A straight line through the five points costs 0.058 ms per layer for W4A16 and 0.043 ms for A8W4. The A8W4 line starts 0.049 ms higher. The two lines cross near three layers; three to seven layers were not measured. So a single gate or a short stack is the wrong place to look for an A8 gain, and a short-graph slowdown does not rule one out.

At 128 layers, moving the boundary QDQs does not remove the gain: interlayer QDQs alone give 1.34×, adding an input QDQ 1.34×, adding an output QDQ 1.29×, both 1.33×.

## Scale granularity

Same 128-layer weights: one scalar scale gives 1.34×, equal per-channel scales 1.36×, unequal per-channel scales 1.31×. Per-channel scales do not by themselves block the speed-up.

## Weights

The same 128-layer shapes with two weight sets. The dense set is the positive control's ±1 Hadamard-derived weights; the sixteen-code set has 73.63% exact zeros.

| Weights | FP16 (ms) | A8W4 speed vs FP16 | A8W4 speed vs W4A16 |
|---|---:|---:|---:|
| Dense ±1 | 14.29 | 1.89× | 1.88× |
| Sixteen codes, 73.63% zeros | 7.67 | 1.36× | 1.35× |

A separate FP16-only round isolates the zeros. Six weight sets, same shapes, three processes each:

| FP16 weights | Exact zeros | Median of process p50 (ms) | Sparse set runs faster by |
|---|---:|---:|---:|
| Dense ±1 | 0 | 14.77 | 1.88× |
| Dense ±1, each row shuffled | 0 | 14.75 | 1.88× |
| Sixteen codes | 73.63% | 7.85 | — |
| Sixteen codes, hidden channels permuted | 73.63% | 7.84 | 1.00× |
| Sixteen codes, zeros set to ±2⁻¹⁴ | 0 | 14.74 | 1.88× |
| Sixteen codes, zeros set to ±scale/16 | 0 | 14.74 | 1.88× |

Replacing the zeros with the smallest normal FP16 magnitude leaves every nonzero weight unchanged and raises each row norm by 0.7 ppm, yet restores the dense time. Shuffling the dense rows and permuting the sparse channels change nothing. The FP16 path is faster with exact zero weights, not with this particular value pattern.

For benchmarks this has two consequences. Synthetic weights with many exact zeros (zero initialisation, pruning, a quantized codebook that includes 0) can run an FP16 baseline about 1.9× faster than dense weights of the same shape. And a quantized speed-up measured against such a baseline is smaller: 1.35× instead of 1.88× here. The positive control uses dense weights, so its FP16 baseline is not the fast case.

## A released QAT checkpoint

[Gemma 4 E4B mobile QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) (revision `3624117c`): its first MLP, with original INT4 codes, per-channel weight scales and static A8 activation scales, repeated eight times behind a shared FP16 RMS norm, at 64 positions. Three processes.

| Graph | Median pipeline (ms) | Speed vs W4A16 | Relative L2, positions 6–63 |
|---|---:|---:|---:|
| W4A16 | 4.24 | — | 0.38% |
| A8W4 as exported | 4.34 | 0.977× | 509% |
| A8W4 with a product clamp | 4.34 | 0.976× | 21.6% |
| A8W4 split around FP32 corrections, 17 calls | 26.07 | 0.163× | 22.1% |

The A8 graphs are wrong because of [the QDQ multiply defect](../coreai-qdq-multiply-scale/), and neither the exported nor the clamped graph is faster. Call size alone does not explain the missing gain. One call does 80.5 G source-equivalent ops, as much as 37.5 synthetic layers, and the synthetic chain was already 1.27× faster at 32 layers. The two graphs also differ in weights per layer (78.6 million against 262,144), in GELU and the gated multiply, and in shape. None of those has been isolated.

The eight layers are one layer repeated, and the inputs are synthetic control rows, so the L2 column describes implementation error, not model quality.

## Reproduce

```sh
python scripts/verify_g1w.py
```

This recomputes every time, ratio and fitted line above from the bundled per-call timings, and checks the times and ratios against the summaries recorded when each round closed.

## What this does not show

No evidence here shows a physical INT8 datapath: an A8 graph can be faster for other reasons. Whether the zero effect comes from weight compression, skipped work or kernel selection is not distinguished, and no sparsity threshold was scanned. Depths of three to seven layers, real (non-repeated) E4B layers and full-model quality were not measured. coreai-torch 0.4.1, macOS 27.0 (26A428).

## Evidence

- [g1w-e4b-mobile-qat/timings.json.gz](../../results/historical/g1w-e4b-mobile-qat/) — 56,768 measured calls
- [evidence.json](../../results/historical/g1w-e4b-mobile-qat/evidence.json) — fixtures, summaries recorded at close, E4B relative L2
- [PROVENANCE.md](../../docs/PROVENANCE.md#g1-w-and-g5-imports) — sources and transformations

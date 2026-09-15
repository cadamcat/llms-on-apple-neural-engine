# A8 speed-ups depend on work per call and on the weights; the E4B QAT MLP gets none

The [positive control](../../docs/MEASUREMENTS.md#128-layer-positive-controls) shows Core AI W8A8 running 1.86–1.87× faster than FP16 on a 128-layer synthetic chain. The same measurements on shorter chains, other weights and a released QAT checkpoint show how much of that belongs to the format and how much to the fixture.

Unless stated otherwise, each synthetic layer is a 512 → 512 1×1 convolution over 4,096 positions. Times are awaited `function.run` calls, so they are API latencies, not continuous throughput.

## Depth

The sixteen-code weights described under [Weights](#weights), <!-- claim:g1w.weights.zero-fraction@g1w-001 -->73.63%<!-- /claim --> exact zeros, with unequal per-output-channel scales. The A8W4 graph has a QDQ at both ends and between every layer; two processes per depth.

| Layers | W4A16 (ms) | A8W4 (ms) | A8W4 speed vs W4A16 |
|---:|---:|---:|---:|
| 1 | <!-- claim:g1w.depth.bare-1.ms@g1w-002 -->0.35<!-- /claim --> | <!-- claim:g1w.depth.both-1.ms@g1w-003 -->0.40<!-- /claim --> | <!-- claim:g1w.depth.both-1.speed@g1w-004 -->0.86×<!-- /claim --> |
| 2 | <!-- claim:g1w.depth.bare-2.ms@g1w-005 -->0.45<!-- /claim --> | <!-- claim:g1w.depth.both-2.ms@g1w-006 -->0.48<!-- /claim --> | <!-- claim:g1w.depth.both-2.speed@g1w-007 -->0.92×<!-- /claim --> |
| 8 | <!-- claim:g1w.depth.bare-8.ms@g1w-008 -->0.83<!-- /claim --> | <!-- claim:g1w.depth.both-8.ms@g1w-009 -->0.74<!-- /claim --> | <!-- claim:g1w.depth.both-8.speed@g1w-010 -->1.13×<!-- /claim --> |
| 32 | <!-- claim:g1w.depth.bare-32.ms@g1w-011 -->2.25<!-- /claim --> | <!-- claim:g1w.depth.both-32.ms@g1w-012 -->1.78<!-- /claim --> | <!-- claim:g1w.depth.both-32.speed@g1w-013 -->1.27×<!-- /claim --> |
| 128 | <!-- claim:g1w.depth.bare-128.ms@g1w-014 -->7.80<!-- /claim --> | <!-- claim:g1w.depth.both-128.ms@g1w-015 -->5.87<!-- /claim --> | <!-- claim:g1w.depth.both-128.speed@g1w-016 -->1.33×<!-- /claim --> |

A straight line through the five points costs <!-- claim:g1w.fit.w4a16.slope@g1w-017 -->0.058<!-- /claim --> ms per layer for W4A16 and <!-- claim:g1w.fit.a8w4.slope@g1w-018 -->0.043<!-- /claim --> ms for A8W4. The A8W4 line starts <!-- claim:g1w.fit.intercept-gap@g1w-019 -->0.049<!-- /claim --> ms higher. The two lines cross near three layers; three to seven layers were not measured. On this 512-channel chain a single layer or a short stack shows no A8 gain, and a short-graph slowdown does not rule one out.

**Corrected 2026-09-15.** This paragraph said a single gate or a short stack is the wrong place to look for an A8 gain, and the title named chain depth. A single real E4B gate at 1024 positions [does gain](#positions-per-call-on-a-real-projection); depth is one way to add work to a call, and positions per call are another.

At 128 layers, moving the boundary QDQs does not remove the gain: interlayer QDQs alone give <!-- claim:g1w.depth.inner-128.speed@g1w-020 -->1.34×<!-- /claim -->, adding an input QDQ <!-- claim:g1w.depth.input-128.speed@g1w-021 -->1.34×<!-- /claim -->, adding an output QDQ <!-- claim:g1w.depth.output-128.speed@g1w-022 -->1.29×<!-- /claim -->, both <!-- claim:g1w.depth.both-128.speed@g1w-023 -->1.33×<!-- /claim -->.

## Positions per call on a real projection

G7 ran the first E4B gate projection, 2560 → 10240 with the checkpoint's four-bit codes and per-channel scales, through Core AI on the Neural Engine. Inputs are seeded RMS-normalized activations; each configuration ran in three new hosts for 120 s. Rates are three-round medians in positions per second; ratios are the three paired rounds.

| Positions per call | FP16 | W4A16 | W8A8 | A8W4 | A8W4 speed vs W4A16 | A8W4 energy vs W4A16 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | <!-- claim:g7.rate.gate.64.coreai.fp16@g7-054 -->138,305<!-- /claim --> | <!-- claim:g7.rate.gate.64.coreai.w4a16@g7-055 -->216,303<!-- /claim --> | <!-- claim:g7.rate.gate.64.coreai.w8a8@g7-056 -->213,932<!-- /claim --> | <!-- claim:g7.rate.gate.64.coreai.a8w4@g7-057 -->212,866<!-- /claim --> | <!-- claim:g7.coreai.gate.64.a8w4-over-w4a16.speed@g7-058 -->0.976–0.990×<!-- /claim --> | <!-- claim:g7.coreai.gate.64.a8w4-over-w4a16.energy@g7-059 -->0.768–0.778×<!-- /claim --> |
| 1024 | <!-- claim:g7.rate.gate.1024.coreai.fp16@g7-060 -->258,787<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.w4a16@g7-061 -->379,220<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.w8a8@g7-062 -->468,405<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.a8w4@g7-063 -->515,465<!-- /claim --> | <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.speed@g7-064 -->1.359–1.360×<!-- /claim --> | <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.energy@g7-065 -->0.642–0.647×<!-- /claim --> |

At 64 positions the four-bit weights already bring W4A16 level with W8A8 (<!-- claim:g7.coreai.gate.64.w8a8-over-w4a16.speed@g7-066 -->0.987–0.990×<!-- /claim --> W4A16's speed), and adding A8 changes nothing. At 1024 positions A8W4 is <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.speed@g7-067 -->1.359–1.360×<!-- /claim --> faster than W4A16 and <!-- claim:g7.coreai.gate.1024.a8w4-over-fp16.speed@g7-068 -->1.991–1.993×<!-- /claim --> faster than FP16, with every block's numeric control and ANE requests checked. The eight repeated MLPs [below](#a-released-qat-checkpoint) do more work per call at 64 positions than this gate does at 1024 and gained nothing, so total work per call does not predict the result; positions per call changed it for this projection. Core ML's four-bit form of the same gate is [slower than its own FP16](../coreml-coreai-same-codes/), so its A8W4 ratio is not comparable.

## Scale granularity

Same 128-layer weights: one scalar scale gives <!-- claim:g1w.scale.scalar.speed@g1w-024 -->1.34×<!-- /claim -->, equal per-channel scales <!-- claim:g1w.scale.equal.speed@g1w-025 -->1.36×<!-- /claim -->, unequal per-channel scales <!-- claim:g1w.scale.varied.speed@g1w-026 -->1.31×<!-- /claim -->. Per-channel scales do not by themselves block the speed-up.

## Weights

The same 128-layer shapes with two weight sets. The dense set is the positive control's ±1 Hadamard-derived weights; the sixteen-code set has <!-- claim:g1w.weights.zero-fraction@g1w-027 -->73.63%<!-- /claim --> exact zeros.

| Weights | FP16 (ms) | A8W4 speed vs FP16 | A8W4 speed vs W4A16 |
|---|---:|---:|---:|
| Dense ±1 | <!-- claim:g1w.codebook.old.fp16-ms@g1w-028 -->14.29<!-- /claim --> | <!-- claim:g1w.codebook.old.a8-over-fp16@g1w-029 -->1.89×<!-- /claim --> | <!-- claim:g1w.codebook.old.a8-over-w4a16@g1w-030 -->1.88×<!-- /claim --> |
| Sixteen codes, <!-- claim:g1w.weights.zero-fraction@g1w-031 -->73.63%<!-- /claim --> zeros | <!-- claim:g1w.codebook.full.fp16-ms@g1w-032 -->7.67<!-- /claim --> | <!-- claim:g1w.codebook.full.a8-over-fp16@g1w-033 -->1.36×<!-- /claim --> | <!-- claim:g1w.codebook.full.a8-over-w4a16@g1w-034 -->1.35×<!-- /claim --> |

A separate FP16-only round isolates the zeros. Six weight sets, same shapes, three processes each:

| FP16 weights | Exact zeros | Median of process p50 (ms) | Sparse set runs faster by |
|---|---:|---:|---:|
| Dense ±1 | 0 | <!-- claim:g1w.density.old.ms@g1w-035 -->14.77<!-- /claim --> | <!-- claim:g1w.density.old.speed@g1w-036 -->1.88×<!-- /claim --> |
| Dense ±1, each row shuffled | 0 | <!-- claim:g1w.density.shuffled.ms@g1w-037 -->14.75<!-- /claim --> | <!-- claim:g1w.density.shuffled.speed@g1w-038 -->1.88×<!-- /claim --> |
| Sixteen codes | <!-- claim:g1w.weights.zero-fraction@g1w-039 -->73.63%<!-- /claim --> | <!-- claim:g1w.density.sparse.ms@g1w-040 -->7.85<!-- /claim --> | — |
| Sixteen codes, hidden channels permuted | <!-- claim:g1w.weights.zero-fraction@g1w-041 -->73.63%<!-- /claim --> | <!-- claim:g1w.density.permuted.ms@g1w-042 -->7.84<!-- /claim --> | <!-- claim:g1w.density.permuted.speed@g1w-043 -->1.00×<!-- /claim --> |
| Sixteen codes, zeros set to ±2⁻¹⁴ | 0 | <!-- claim:g1w.density.fill-min.ms@g1w-044 -->14.74<!-- /claim --> | <!-- claim:g1w.density.fill-min.speed@g1w-045 -->1.88×<!-- /claim --> |
| Sixteen codes, zeros set to ±scale/16 | 0 | <!-- claim:g1w.density.fill-16.ms@g1w-046 -->14.74<!-- /claim --> | <!-- claim:g1w.density.fill-16.speed@g1w-047 -->1.88×<!-- /claim --> |

Replacing the zeros with the smallest normal FP16 magnitude leaves every nonzero weight unchanged and raises each row norm by <!-- claim:g1w.weights.norm-ppm@g1w-048 -->0.7<!-- /claim --> ppm, yet restores the dense time. Shuffling the dense rows and permuting the sparse channels change nothing. The FP16 path is faster with exact zero weights, not with this particular value pattern.

For benchmarks this has two consequences. Synthetic weights with many exact zeros (zero initialisation, pruning, a quantized codebook that includes 0) can run an FP16 baseline about 1.9× faster than dense weights of the same shape. And a quantized speed-up measured against such a baseline is smaller: <!-- claim:g1w.codebook.full.a8-over-w4a16@g1w-049 -->1.35×<!-- /claim --> instead of <!-- claim:g1w.codebook.old.a8-over-w4a16@g1w-050 -->1.88×<!-- /claim --> here. The positive control uses dense weights, so its FP16 baseline is not the fast case.

## A released QAT checkpoint

[Gemma 4 E4B mobile QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) (revision `3624117c`): its first MLP, with original INT4 codes, per-channel weight scales and static A8 activation scales, repeated eight times behind a shared FP16 RMS norm, at 64 positions. Three processes.

| Graph | Median pipeline (ms) | Speed vs W4A16 | Relative L2, positions 6–63 |
|---|---:|---:|---:|
| W4A16 | <!-- claim:g1w.e4b.bare.ms@g1w-051 -->4.24<!-- /claim --> | — | <!-- claim:g1w.e4b.8.bare.l2@g1w-052 -->0.38%<!-- /claim --> |
| A8W4 as exported | <!-- claim:g1w.e4b.native.ms@g1w-053 -->4.34<!-- /claim --> | <!-- claim:g1w.e4b.native.speed@g1w-054 -->0.977×<!-- /claim --> | <!-- claim:g1w.e4b.8.native.l2@g1w-055 -->509%<!-- /claim --> |
| A8W4 with a product clamp | <!-- claim:g1w.e4b.clip-product.ms@g1w-056 -->4.34<!-- /claim --> | <!-- claim:g1w.e4b.clip-product.speed@g1w-057 -->0.976×<!-- /claim --> | <!-- claim:g1w.e4b.8.clip-product.l2@g1w-058 -->21.6%<!-- /claim --> |
| A8W4 split around FP32 corrections, <!-- claim:g1w.e4b.coarse-calls@g1w-059 -->17<!-- /claim --> calls | <!-- claim:g1w.e4b.split-coarse-a8.ms@g1w-060 -->26.07<!-- /claim --> | <!-- claim:g1w.e4b.split-coarse-a8.speed@g1w-061 -->0.163×<!-- /claim --> | <!-- claim:g1w.e4b.8.split-coarse-a8.l2@g1w-062 -->22.1%<!-- /claim --> |

The exported A8 graph exhibits [the QDQ multiply defect](../coreai-qdq-multiply-scale/). The product clamp greatly reduces the error, but substantial residuals remain, including after splitting around FP32 corrections. Those residuals have not been fully attributed. Neither the exported nor the clamped graph is faster.

Call size alone does not explain the missing gain here, although positions per call do change it for [one projection](#positions-per-call-on-a-real-projection). One call does <!-- claim:g1w.work.stack-gops@g1w-063 -->80.5<!-- /claim --> G source-equivalent ops, as much as <!-- claim:g1w.work.equivalent-layers@g1w-064 -->37.5<!-- /claim --> synthetic layers, and the synthetic chain was already <!-- claim:g1w.depth.both-32.speed@g1w-065 -->1.27×<!-- /claim --> faster at 32 layers. The two graphs also differ in weights per layer (<!-- claim:g1w.work.e4b-million-weights@g1w-066 -->78.6<!-- /claim --> million against <!-- claim:g1w.work.synthetic-weights@g1w-067 -->262,144<!-- /claim -->), in GELU and the gated multiply, and in shape. None of those has been isolated.

The eight layers are one layer repeated, and the inputs are synthetic control rows, so the L2 column describes implementation error, not model quality.

## Reproduce

```sh
python scripts/verify_g1w.py
python scripts/verify_g7.py
```

The first recomputes every G1-W time, ratio and fitted line above from the bundled per-call timings, and checks them against the summaries recorded when each round closed. The second recomputes the G7 projection rates and component energy from block windows, call counts and power frames.

## What this does not show

No evidence here shows a physical INT8 datapath: an A8 graph can be faster for other reasons. Whether the zero effect comes from weight compression, skipped work or kernel selection is not distinguished, and no sparsity threshold was scanned. Depths of three to seven layers, positions between 64 and 1024, a correct and timed A8 graph of a real E4B MLP, and full-model quality were not measured. coreai-torch 0.4.1, macOS 27.0 (26A428).

## Evidence

- [g1w-e4b-mobile-qat/timings.json.gz](../../results/historical/g1w-e4b-mobile-qat/) — <!-- claim:g1w.measured-calls@g1w-068 -->56,768<!-- /claim --> measured calls
- [evidence.json](../../results/historical/g1w-e4b-mobile-qat/evidence.json) — fixtures, summaries recorded at close, E4B relative L2
- [g7-coreml-coreai](../../results/historical/g7-coreml-coreai/) — the G7 projection blocks, admissions and power frames
- [PROVENANCE.md](../../docs/PROVENANCE.md#g1-w-and-g5-imports) — sources and transformations

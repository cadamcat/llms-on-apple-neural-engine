# Group quantization on ANE: representation, compatibility and SplitConv cost

English | [中文](zh/02-分组量化如何改变ANE的执行图.md)

2026-09-10 · [Series index](README.md)

Group scales determine decoded weight values, but also constrain how a graph can be expressed, whether a compiler accepts it and what execution costs. A small compatibility probe and a separate performance ablation with the same weights expose different parts of this relationship.

## 1. Storing the weights is only the first check

LLM group quantization is often described as a storage format: several input channels share a scale, and weight codes multiplied by that scale participate in matrix multiplication. On Apple's inference stack, asset correctness, compiler support for the scale granularity and ANE participation need separate checks. Successful export and correct CPU output do not establish ANE execution.

This article uses two synthetic workloads. The performance experiment is a 512-channel chain with seeded Hadamard weights and Q8-grid inputs tiled across 4096 positions. The compatibility experiment uses 64 output channels, small K64 grouped weights and an identity-matrix input. They answer different questions, and neither evaluates LLM quality. [Methods and evidence scope](../docs/METHODS.md).

## 2. What K32 group scales mean

Consider one output channel with 64 signed INT4 input weight codes. The K64 grouped-weight probe divides them into two K32 groups, each with its own scale, `s₀` or `s₁`:

$$
W = \operatorname{concat}(s_0q_0, s_1q_1),\qquad
(Wx)_o = \sum_{g\in\{0,1\}} s_g\sum_{k\in g}q_{o,k}x_k .
$$

These are weight scales per input group and output channel. They are neither activation scales nor one weight scale shared by the whole output channel, often called a per-cout scale.

That distinction enables splitting. Express K64 as two K32 convolutions, and each sub-convolution needs only one weight scale per output channel:

$$
W x = (s_0q_0)x_0 + (s_1q_1)x_1.
$$

In exact real arithmetic this is equivalent to the original grouped weights. Applying the same decomposition to K512 gives 16 K32 convolutions and 15 additions per layer, or 32 convolutions and 30 additions over two layers.

![Two-layer wide K512 convolutions compared with sixteen K32 partial convolutions per layer and balanced FP16 reduction.](../docs/figures/split-structure.svg)

Figure 1. Each path executes a complete graph in one host prediction. The split graph contains 32 convolutions, 30 additions and one inter-layer QDQ across two layers. The smaller K64 compatibility probe is shown separately below the throughput graph.

These counts describe operations inside the exported graph, not host calls. Both layers and the intermediate QDQ are exported as one model function, and one prediction executes the whole graph.

Graph equivalence does not imply equivalent floating-point execution. The split reference has FP16 boundaries at the partial sums, balanced reduction tree and inter-layer QDQ, unlike the wide-K path. Each therefore uses its own frozen reference. [Reference definitions](../docs/METHODS.md).

## 3. What the wide-K positive control establishes

In the 128-layer wide K512 chain, Core AI W8A8 reaches 34.95–35.06 T ops/s and A8W4 reaches 35.07–35.09 T ops/s. The quantized paths run at about 1.86–1.87× FP16 speed in paired rounds. The [fresh throughput records](../results/fresh/throughput.json) show that timed configurations passed numerical and ANE controls.

This establishes a working low-bit ANE path for the fixed graph, chip and toolchain. Low-entropy ±1 weights and readily expressible scales do not generalize to arbitrary Q4_0 weights, LLM throughput or energy efficiency. Core AI provides no per-operation execution trace, and `physical_INT8_proven=false`.

## 4. The cost of SplitConv

The Core AI A8W4 wide K512 and K32 SplitConv versions of the synthetic bridge graph both pass numerical and ANE controls. Each configuration runs in three independent processes, with four controls, 10 warmups and 30 measurements per process. The three rounds run wide/split, split/wide, then wide/split. The wide path runs at 4.10×, 4.05× and 3.96× the split path's speed in the paired rounds, a range of 3.96–4.10×. The [measurements](../docs/MEASUREMENTS.md) give p50 ranges of 0.432–0.446 ms for the wide path and 1.764–1.772 ms for the split path.

This measures a graph-structure change; it does not establish a universal fourfold cost for group quantization. K32 decomposition adds graph convolutions, partial sums and an addition tree. It also changes fusion opportunities, scheduling, narrow-K kernels and FP16 reduction boundaries. The experiment does not isolate those contributions. It establishes that the complete split graph is substantially slower on this host. The paths use separate frozen numerical references, so identical weights do not require bitwise-identical floating-point outputs.

A historical measurement has the wide path at 4.44× split speed, with one process per configuration. That [historical pair](../results/historical/selected-evidence.json) belongs to a different run and cannot be pooled with the three fresh pairs. The difference between runs cannot be attributed directly to a software change.

![Three independent-process pairs: wide K512 takes about 0.43–0.45 ms and K32 split takes about 1.76–1.77 ms.](../docs/figures/split-latency.svg)

Figure 2. Lines connect the wide and split configurations within each round; they are not a time series. The axis starts at zero. Dividing split p50 by wide p50 gives the wide path's relative speed. The historical 4.44× pair is excluded from the three fresh pairs. [Point values](../docs/figures/manifest.json).

## 5. Native K64 separates two compatibility failures

The small grouped test uses `[64,64,1,1]` signed INT4 weights, two K32 scales per output row and an identity-matrix input. The persisted Core ML direct-INT4 native K64 asset decodes correctly and produces correct output, but its compiler plan selects CPU. The split version is also numerically correct and selects CPU. More specifically, the native K64 convolution lists only CPU support. Both split convolutions list CPU and ANE support but prefer `MLCPUComputeDevice`, with no successful ANE requests in any of the four control windows. ANE support therefore does not substitute for ANE preference or observed ANE requests.

The Core AI native K64 lookup-table path has ANE participation but produces incorrect output matching the flattened-scale prediction, with relative L2 approximately 0.450247 against the intended reference. Its K32 split control matches the exact reference and has ANE participation. This supports a scale-indexing hypothesis without proving the internal implementation. The frameworks use different weight representations, so these observations cannot be reduced to a shared backend bug. [Flattened-scale finding](../findings/coreai-flattened-scale/).

A historical K64 fragment with 15360 output channels and Gemma-derived weights triggered a Core ML diagnostic that ANE supports only per-cout/per-tensor scales. The current 64-output test is a different experiment: output shape, weights and scale distributions all differ. [Historical rejection record](../results/historical/coreml-native-k64-rejection.json). The older Core ML K32 split had ANE participation, whereas the current small K32 split selects CPU. This motivates further experiments but does not isolate shape as the cause.

## 6. Observations and mechanism hypotheses

The hypotheses below are explanations to test. They are separate from the observed results.

| Observation | Established evidence | Mechanism hypothesis | Unresolved question |
|---|---|---|---|
| Core ML native K64 is numerically correct but CPU-preferred | Correct assets and outputs; no ANE admission | The compiler restricts scales grouped along K | Which combination of granularity, QDQ and layout triggers the restriction? |
| Core AI native K64 matches the flattened-scale prediction | ANE participation and incorrect output occur together | The native LUT path handles K-group scale indices differently from the reference | How does internal IR map to a device kernel? |
| The small Core AI K32 split is correct and has ANE participation | The split reference, controls and device events pass | Per-cout scales in sub-convolutions are easier for the runtime to accept | What do addition, scheduling and kernel selection each cost? |
| The wide path runs at about four times split speed | Synchronous prediction timings from three independent-process pairs | Narrow-K computation, reduction or fusion/scheduling costs increase | What are the individual operation costs, and which can be fused? |

Static decoding, device participation, physical integer instructions and model quality remain separate evidence layers. A failed numerical or device check excludes that representation from the benchmark comparison.

## 7. Reproduction and next experiments

In the [locked environment](../docs/REPRODUCING.md), use new output directories:

```sh
.venv/bin/ane-scope run --suite split --output runs/article-split
.venv/bin/ane-scope verify runs/article-split
.venv/bin/ane-scope run --suite compatibility --output runs/article-groups
.venv/bin/ane-scope verify runs/article-groups
```

The resource guard runs these serially. Candidates that fail numerical or device admission remain recorded and skip timing. To check existing results without ANE, run `.venv/bin/python scripts/summarize.py`; it recomputes the tables from [per-call split measurements](../results/fresh/split.json).

Further experiments should change input A8 QDQ, K partitioning or reduction independently while holding the data fixed. When changing scale granularity, first check whether decoded weight values stay the same. If requantization is needed, report its quality effects separately. Real Q4_0 weights with broader distributions, dense activations, other shapes, chips and software versions need independent validation.

The later [G2 service study](04-w4a16-service-tradeoffs.md) measures W4A16 component speed, thermal response and GPU coexistence as a separate experiment.

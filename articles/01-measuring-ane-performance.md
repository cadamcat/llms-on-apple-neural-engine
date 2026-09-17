# Measuring quantized ANE performance: a 35 T ops/s positive control

English | [中文](zh/01-如何验证ANE的量化加速.md)

2026-09-10 · [Series index](README.md)

## Abstract

A synthetic 128-layer convolution chain reaches approximately 35 T source-equivalent operations per second on M5 Pro. To establish what this measures, we check persisted graph representations, numerical references, compiler plans, preferred devices and ANE participation separately before timing. The results support acceleration on this quantized graph path. Per-operation execution placement, physical instructions and full-model benefits remain unproven.

## 1. Define the work before interpreting the speed

The workload is a 128-layer convolution chain with a fixed shape of `[1,512,64,64]`. Each layer is a 512×512 pointwise convolution over 4096 spatial positions. We define its source-graph work as

`2 × depth × 512² × 4096`

The factor of 2 counts a multiply and an add per MAC; it does not count device instructions. At depth 128, this gives 274,877,906,944 source-equivalent operations, or approximately 0.2749 T operations. At depth 2, it gives 4,294,967,296. Reported T ops/s divides this work by the p50 synchronous prediction time in seconds and then by 10¹². For example, 274,877,906,944 operations divided by approximately 7.84 ms gives about 35.06 T ops/s. The numerator counts source convolutions only, while the prediction time also includes graph operations such as QDQ, slicing and reduction. This measures how fast the synthetic source graph completes; it cannot be converted directly to model token/s.

Weights come from Hadamard matrices, layer-specific permutations and sign flips, with the same initial values reconstructed from a seed. The weight codes use only ±1 with a shared scale, so they cover neither the full four-bit code range nor real group-scale distributions. Inputs contain only 16 independent spatial vectors, repeated across 4096 positions. This supports stable reference calculations and byte-level repeatability checks, but does not represent high-entropy LLM activations. [Methods and quantization boundaries](../docs/METHODS.md).

## 2. Compare each representation with its numerical reference

FP16, W8A8 and A8W4 start from the same initial weights in this synthetic workload. W8A8 and A8W4 insert activation quantize–dequantize operations, or QDQ, between layers, so the three graphs are not quality-equivalent models. The FP16 reference accumulates in FP32 and converts to binary16 at the boundaries. The quantized reference uses the declared scales, binary16 intermediate boundaries and Q8 round-to-nearest with ties away from zero (RZA). A separate ties-to-even (RNE) reference diagnoses rounding differences; it does not replace the admission reference.

Similar W8A8 and A8W4 speeds therefore establish similar throughput under these synthetic controls. They do not establish equal speed or quality for arbitrary real weights. The split experiment also changes K-axis slicing, the FP16 addition tree and rounding boundaries, so the wide and split32 paths each need their own frozen reference.

![Frozen workload and persisted-asset checks lead to numerical, resource and device admission, then timing and evidence review.](../docs/figures/evidence-pipeline.svg)

Figure 1. The suite builds evidence in stages. Numerical failures, CPU selection and insufficient evidence remain observations and are excluded from the performance comparison.

## 3. Four controls make correctness testable

Each independent process first runs the original input, an all-zero input, a sign-negated input and a repeat of the original. The linear convolution chains and grouped probes should negate their outputs when the input is negated; the multiplication probes should leave their outputs unchanged. Every output is saved and hashed. The repeated original must match byte for byte, and every zero-control output value must be zero. All controls require finite outputs. Nonzero positive and negative controls assessed by relative L2 also require at least 25% nonzero output values. The relative-L2 limits are 0.01 for depth 2 and 0.05 for depth 128; the exact-grid grouped and multiplication probes require zero numerical difference.

These controls detect failures such as zero outputs, sign errors and unstable repetition, without establishing correctness over all inputs. For example, Core AI `group-native64` has ANE participation in the smoke results but fails its numerical controls, excluding it from timing. Both Core ML grouped assets pass numerical checks but select CPU.

## 4. Audit the persisted assets independently

The Python object before conversion is not the final artifact. Core ML assets must be reloaded from the saved protobuf and weight blobs. The audit follows input dependencies through conv, QDQ, slice and add operations, identifies which weights and intermediate values each convolution reads, and independently decodes weights, scales, zero points and INT4 data. Core AI assets are read from saved bytecode and checked for INT8 weights or four-bit palette indices. Unknown representations, including optimizer transformations the audit cannot identify, fail the audit.

This separates a request for low-bit operations from a saved model containing an identifiable low-bit representation. It also exposes two frontend forms of grouped weights: Core ML uses direct signed INT4 with blockwise scaling; Core AI uses four-bit indices, an INT8 lookup table and scales. Matching probe semantics does not imply matching serialization or device kernels. The audit records relative paths and SHA-256 hashes for all asset files and checks them again later.

## 5. Compiler plans, preferred devices and ANE participation are different kinds of evidence

The Core ML host saves model descriptions and compute plans, including supported and preferred devices for operations. A separate capture of the target process's unified log must contain successful ANE requests within every control window. The Core AI host requests Neural Engine specialization and records the loaded function identity and control-window participation, but the current interface provides no per-operation placement trace.

A serialized `conv` or `constexpr` describes the graph. A preferred device describes the compiler plan. An ANE request shows participation during a control window. None establishes exclusive ANE execution for every operation or physical INT8 instructions. Requesting `.cpuAndNeuralEngine` or four-bit weights does not establish either claim. Reports therefore separate serialized representation, numerical checks, plan, preferred device, ANE participation and physical INT8 evidence.

## 6. What the 35 T ops/s positive control establishes

The fresh throughput experiment uses Apple M5 Pro, macOS 27.0 build 26A428, Xcode 27.0, coremltools 9.0 and coreai-torch 0.4.1 / coreai-core 1.0.0b2. Each configuration runs in three independent processes. After the four controls, each process performs 10 warmups and 30 synchronous measurements. Configuration order is forward, reverse, then forward across the three rounds. [Per-call records](../results/fresh/throughput.json).

| Configuration | p50 range (ms) | Source-equivalent throughput (T ops/s) |
|---|---:|---:|
| Core ML FP16 128 | 14.768–15.115 | 18.19–18.61 |
| Core ML W8A8 128 | 7.934–8.052 | 34.14–34.65 |
| Core AI FP16 128 | 14.631–14.662 | 18.75–18.79 |
| Core AI W8A8 128 | 7.839–7.865 | 34.95–35.06 |
| Core AI A8W4 128 | 7.833–7.837 | 35.07–35.09 |

The timing window covers the Swift synchronous prediction or function call, including runtime, waiting and synchronization costs within that call. Output copying, validation, hashing and file writes happen outside the window. Unified-log capture stops after the controls, before warmup and measurement. This is not isolated kernel execution time.

Approximately 35 T ops/s is the Core AI W8A8/A8W4 rate calculated from each process's p50 and the source-graph work definition. The highest A8W4 process reaches 35.09 T ops/s. Against FP16 at approximately 18.8 T ops/s, the quantized chain runs at about 1.86–1.87× the speed in paired rounds. This supplies a positive control for the tested path, without establishing a general ANE peak or full-model performance. The measured throughput and speed-up ratio depend on the test weights, inputs and graph structure, not just the quantization format. At the same shapes, weights with <!-- claim:g1w.weights.zero-fraction@g1w-001 -->73.63%<!-- /claim --> exact zeros make FP16 itself <!-- claim:g1w.density.old.speed@g1w-002 -->1.88×<!-- /claim --> faster and cut the A8W4 gain over FP16 from <!-- claim:g1w.codebook.old.a8-over-fp16@g1w-003 -->1.89×<!-- /claim --> to <!-- claim:g1w.codebook.full.a8-over-fp16@g1w-004 -->1.36×<!-- /claim -->; the gain also shrinks with depth, to <!-- claim:g1w.depth.both-1.speed@g1w-005 -->0.86×<!-- /claim --> of W4A16 speed for a single layer. [Speed-up conditions](../findings/quantized-speedup-conditions/). The 30 samples within a process are correlated observations; the three-process range is not a confidence interval.

![Throughput for five full-shape configurations: bars show the median rate across three independent processes, with paired speed ratios on the right.](../docs/figures/throughput-by-process.svg)

Figure 2. Bar lengths are medians of the three process rates; labels give their full ranges, not confidence intervals. Rates divide source operations by each process's p50. Ratios on the right compare each quantized path's speed with FP16 in the same round. [Figure sources and exact fields](../docs/figures/manifest.json).

## 7. Reproduce or recompute

On a supported Apple host, use the locked environment:

```sh
uv sync --locked --extra apple
.venv/bin/ane-scope run --suite throughput --output runs/my-throughput
.venv/bin/ane-scope verify runs/my-throughput
.venv/bin/ane-scope report runs/my-throughput
```

The output directory must be new. Export, compilation, host-execution or damaged-evidence failures are recorded as execution failures. Candidates that fail numerical or device admission remain observations and skip timing. Check `manifest.json`, each process's `admission.json` and its measurement count: a successful exit code does not mean every configuration accelerated successfully. Before running, check the machine, OS, SDK, dependencies and source identity. Saved source snapshots, data and asset hashes identify the run.

Without a device, read the result JSON with the Python standard library and recompute T ops/s as `source_ops / (p50_ms × 10^9)`, including the process ranges and paired ratios. That checks the arithmetic of the report; it supplies no new device-participation or physical-INT8 evidence. Installation requirements, offline caching and resource limits are in the [reproduction guide](../docs/REPRODUCING.md). The runner does not download models automatically. These commands check the bundled tables without executing device code:

```sh
.venv/bin/python scripts/summarize.py
.venv/bin/python results/historical/tests/verify_historical.py
```

The positive control provides a comparison point for the [next article](02-group-quantization-and-split.md), which examines how preserving quantization granularity changes executable graphs and split costs. The later [G2 service study](04-w4a16-service-tradeoffs.md) measures W4A16 component speed, thermal response and GPU coexistence as a separate experiment.

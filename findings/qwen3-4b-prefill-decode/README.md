# Qwen3-4B: prefill, decode and energy

A complete Qwen3-4B FP16 model, the same initial token IDs and fixed KV capacity, measured through Core AI on ANE and GPU on one M5 Pro. The GPU is faster throughout the measured range. ANE uses less component energy for short prefill. At long contexts it selects a larger fixed graph and is slower and more energy-intensive per token.

![Warmed prefill and decode speed.](../../docs/figures/g3-speed.svg)

At <!-- claim:g3.n.1024@g3-001 -->1K<!-- /claim --> input, ANE prefill is <!-- claim:g3.prefill.1024.ane-rate@g3-002 -->832.4 token/s<!-- /claim -->, against the GPU's <!-- claim:g3.prefill.1024.gpu-rate@g3-004 -->3,235.9 token/s<!-- /claim -->. From the same starting KV length, decode is <!-- claim:g3.decode.1024.ane-rate@g3-005 -->13.9 token/s<!-- /claim --> on ANE and <!-- claim:g3.decode.1024.gpu-rate@g3-006 -->30.0 token/s<!-- /claim --> on GPU. Main-plot rates use the same warmed work blocks as the energy calculation.

![Projection FLOP/s and modelled weight/KV reads, below the ANE graph change.](../../docs/figures/g3-implied.svg)

Those rates can be expressed as effective model work. Prefill uses <!-- claim:g3.projection-parameters@g3-012 -->3.63 billion<!-- /claim --> projection weights per token, so ANE sustains <!-- claim:g3.prefill.1024.ane-tflops@g3-013 -->6.0 TFLOP/s<!-- /claim --> at <!-- claim:g3.n.1024@g3-014 -->1K<!-- /claim --> against the GPU's <!-- claim:g3.prefill.1024.gpu-tflops@g3-015 -->23.5 TFLOP/s<!-- /claim -->, and holds <!-- claim:g3.short-prefill-ane-tflops@g3-016 -->5.9–6.4 TFLOP/s<!-- /claim --> across <!-- claim:g3.short-contexts@g3-017 -->500–2K<!-- /claim -->. Assuming one read of <!-- claim:g3.weight-bytes@g3-018 -->8.04 GB<!-- /claim --> of FP16 weights and the existing KV per decode step, summed over KV growth, the model gives <!-- claim:g3.decode.1024.ane-bandwidth@g3-019 -->115 GB/s<!-- /claim --> against <!-- claim:g3.decode.1024.gpu-bandwidth@g3-020 -->249 GB/s<!-- /claim -->. The GPU figure holds near that level to <!-- claim:g3.n.8192@g3-021 -->8K<!-- /claim -->, which is consistent with bandwidth-dominated execution but does not identify the bottleneck. ANE’s lower rate could reflect its own memory-access limits, fixed-graph work or host overhead; these have not been separated. Both are useful-work rates from the measured token rates and the model structure: the byte model excludes current-token writes and fixed-graph padding, and no device counter or DRAM traffic was measured.

![Component energy per token, with timing bounds.](../../docs/figures/g3-energy.svg)

ANE prefill energy is <!-- claim:g3.short-prefill-energy-x@g3-007 -->0.70–0.75×<!-- /claim --> the GPU's at <!-- claim:g3.short-contexts@g3-008 -->500–2K<!-- /claim -->, and <!-- claim:g3.long-prefill-energy-ratio@g3-009 -->2.08–2.43×<!-- /claim --> as much at <!-- claim:g3.long-contexts@g3-010 -->4K–16K<!-- /claim -->. The CPU + GPU + ANE sum includes host activity and idle consumption during the work block. Model loading, warmup and recovery are outside it. Software estimator accuracy is uncalibrated; timing-attribution bounds and unmatched thermal starts are described in [SCOPE](../../docs/SCOPE.md#g3-complete-model-observations).

## What changes with context

The ANE path has compiled contexts <!-- claim:g3.graph-contexts@g3-011 -->256 / 2K / 32K<!-- /claim -->. Prefill can complete within the smaller graph while the next decode step crosses its capacity. That is why the decode curve changes before the prefill curve. If each query position evaluates all keys in the <!-- claim:g3.largest-graph@g3-022 -->32K<!-- /claim --> graph, attention work is <!-- claim:g3.attention-flops-largest-graph@g3-023 -->19.3 GFLOP/token<!-- /claim --> against <!-- claim:g3.projection-flops@g3-024 -->7.27 GFLOP/token<!-- /claim --> for the projections. This is a conditional count for one graph; complete requests use several capacities, and masked work may be skipped. The measured ANE prefill drop from <!-- claim:g3.n.2048@g3-025 -->2K<!-- /claim --> to <!-- claim:g3.n.4096@g3-026 -->4K<!-- /claim --> is <!-- claim:g3.prefill-drop@g3-027 -->9.6×<!-- /claim -->; this count does not determine how much of the drop comes from graph capacity. Raw graph events identify the functions used for each request. A new experiment changing the shape ladder would be needed to isolate how much of the gap it causes.

## Recompute

```sh
python scripts/verify_g3.py
python scripts/summarize.py
python scripts/render_figures.py --check
```

These commands read the bundled raw request and selected power fields. They do not load a model or execute a device. The [bundle](../../results/historical/g3-qwen3-4b/) includes run identities, input IDs, per-token clocks, KV and graph events, power receipt anchors and source-field extracts. Original logits and compiled assets remain outside the portable package; their recorded controls are not a portable quality replay.

[Exact measurements](../../docs/MEASUREMENTS.md#g3-complete-qwen3-4b-fp16) · [Full article](../../articles/05-qwen3-4b-prefill-decode-energy.md) · [Methods](../../docs/METHODS.md#g3-complete-model-stages).

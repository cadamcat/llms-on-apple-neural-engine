# Qwen3-4B on ANE: prefill, decode and energy

English | [中文](zh/05-Qwen3-4B的Prefill、Decode与能耗.md)

The earlier W4A16 experiment measured one MLP component. G3 moves the comparison to complete Qwen3-4B FP16 inference: attention, KV updates, layer transitions and token sampling are inside the path. Both arms use Core AI through the same Swift host, with a static-shape engine for ANE and a sequential engine for GPU. The GPU baseline here is different from G2's MLX host.

The question is how quickly each path completes the work and how much energy it takes. Each engine runs at its own speed. Matching a low request arrival rate would answer a different question about background service.

## 1. The measured work

The inputs contain <!-- claim:g3.contexts@g3-001 -->500 / 1,024 / 2,048 / 4,096 / 8,192 / 16,384<!-- /claim --> tokens, drawn from a fixed snapshot of this repository's research documents. Both paths use the same initial token IDs, source weights, FP16 precision and <!-- claim:g3.capacity@g3-002 -->32,768<!-- /claim -->-position KV capacity. This is one input family on one M5 Pro.

G3 first completed one request per context and path, with prefill followed by <!-- claim:g3.coverage-steps@g3-003 -->256<!-- /claim --> decode forwards. A second run measured warmed prefill-only and decode-only blocks. The main figures use this second set: completed input or output tokens divided by the whole block's elapsed time. Request dispatch, reset and recording gaps inside the block remain included. Model loading, the separate warmup and recovery are excluded.

Prefill resets the cache and includes sampling the first token. Decode starts from the prepared cache and performs <!-- claim:g3.decode-steps@g3-004 -->1,024<!-- /claim --> sequential forwards. It continues through EOS to keep the work count fixed. Continuations are freely generated, so equal forward counts do not mean identical output text or equal task quality. The recorded controls compare first-output logits against an independent CPU FP16 reference and check short answers and cache state; they are not a distributional quality evaluation.

## 2. Prefill and decode have different transitions

![Warmed prefill and decode throughput.](../docs/figures/g3-speed.svg)

At <!-- claim:g3.n.1024@g3-005 -->1K<!-- /claim -->, ANE prefill runs at <!-- claim:g3.prefill.1024.ane-rate@g3-006 -->832.4 token/s<!-- /claim --> and GPU at <!-- claim:g3.prefill.1024.gpu-rate@g3-007 -->3,235.9 token/s<!-- /claim -->. The GPU is <!-- claim:g3.prefill.1024.gpu-faster@g3-008 -->3.89×<!-- /claim --> faster. At <!-- claim:g3.n.16384@g3-009 -->16K<!-- /claim -->, the rates are <!-- claim:g3.prefill.16384.ane-rate@g3-010 -->50.9 token/s<!-- /claim --> and <!-- claim:g3.prefill.16384.gpu-rate@g3-011 -->2,312.7 token/s<!-- /claim -->, respectively. The change is much larger than a gradual loss of throughput as the prompt grows.

The ANE graph ladder contains contexts <!-- claim:g3.graph-contexts@g3-012 -->256 / 2K / 32K<!-- /claim -->. A prompt that fits in the smaller context can still cross its boundary on the next decode step. The raw events show the larger extend graph being used from a <!-- claim:g3.n.2048@g3-013 -->2K<!-- /claim --> starting KV, while <!-- claim:g3.n.2048@g3-014 -->2K<!-- /claim --> prefill still fits in the smaller graph. Longer prefill also reaches the larger prompt graph. That maps the steps in the curves to the implementation's function selection; it does not isolate the contribution of each compiler or runtime cost.

Decode from <!-- claim:g3.n.1024@g3-015 -->1K<!-- /claim --> KV reaches <!-- claim:g3.decode.1024.ane-rate@g3-016 -->13.9 token/s<!-- /claim --> on ANE and <!-- claim:g3.decode.1024.gpu-rate@g3-017 -->30.0 token/s<!-- /claim --> on GPU. From <!-- claim:g3.n.2048@g3-018 -->2K<!-- /claim -->, ANE reaches <!-- claim:g3.decode.2048.ane-rate@g3-019 -->3.1 token/s<!-- /claim -->, while GPU reaches <!-- claim:g3.decode.2048.gpu-rate@g3-020 -->29.4 token/s<!-- /claim -->. These graph and host differences prevent reading the gap as a direct comparison of usable UMA bandwidth.

![Projection FLOP/s and modelled weight/KV reads, below the ANE graph change.](../docs/figures/g3-implied.svg)

The token rates can be expressed as effective model work. Each prefill token passes <!-- claim:g3.projection-parameters@g3-050 -->3.63 billion<!-- /claim --> projection weights, so ANE sustains <!-- claim:g3.prefill.1024.ane-tflops@g3-051 -->6.0 TFLOP/s<!-- /claim --> at <!-- claim:g3.n.1024@g3-052 -->1K<!-- /claim --> and <!-- claim:g3.short-prefill-ane-tflops@g3-053 -->5.9–6.4 TFLOP/s<!-- /claim --> across <!-- claim:g3.short-contexts@g3-054 -->500–2K<!-- /claim -->, against <!-- claim:g3.prefill.1024.gpu-tflops@g3-055 -->23.5 TFLOP/s<!-- /claim --> on GPU. For reference, the same machine's ANE synthetic FP16 convolution chain reaches about <!-- claim:g3.synthetic-fp16@g3-066 -->18.8 T source-equivalent ops/s<!-- /claim --> on one fixed shape, under the same two-ops-per-MAC definition; that is a different graph, not a hardware peak. Assuming one read of <!-- claim:g3.weight-bytes@g3-056 -->8.04 GB<!-- /claim --> of FP16 weights and the existing KV per decode step, summed over the growing cache, the model gives <!-- claim:g3.decode.1024.ane-bandwidth@g3-057 -->115 GB/s<!-- /claim --> on ANE against <!-- claim:g3.decode.1024.gpu-bandwidth@g3-058 -->249 GB/s<!-- /claim --> on GPU, and the GPU figure stays near that level to <!-- claim:g3.n.8192@g3-059 -->8K<!-- /claim -->. The nearly flat modelled byte rate is consistent with bandwidth-dominated execution, but does not identify the bottleneck. ANE’s lower rate could reflect its own memory-access limits, fixed-graph work or host overhead; these have not been separated. Both figures are useful work per second computed from the measured token rates and the model structure. The byte model excludes current-token writes and fixed-graph padding; no device counter or DRAM traffic was measured.

Graph capacity suggests a testable explanation for the drop. If a query position evaluates every key in the <!-- claim:g3.largest-graph@g3-060 -->32K<!-- /claim --> graph, its attention work is <!-- claim:g3.attention-flops-largest-graph@g3-061 -->19.3 GFLOP/token<!-- /claim --> against <!-- claim:g3.projection-flops@g3-062 -->7.27 GFLOP/token<!-- /claim --> for the projections. This is a conditional operation count for that graph; a full prefill request uses several graph capacities, and the compiler may skip masked work. The measured ANE prefill drop from <!-- claim:g3.n.2048@g3-063 -->2K<!-- /claim --> to <!-- claim:g3.n.4096@g3-064 -->4K<!-- /claim --> is <!-- claim:g3.prefill-drop@g3-065 -->9.6×<!-- /claim -->. This operation count does not attribute any part of the measured drop to graph capacity. The shape-ladder experiment described at the end would test that explanation.

## 3. Lower power and lower energy are different results

![Mean software component power during prefill and decode.](../docs/figures/g3-power.svg)

For <!-- claim:g3.n.1024@g3-021 -->1K<!-- /claim --> prefill, mean CPU + GPU + ANE power is <!-- claim:g3.prefill.1024.ane-power@g3-022 -->7.83 W<!-- /claim --> under the ANE path and <!-- claim:g3.prefill.1024.gpu-power@g3-023 -->41.18 W<!-- /claim --> under the GPU path. The energy per input token is <!-- claim:g3.prefill.1024.ane-energy@g3-024 -->0.00941 J/token<!-- /claim --> and <!-- claim:g3.prefill.1024.gpu-energy@g3-025 -->0.01273 J/token<!-- /claim -->. The slower ANE path saves component energy for this work.

![Absolute component energy per input or decode token.](../docs/figures/g3-energy.svg)

Across <!-- claim:g3.short-contexts@g3-026 -->500–2K<!-- /claim --> prefill, ANE energy per input token is <!-- claim:g3.short-prefill-energy-x@g3-027 -->0.70–0.75×<!-- /claim --> the GPU's. At <!-- claim:g3.long-contexts@g3-028 -->4K–16K<!-- /claim -->, it is <!-- claim:g3.long-prefill-energy-ratio@g3-029 -->2.08–2.43×<!-- /claim --> as much. The long-context ANE path draws little power but runs long enough to use more energy per token. Short-context decode is close; its timing-attribution ranges include equal energy. At <!-- claim:g3.decode-long-contexts@g3-030 -->2K–16K<!-- /claim -->, ANE decode uses <!-- claim:g3.long-decode-energy-ratio@g3-031 -->1.67–2.37×<!-- /claim --> as much component energy per token.

The primary quantity is the software CPU + GPU + ANE energy sum during the work interval, without subtracting an idle baseline. It is neither wall-input energy nor energy attributed exclusively to the named accelerator. The two paths have different thermal starts, and observer overhead was not independently isolated. The source power fields, receipt clocks, timing assumptions and sensitivity calculations are described in [METHODS](../docs/METHODS.md#g3-complete-model-stages).

## 4. The supplementary pair and the coverage curves

The first GPU prefill block at <!-- claim:g3.n.500@g3-032 -->500<!-- /claim --> lasted <!-- claim:g3.old-short-gpu-seconds@g3-033 -->8.57 s<!-- /claim --> after warmup. It had too few interior samples for the power-response check. After the main matrix finished, both engines completed a longer pair of <!-- claim:g3.supplement-count@g3-034 -->192<!-- /claim --> requests each using the same ongoing capture. The primary energy figure uses this pair; the original short blocks remain in the data table with their response status.

![Single-request coverage speed, separate from the warmed stage blocks.](../docs/figures/g3-coverage-speed.svg)

The initial coverage requests include first-use effects that the warmed blocks need not share. For example, GPU <!-- claim:g3.n.500@g3-035 -->500<!-- /claim --> prefill is <!-- claim:g3.coverage.500.gpu-prefill-rate@g3-036 -->1,133.2 token/s<!-- /claim --> in coverage and <!-- claim:g3.prefill.500.gpu-rate@g3-037 -->2,749.6 token/s<!-- /claim --> across the later warmed block. Neither curve measures model loading. Dividing warmed-block power by coverage speed would yield a per-token energy estimate that corresponds to neither measured block.

![Successive decode segments as KV grows.](../docs/figures/g3-decode-kv.svg)

The KV plot divides each decode request into consecutive segments. It shows behaviour inside the measured continuation, not independent repetitions. The [generated tables](../docs/MEASUREMENTS.md#g3-complete-qwen3-4b-fp16) give exact rates, component energy, work counts and source runs.

## 5. What to test next

The next useful comparison changes the ANE context ladder while keeping weights, inputs and the GPU baseline fixed. If a closer-fitting graph improves both throughput and energy at long contexts, that would separate a representation cost from the broader device comparison. More independent hosts, input families and a broader quality evaluation are also needed before using these curves to predict an application's performance.

**(2026-09-13)** [Article 06](06-qwen3-4b-matched-graphs.md) reports that comparison: with an ANE graph sized to each input, most of the long-context slowdown and extra energy went away, and the GPU stayed faster.

G3 leaves the existing quantization results as separate experiments. It uses FP16, so it does not establish an A8W4 benefit. G2's component coexistence and fan results answer other questions. [Scope](../docs/SCOPE.md#g3-complete-model-observations) · [Portable evidence](../results/historical/g3-qwen3-4b/) · [Recompute](../docs/REPRODUCING.md#g3-recomputation-and-device-replay).

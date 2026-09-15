# Findings

Each finding is a directory: the symptom, a way to reproduce or recompute it,
what the evidence establishes, and — kept separate — what is still only a
hypothesis. Boundaries that apply to every number are in
[SCOPE.md](../docs/SCOPE.md).

They inform one question: **when could an LLM earn its place on the Neural Engine instead of the GPU?** G3 and G4 A add complete-model FP16 speed and software component energy; the quantization, thermal and coexistence findings come from separate component experiments. For the other half — the things that *do* work, with
their prices — see [workarounds/](../workarounds/).

## The measured answer

**[Qwen3-4B with an ANE graph sized to each input](qwen3-4b-graph-capacity/)** — Both paths use Core AI. The GPU is faster at every input, <!-- claim:g4a.decode-gpu-faster@g4a-001 -->2.0–3.9×<!-- /claim --> in decode and <!-- claim:g4a.prefill-gpu-faster@g4a-002 -->2.9–15.9×<!-- /claim --> in prefill. ANE uses <!-- claim:g4a.short-prefill-energy-x@g4a-003 -->0.69–0.79×<!-- /claim --> the GPU's component energy for <!-- claim:g4a.short-contexts@g4a-004 -->500–2K<!-- /claim --> prefill; decode energy is close. A repeat in new host sessions runs at <!-- claim:g6.repeat.speed@g6-001 -->0.97–1.02×<!-- /claim --> the original speed; across both runs short-prefill ANE energy stays at <!-- claim:g6.two-run.short-prefill-energy-x@g6-002 -->0.68–0.81×<!-- /claim -->. Against the earlier graph ladder, ANE decode is <!-- claim:g4a.vs-g3.long-decode-speed@g4a-005 -->1.80–4.40×<!-- /claim --> faster from <!-- claim:g4a.n.2048@g4a-006 -->2K<!-- /claim -->. [Scope](../docs/SCOPE.md#g4-a-matched-graph-observations).

**[Qwen3-4B on a 256 / 2K / 32K graph ladder](qwen3-4b-prefill-decode/)** — The earlier complete-model run. GPU is faster across the measured contexts; ANE uses less component energy for short prefill, but more once it selects the 32K graph. [Complete-model scope](../docs/SCOPE.md#g3-complete-model-observations).

**[W4A16 service: a quarter of the GPU's speed, flat memory, idle fans at equal load](w4a16-service-tradeoffs/)** —
G2 measures three hosts per engine at seven sizes. Two native ANE hosts ran 33,728 stage
calls each without the historical per-call growth. At equal load both engines left the
fans at idle; only saturated GPU blocks raised them. A matrix foreground shows a smaller
tail penalty beside ANE inference. Energy remains undetermined; the screensaver and unmatched
thermal starts qualify the results.

The earlier measurements below remain separate historical observations.

**[The GPU is 2.9–5.1× faster on the same 4-bit weights](ane-vs-gpu-prefill/)** —
one first-layer MLP, same source Q4_0 weights, MLX as the GPU baseline. The GPU
runs **5.09×** faster at 1024 positions and **4.41×** at 4096. The Neural Engine
holds about 5,300 positions per second at every size while the GPU speeds up
from 64 to 1024 positions, then falls at 4096; the gap survives removal of IPC, and
at 4096 positions A8W4 runs at 1.013× W4A16 speed.

**[The historical Python gate grew by one output-sized increment per call](iosurface-per-call-growth/)**
— 1,966,080 bytes, exactly the gate output size, through four different
mitigations. That round's sustained-energy and GPU-coexistence runs were skipped.
Later native host and tiled-asset rounds passed bounded C64/C256 memory checks, with C256
at 582.19 ms versus its same-round GPU 152.25 ms for 4K, and two native G2 hosts ran
33,728 stage calls each without the growth. [Follow-up](../results/historical/native-mlp-followup.json).
The native results are separate, single-process observations.

## Constraints, and the links still to test

**[Core ML rejects the tested direct grouped-scale graph](coreml-grouped-scale-cpu/)** — the
compiler states the constraint in one line: scale may vary per output channel or
per tensor, not along the reduction dimension. Group quantization is defined by
varying it along the reduction dimension.

**[The one native path that accepts something close computes the wrong answer](coreai-flattened-scale/)**
— 1921 of 4096 values wrong, matching a separately frozen error model to the
value, with a matched negative control.

**[The rewrite that is accepted runs about 4× slower](split-decomposition-cost/)**
— a separate synthetic ablation, not a causal account of the real MLP/GPU gap.

**[Core AI runs the iOS 4-bit palettized Qwen3-4B preset on the GPU](coreai-palettized-weights-gpu/)** — with ANE preferred, the bundle passes ANE validation, its ANE compile fails, and it runs on the GPU with correct outputs and no error to the host. A matched FP16 bundle compiles and makes ANE requests; the palettized one makes none. On the complete model, FP16 on ANE decodes <!-- claim:g6.w4.fp16-faster-decode@g6-003 -->5.1–8.1×<!-- /claim --> faster.

**[A QDQ multiply on the Neural Engine dequantizes with another QDQ's scale](coreai-qdq-multiply-scale/)** — a model-free probe returns 2, 4 and 8 where the answer is 1, matching one scale-substitution rule that also predicts the older exact-grid probe. Core ML returns the same values on the Neural Engine and correct values on the CPU; this entry was titled as a Core AI defect until 2026-09-15. The first MLP of a released Gemma 4 E4B QAT checkpoint is <!-- claim:g1w.e4b.1.native.l2@g1w-001 -->339%<!-- /claim --> off; a product clamp avoids the gross error. The same page records a compile failure that moves a whole graph to the GPU, and a GELU that is nonzero at zero.

**[A8 speed-ups depend on work per call and on the weights](quantized-speedup-conditions/)** — A8W4 runs at <!-- claim:g1w.depth.both-1.speed@g1w-002 -->0.86×<!-- /claim --> W4A16 speed as one layer and <!-- claim:g1w.depth.both-128.speed@g1w-003 -->1.33×<!-- /claim --> as 128. Exact zero weights make FP16 itself <!-- claim:g1w.density.old.speed@g1w-004 -->1.88×<!-- /claim --> faster, which shrinks a quantized ratio measured against it. Eight repeated E4B QAT MLPs gain nothing: <!-- claim:g1w.e4b.native.speed@g1w-005 -->0.977×<!-- /claim -->. One real E4B gate through Core AI runs A8W4 at <!-- claim:g7.coreai.gate.64.a8w4-over-w4a16.speed@g7-069 -->0.976–0.990×<!-- /claim --> W4A16 speed at 64 positions and <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.speed@g7-070 -->1.359–1.360×<!-- /claim --> at 1024.

**[Core ML and Core AI run the same codes at similar speeds, except as a four-bit palette](coreml-coreai-same-codes/)** — on the same E4B gate and MLP codes, Core ML runs FP16 at <!-- claim:g7.ml-over-ai.fp16.speed@g7-071 -->0.979–0.998×<!-- /claim --> and the gate's W8A8 at <!-- claim:g7.ml-over-ai.w8a8.speed@g7-072 -->0.962–0.967×<!-- /claim --> Core AI's speed, but its four-bit palette graphs at <!-- claim:g7.ml-over-ai.four-bit.speed@g7-073 -->0.065–0.280×<!-- /claim -->, slower than its own FP16. The A8 graphs of the full MLP fail the reference in both runtimes.

## What a candidate arithmetic model predicts

**[An execution model for group-quantized matmul](execution-model/)** — a
written-down formula checked against 7,163,904 final Q8 outputs from two real
models, with 13 numeric exceptions. Before QDQ, E2B has 16 residuals and Qwen8B 1,194.

**[One of those exceptions is a dot product outside its own binary16 bracket](fp16-dot-residual/)**
— which rules out changing only the exact dot product's final rounding. Checkable
from 32 published scalars, with no Apple hardware.

## Running ANE models

**[One- and two-position decode queries fail on ANE](ane-short-decode-query/)** — a static-shape Qwen3-4B export with a one-position decode function fails on its first ANE request with `0xe00002c2`, in the full model and in one layer, with or without an eight-position function in the bundle. Two positions fail too; four and eight run and return the same logits. The widths are outside the exporter's default set.

**[ANECompilerService keeps deleted ANE compile inputs open](ane-compiler-service-disk/)** — every ANE model load leaves its compile input allocated until the service exits: <!-- claim:g4a.disk.held-files@g4a-007 -->29<!-- /claim --> files, <!-- claim:g4a.disk.held@g4a-008 -->178.4 GiB<!-- /claim -->, on one machine; ending the service released <!-- claim:g4a.disk.released@g4a-009 -->189.1 GiB<!-- /claim -->. [Workaround](../workarounds/#4-reclaim-disk-space-held-by-the-ane-compiler-service).

## Precision

**[Small attention products lose precision](attention-product-precision/)** — a weight-free 4,096-key attention graph is <!-- claim:g5.random.dense-l2@g5-001 -->5.22–5.42%<!-- /claim --> from its FP64 reference, localized to `P @ V` with operands near 2.4e-4. Scaling P by 64 cuts the error to <!-- claim:g5.pv.scale-64-l2@g5-002 -->0.060%<!-- /claim -->. An accurate block expression runs at <!-- claim:g5.speed.kv-unrolled-vs-dense.resident@g5-003 -->0.376×<!-- /claim --> the dense graph's speed, and blocking inside one graph gains <!-- claim:g5.speed.b1024-vs-b4096.resident@g5-004 -->1.009×<!-- /claim -->.

## Status

| Finding | Status | Reproduce without a device? |
|---|---|---|
| [Matched ANE graphs](qwen3-4b-graph-capacity/) | Two runs, six inputs, both paths; all 48 energy blocks admitted; energy ratios move between runs; decode query width 4 versus 8 unchanged in speed | Yes — recompute from per-token clocks and power fields |
| [Graph ladder](qwen3-4b-prefill-decode/) | One warmed matrix plus coverage requests; long-context graph selection mapped, cost not isolated | Same |
| [ANE compiler service disk](ane-compiler-service-disk/) | Observed on one machine and macOS build; minimal reproduction pending | Recompute the listed sizes and free-space changes |
| [Native W4A16 service](w4a16-service-tradeoffs/) | Completed component run; equal-load fans identical; coexistence signal from one matched group; energy undetermined | Recompute all selected service and sensor scalars; device rerun needs workspace assets |
| [GPU 2.9–5.1× faster than ANE](ane-vs-gpu-prefill/) | First observation: 3 of 9 planned processes | Recompute the ratios; re-measuring needs the closed workspace |
| [Output-sized Python growth per call](iosurface-per-call-growth/) | Historical Python result; native G2 hosts ran 33,728 stage calls each without it; long-term stability open | Same |
| [Core ML rejects K-grouped scales](coreml-grouped-scale-cpu/) | Reproduced on two shapes | Stored records yes; a fresh run needs a device |
| [Core AI grouped-scale discrepancy](coreai-flattened-scale/) | Reproduced; flattened-scale prediction matches, internal mechanism unobserved | Same |
| [Decomposition runs ~4× slower](split-decomposition-cost/) | Measured, cause not isolated | Yes — recompute from stored per-call rows |
| [Execution model](execution-model/) | Two models: 13 final Q8 residuals; more before QDQ | Recompute published counts; full arrays are not distributed |
| [Dot product outside the bracket](fp16-dot-residual/) | Localized; mechanism unproven | Yes — 32 published terms |
| [Short decode queries](ane-short-decode-query/) | Reproduced in 36 and 1 layers; failing operator not located; width 3 untested | Recorded outcomes yes; a device run needs the model source and exporter |
| [Palettized preset on the GPU](coreai-palettized-weights-gpu/) | One preset, one-layer matched pair plus the complete model at two inputs; rejected operation unknown | Recorded logs and G6 blocks yes; a device run needs the model source |
| [QDQ multiply scale](coreai-qdq-multiply-scale/) | Reproduced model-free in Core AI and Core ML; substitution rule predicts both Core AI probes and a new Core ML input; mechanism unobserved | Yes — raw probe outputs; a device run needs no model |
| [Quantized speed-up conditions](quantized-speedup-conditions/) | Measured on synthetic chains, one repeated E4B MLP and one real E4B gate at two sizes; causes not isolated | Yes — recompute from stored per-call timings and G7 blocks |
| [Core ML and Core AI on the same codes](coreml-coreai-same-codes/) | One Core ML four-bit representation; where Core ML's four-bit time goes is not located | Yes — recompute from G7 blocks and power frames; a device run needs the checkpoint codes |
| [Attention product precision](attention-product-precision/) | Localized to `P @ V`; synthetic inputs only | Timings yes; relative L2 values are imported scalars |

## What is not claimed

**The performance numbers describe the tested implementations.** The historical
Python path, G2 native component service, and G3/G4 A complete-model paths use
different hosts and graph schedules. Their comparisons do not isolate hardware
limits. G2 has its own three-host repetitions; the earlier stopped three-process
plan remains incomplete.

Physical INT8 execution, exclusive per-operation placement and full-model task
quality remain unestablished. G2 provides finite thermal and coexistence observations;
its energy is **undetermined** because the power capture failed. G3 and G4 A provide
software component energy, with the measurement boundaries described in
[SCOPE](../docs/SCOPE.md). Each finding is scoped to the tested chip, inputs and
recorded software stack. Repeating the arithmetic probes on another chip and
toolchain would test whether their output patterns recur.

The synthetic benchmark suite times candidates that pass its numerical and device
controls. G1-W and G5 diagnostic tables also include variants that fail their
numerical screens, so a timing alone does not establish a usable path. Toolchain
fixes are recorded in new runs, separately from the original observations.

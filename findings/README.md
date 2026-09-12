# Findings

Each finding is a directory: the symptom, a way to reproduce or recompute it,
what the evidence establishes, and — kept separate — what is still only a
hypothesis. Boundaries that apply to every number are in
[SCOPE.md](../docs/SCOPE.md).

They inform one question: **when could an LLM earn its place on the Neural Engine instead of the GPU?** G3 adds complete-model FP16 speed and software component energy; the quantization, thermal and coexistence findings come from separate component experiments. For the other half — the three things that *do* work, with
their prices — see [workarounds/](../workarounds/).

## The measured answer

**[Qwen3-4B: prefill, decode and energy](qwen3-4b-prefill-decode/)** — Both paths use Core AI. GPU is faster across the measured contexts; ANE uses less component energy for short prefill, but more with its current long-context graph. [Complete-model scope](../docs/SCOPE.md#g3-complete-model-observations).

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
quantizing activations to 8 bits changes it by 1.3%.

**[The historical Python gate grew by one output-sized increment per call](iosurface-per-call-growth/)**
— 1,966,080 bytes, exactly the gate output size, through four different
mitigations. That round's sustained-energy and GPU-coexistence runs were skipped.
Later native HOST / PIO / TILE work passed bounded C64/C256 memory checks, with C256
at 582.19 ms versus its same-round GPU 152.25 ms for 4K, and two native G2 hosts ran
33,728 stage calls each without the growth. [Follow-up](../results/historical/native-mlp-followup.json).
The HOST / PIO / TILE results are separate, single-process observations.

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

## What a candidate arithmetic model predicts

**[An execution model for group-quantized matmul](execution-model/)** — a
written-down formula checked against 7,163,904 final Q8 outputs from two real
models, with 13 numeric exceptions. Before QDQ, E2B has 16 residuals and Qwen8B 1,194.

**[One of those exceptions is a dot product outside its own binary16 bracket](fp16-dot-residual/)**
— which rules out changing only the exact dot product's final rounding. Checkable
from 32 published scalars, with no Apple hardware.

## Status

| Finding | Status | Reproduce without a device? |
|---|---|---|
| [Native W4A16 service](w4a16-service-tradeoffs/) | Completed component run; equal-load fans identical; coexistence signal from one matched group; energy undetermined | Recompute all selected service and sensor scalars; device rerun needs workspace assets |
| [GPU 2.9–5.1× faster than ANE](ane-vs-gpu-prefill/) | First observation: 3 of 9 planned processes | Recompute the ratios; re-measuring needs the closed workspace |
| [Output-sized Python growth per call](iosurface-per-call-growth/) | Historical Python result; native G2 hosts ran 33,728 stage calls each without it; long-term stability open | Same |
| [Core ML rejects K-grouped scales](coreml-grouped-scale-cpu/) | Reproduced on two shapes | Stored records yes; a fresh run needs a device |
| [Core AI grouped-scale discrepancy](coreai-flattened-scale/) | Reproduced; flattened-scale prediction matches, internal mechanism unobserved | Same |
| [Decomposition runs ~4× slower](split-decomposition-cost/) | Measured, cause not isolated | Yes — recompute from stored per-call rows |
| [Execution model](execution-model/) | Two models: 13 final Q8 residuals; more before QDQ | Recompute published counts; full arrays are not distributed |
| [Dot product outside the bracket](fp16-dot-residual/) | Localized; mechanism unproven | Yes — 32 published terms |

## What is not claimed

**The performance numbers describe implementations, not the hardware.** The
historical Python 64-position path and later native pipeline are distinct
measurements. The native host exists, C64/C256 passed bounded memory checks,
and C256 remains slower than its same-round GPU baseline. None of these results
isolates hardware cost. G2 has its own three-host repetitions; the old stopped
three-process plan remains incomplete.

Nothing here establishes physical INT8 execution, exclusive per-operation
placement, or any model-quality result. G2 now provides finite thermal and
coexistence observations. Energy is **undetermined** after a failed power
capture; the earlier skipped measurements remain skipped. Everything is scoped to one chip and the recorded stack
of each run; the two arithmetic findings are about *semantics*, which is the part
most likely to hold elsewhere, and that is what an independent replication
should test first.

Wrong outputs, CPU-selected
cases and incomplete evidence are recorded as observations; they are never given
a benchmark time. If a future toolchain fixes one of
these, the change goes into a new record; the old one is not edited.

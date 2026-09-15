# A QDQ multiply on the Neural Engine dequantizes with another QDQ's scale

**Corrected 2026-09-15.** This page was titled "A Core AI QDQ multiply dequantizes with another QDQ's scale". The same graph built with coremltools returns the same wrong values when Core ML runs it on the Neural Engine and correct values on the CPU ([Core ML](#core-ml-returns-the-same-values)). The defect is not specific to Core AI.

## Symptom

A graph with no weights and no model: split an all-ones FP16 input into `a` and `b`, quantize and dequantize `b` at scale 1/16, multiply, then quantize and dequantize the product at an output scale. Signed INT8 with zero point 0; no rounding tie or saturation occurs, so every correct output is exactly 1.

```text
y = Q_out(a × Q_1/16(b))        Q_s(x) = s × clamp(round(x / s), −128, 127)
```

Through Core AI:

| Output scale | Correct output | ANE output | ANE output with a product clamp |
|---|---:|---:|---:|
| <!-- claim:g1w.probe.s16.scale@g1w-012 -->1/16<!-- /claim --> | <!-- claim:g1w.probe.s16.reference@g1w-013 -->1<!-- /claim --> | <!-- claim:g1w.probe.s16.observed@g1w-014 -->1<!-- /claim --> | <!-- claim:g1w.probe.s16-clip.observed@g1w-015 -->1<!-- /claim --> |
| <!-- claim:g1w.probe.s8.scale@g1w-016 -->1/8<!-- /claim --> | <!-- claim:g1w.probe.s8.reference@g1w-017 -->1<!-- /claim --> | <!-- claim:g1w.probe.s8.observed@g1w-018 -->2<!-- /claim --> | <!-- claim:g1w.probe.s8-clip.observed@g1w-019 -->1<!-- /claim --> |
| <!-- claim:g1w.probe.s4.scale@g1w-020 -->1/4<!-- /claim --> | <!-- claim:g1w.probe.s4.reference@g1w-021 -->1<!-- /claim --> | <!-- claim:g1w.probe.s4.observed@g1w-022 -->4<!-- /claim --> | <!-- claim:g1w.probe.s4-clip.observed@g1w-023 -->1<!-- /claim --> |
| <!-- claim:g1w.probe.s2.scale@g1w-024 -->1/2<!-- /claim --> | <!-- claim:g1w.probe.s2.reference@g1w-025 -->1<!-- /claim --> | <!-- claim:g1w.probe.s2.observed@g1w-026 -->8<!-- /claim --> | <!-- claim:g1w.probe.s2-clip.observed@g1w-027 -->1<!-- /claim --> |

Every one of the 1,024 output values takes the value shown. Each call made one successful ANE request, the zero input returned zero, and the repeated input returned the same bytes. The clamp limits the product to `[−128 × output scale, 127 × output scale]`. The true product 1 lies inside every interval, so the clamp changes nothing mathematically.

## One substitution predicts both probes

Each wrong output equals a branch's integer code dequantized with the scale of a *different* QDQ in the same multiply:

- **This probe.** `b` quantizes to code 16. Dequantized with the output scale instead of 1/16, it gives 2, 4 and 8, matching the three wrong rows.
- **The exact-grid probe in [article 03](../../articles/03-arithmetic-compatibility.md#4-exact-grid-multiplication-exposes-a-different-problem).** `g = 2` at scale 0.5 and `u = 8` at scale 2 both quantize to code 4, so the product should be 16. The `unequal` graph returns 4: both codes dequantized with 0.5, the scale of the branch built first. Built in the opposite order, `reverse_order` returns 64: both with 2. The published record keeps only error norms and output hashes for these cases. The 1,024 outputs are all-equal FP16 arrays, so the hashes identify the values: 4 and 64.

The rule fits all seven outputs of an unclamped QDQ multiply, including the two where the substituted scale happens to be the right one (output scale 1/16, and `equal`). The clamped graphs and `explicit_q` return correct values. The rule predicts values; it does not locate the compiler or runtime stage responsible.

## Core ML returns the same values

The same eight graphs were written with the coremltools MIL builder (`slice_by_index`, `quantize`, `dequantize`, `mul` and, in the clamped arms, `clip`), converted for iOS 18 and run by a native Core ML host, once with CPU and Neural Engine allowed and once CPU only. At <!-- claim:g7.qdq.channels@g7-037 -->1,024<!-- /claim --> channels × <!-- claim:g7.qdq.positions@g7-038 -->1,024<!-- /claim --> positions, every operation prefers the Neural Engine in the compute plan and each call makes one ANE request. Every output tensor holds a single value:

| Output scale | ANE, all-ones input | ANE, all −1 | ANE, a = 1 and b = 0.5 |
|---|---:|---:|---:|
| 1/16 | <!-- claim:g7.qdq.s16.ane.original@g7-039 -->1<!-- /claim --> | <!-- claim:g7.qdq.s16.ane.negative@g7-040 -->1<!-- /claim --> | <!-- claim:g7.qdq.s16.ane.benchmark@g7-041 -->0.5<!-- /claim --> |
| 1/8 | <!-- claim:g7.qdq.s8.ane.original@g7-042 -->2<!-- /claim --> | <!-- claim:g7.qdq.s8.ane.negative@g7-043 -->2<!-- /claim --> | <!-- claim:g7.qdq.s8.ane.benchmark@g7-044 -->1<!-- /claim --> |
| 1/4 | <!-- claim:g7.qdq.s4.ane.original@g7-045 -->4<!-- /claim --> | <!-- claim:g7.qdq.s4.ane.negative@g7-046 -->4<!-- /claim --> | <!-- claim:g7.qdq.s4.ane.benchmark@g7-047 -->2<!-- /claim --> |
| 1/2 | <!-- claim:g7.qdq.s2.ane.original@g7-048 -->8<!-- /claim --> | <!-- claim:g7.qdq.s2.ane.negative@g7-049 -->8<!-- /claim --> | <!-- claim:g7.qdq.s2.ane.benchmark@g7-050 -->4<!-- /claim --> |

The correct values are 1, 1 and 0.5; the first column matches the Core AI table. The third input was not used to fit the rule: `b` = 0.5 quantizes to code 8, and code 8 dequantized with output scales 1/8, 1/4 and 1/2 gives 1, 2 and 4. With the product clamp, and on the CPU, all eight graphs return 1, 1 and 0.5. Zero inputs return zero and repeated inputs return the same bytes.

Size decides whether Core ML uses the Neural Engine at all. At the Core AI probe's 32 × 64, the compute plan prefers the CPU for every operation even with the Neural Engine allowed, with or without an identity 1×1 convolution added, and all <!-- claim:g7.qdq.small-runs@g7-051 -->32<!-- /claim --> such runs return correct values with no ANE request. The operations are elementwise, so the size does not change the expected values.

Since two frontends produce the same outputs, the substitution most likely happens in a stage they share below the frontend; that stage has not been observed.

## A released QAT checkpoint triggers it

The first MLP of [Gemma 4 E4B mobile QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) (revision `3624117c`) was exported with its original INT4 codes, per-channel weight scales and static A8 activation scales at 64 positions. Its GELU branch multiplies the up projection's QDQ output, and the product passes through another QDQ before the down projection: the probe's shape. These graphs went through Core AI. Relative L2 below is against each graph's CPU reference, over positions 6–63, which hold RMS-normalized random rows. The inputs are synthetic controls, not text activations.

| Graph | One MLP | Eight repeated MLPs |
|---|---:|---:|
| W4A16 | <!-- claim:g1w.e4b.1.bare.l2@g1w-001 -->0.21%<!-- /claim --> | <!-- claim:g1w.e4b.8.bare.l2@g1w-002 -->0.38%<!-- /claim --> |
| A8W4 as exported | <!-- claim:g1w.e4b.1.native.l2@g1w-003 -->339%<!-- /claim --> | <!-- claim:g1w.e4b.8.native.l2@g1w-004 -->509%<!-- /claim --> |
| A8W4 with the product clamp | <!-- claim:g1w.e4b.1.clip-product.l2@g1w-005 -->6.31%<!-- /claim --> | <!-- claim:g1w.e4b.8.clip-product.l2@g1w-006 -->21.6%<!-- /claim --> |

G7 rebuilt this MLP from the same codes and scales, without the RMS norm, in both Core ML and Core AI at 64 and 1024 positions. All eight W8A8 and A8W4 graphs miss the reference by <!-- claim:g7.mlp.a8.l2@g7-052 -->338–339%<!-- /claim --> on ordinary rows, and the W8A8 outputs are byte-identical between the two runtimes. With the same product clamp in both, <!-- claim:g7.mlp.clamp.l2@g7-053 -->5.6–5.8%<!-- /claim --> remains. [G7](../coreml-coreai-same-codes/#a8-graphs-of-the-full-mlp-fail-in-both-runtimes)

The clamp removes the gross error but does not make the A8 graph match; its single-layer residual is still above the 5% screen these tests used. The eight-layer stack repeats the first MLP behind a shared RMS norm, so it is not eight real layers, and none of these values is a model-quality measurement. The clamp adds no measurable time: across eight repeated MLPs the exported A8W4 graph runs at **<!-- claim:g1w.e4b.native.speed@g1w-007 -->0.977×<!-- /claim -->** W4A16 speed and the clamped graph at **<!-- claim:g1w.e4b.clip-product.speed@g1w-008 -->0.976×<!-- /claim -->**. Neither accelerates; [the speed conditions](../quantized-speedup-conditions/) cover why.

## The explicit-quantize workaround has a boundary

This section and the next were measured through Core AI only. [Workaround 2](../../workarounds/README.md#2-fix-a-qdq-multiply) expresses the quantize step as divide, round, clamp and cast. Making that division exact in FP32 before a compressed-weight convolution makes ANE compilation fail, and the whole graph then runs on the GPU. A 32-channel model-free graph, 64 positions, seeded INT4 codes:

| Graph | ANE requests per call | Whole-graph GPU fallback |
|---|---:|---|
| W4 convolution | 1 | no |
| FP16 divide → W4 convolution | 1 | no |
| **FP32 divide → FP16 cast → W4 convolution** | **0** | **yes** |
| W4 convolution → FP32 divide | 1 | no |
| FP32 divide → FP16-constant convolution | 2 | no |
| **FP32 divide → FP16 cast → W8 convolution** | **0** | **yes** |

The target-process log reads `Compiler internal error: Error: MLIR MPS to ANEC conversion failed (default/nonbonded phase)`, then `Full compile with ANE as preferred device failed. Falling back to full compile on GPU.` The fallback outputs match the CPU reference exactly, so a numerical check alone passes them. Search the log for `Falling back`; a case-sensitive search for `fallback` misses the line.

Splitting the E4B MLP so the FP32 correction runs outside the compressed convolutions avoids the fallback. Across eight repeated MLPs, the split graphs run at **<!-- claim:g1w.e4b.coarse-speed-span@g1w-009 -->0.163–0.165×<!-- /claim -->** W4A16 speed with <!-- claim:g1w.e4b.coarse-calls@g1w-010 -->17<!-- /claim --> function calls per eight-layer pass, and remain 22% from the reference.

## GELU returns a nonzero value at zero

A single-operator graph, found while isolating the multiply: Core AI's GELU returns **<!-- claim:g1w.gelu.zero-output@g1w-011 -->−0.0007538795<!-- /claim -->** for a zero input in both `approximate='tanh'` and `approximate='none'`. An explicit FP16 formula returns zero. The multiply probe contains no GELU, so this offset is a separate defect.

## Reproduce

Verify the recorded outputs without a device:

```sh
python findings/coreai-qdq-multiply-scale/repro/verify.py
python findings/coreai-qdq-multiply-scale/repro/coreml/verify.py
```

The [reproduction](repro/README.md) exports the eight graphs and runs them through a native Core AI host, and [its Core ML section](repro/README.md#core-ml) does the same through Core ML. The E4B, fallback and GELU results are imported scalars checked by `python scripts/verify_g1w.py`. The checkpoint and its activations are not redistributed.

## What this does not show

The substitution rule is fitted to the outputs of the two Core AI probes and predicted the Core ML `b` = 0.5 outputs. No compiler pass, runtime stage or hardware unit has been observed producing it, and the E4B outputs have not been checked against it value by value. The Core AI probe ran on coreai-torch 0.4.1, macOS 27.0 (26A428) and Xcode 27.0 (27A266a), with 0.4.2 untested; the Core ML probe ran with coremltools 9.0 on the same macOS build. The Core AI probe is reported as [apple/coreai-torch #112](https://github.com/apple/coreai-torch/issues/112); the Core ML run came after that report.

## Evidence

- [repro/recorded/](repro/recorded/) — the Core AI probe's raw outputs, graphs and per-call ANE request counts
- [repro/coreml/recorded/](repro/coreml/recorded/) — the Core ML probe's outputs, graphs, compute-plan devices and ANE request counts
- [g7-coreml-coreai](../../results/historical/g7-coreml-coreai/) — the G7 MLP admissions and output hashes in both runtimes
- [g1w-e4b-mobile-qat/evidence.json](../../results/historical/g1w-e4b-mobile-qat/evidence.json) — E4B relative L2, the fallback cases and log lines, and the GELU control
- [smoke.json](../../results/fresh/smoke.json) — `coreai-qdq-*`, the exact-grid probe
- [PROVENANCE.md](../../docs/PROVENANCE.md#g1-w-and-g5-imports) — sources and transformations

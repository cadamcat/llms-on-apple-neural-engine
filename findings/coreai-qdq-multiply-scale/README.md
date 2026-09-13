# A Core AI QDQ multiply dequantizes with another QDQ's scale

## Symptom

A graph with no weights and no model: split an all-ones FP16 input into `a` and `b`, quantize and dequantize `b` at scale 1/16, multiply, then quantize and dequantize the product at an output scale. Signed INT8 with zero point 0; no rounding tie or saturation occurs, so every correct output is exactly 1.

```text
y = Q_out(a × Q_1/16(b))        Q_s(x) = s × clamp(round(x / s), −128, 127)
```

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

## A released QAT checkpoint triggers it

The first MLP of [Gemma 4 E4B mobile QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) (revision `3624117c`) was exported with its original INT4 codes, per-channel weight scales and static A8 activation scales at 64 positions. Its GELU branch multiplies the up projection's QDQ output, and the product passes through another QDQ before the down projection: the probe's shape. Relative L2 below is against each graph's CPU reference, over positions 6–63, which hold RMS-normalized random rows. The inputs are synthetic controls, not text activations.

| Graph | One MLP | Eight repeated MLPs |
|---|---:|---:|
| W4A16 | <!-- claim:g1w.e4b.1.bare.l2@g1w-001 -->0.21%<!-- /claim --> | <!-- claim:g1w.e4b.8.bare.l2@g1w-002 -->0.38%<!-- /claim --> |
| A8W4 as exported | <!-- claim:g1w.e4b.1.native.l2@g1w-003 -->339%<!-- /claim --> | <!-- claim:g1w.e4b.8.native.l2@g1w-004 -->509%<!-- /claim --> |
| A8W4 with the product clamp | <!-- claim:g1w.e4b.1.clip-product.l2@g1w-005 -->6.31%<!-- /claim --> | <!-- claim:g1w.e4b.8.clip-product.l2@g1w-006 -->21.6%<!-- /claim --> |

The clamp removes the gross error but does not make the A8 graph match; its single-layer residual is still above the 5% screen these tests used. The eight-layer stack repeats the first MLP behind a shared RMS norm, so it is not eight real layers, and none of these values is a model-quality measurement. The clamp adds no measurable time: across eight repeated MLPs the exported A8W4 graph runs at **<!-- claim:g1w.e4b.native.speed@g1w-007 -->0.977×<!-- /claim -->** W4A16 speed and the clamped graph at **<!-- claim:g1w.e4b.clip-product.speed@g1w-008 -->0.976×<!-- /claim -->**. Neither accelerates; [the speed conditions](../quantized-speedup-conditions/) cover why.

## The explicit-quantize workaround has a boundary

[Workaround 2](../../workarounds/README.md#2-fix-a-qdq-multiply) expresses the quantize step as divide, round, clamp and cast. Making that division exact in FP32 before a compressed-weight convolution makes ANE compilation fail, and the whole graph then runs on the GPU. A 32-channel model-free graph, 64 positions, seeded INT4 codes:

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
```

The [reproduction](repro/README.md) exports the eight graphs and runs them through a native Core AI host. The E4B, fallback and GELU results are imported scalars checked by `python scripts/verify_g1w.py`. The checkpoint and its activations are not redistributed.

## What this does not show

The substitution rule is fitted to the outputs of the two probes. No compiler pass, runtime stage or hardware unit has been observed producing it, and the E4B outputs have not been checked against it value by value. The probe ran on coreai-torch 0.4.1, macOS 27.0 (26A428) and Xcode 27.0 (27A266a); 0.4.2 is untested. Nothing here has been reported upstream yet.

## Evidence

- [repro/recorded/](repro/recorded/) — the probe's raw outputs, graphs and per-call ANE request counts
- [g1w-e4b-mobile-qat/evidence.json](../../results/historical/g1w-e4b-mobile-qat/evidence.json) — E4B relative L2, the fallback cases and log lines, and the GELU control
- [smoke.json](../../results/fresh/smoke.json) — `coreai-qdq-*`, the exact-grid probe
- [PROVENANCE.md](../../docs/PROVENANCE.md#g1-w-and-g5-imports) — sources and transformations

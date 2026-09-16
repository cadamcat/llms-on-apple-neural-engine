# Core ML runs direct INT4 near Core AI's four-bit speed; its palette forms stay slow

[G7](../coreml-coreai-same-codes/) found Core ML's four-bit palette graphs far slower than Core AI's on the same E4B codes and left open whether another Core ML form avoids it. G8 stored the first gate projection (2560 → 10240) and MLP of [Gemma 4 E4B mobile QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) four ways in Core ML, from the same four-bit codes and per-output-channel scales, and kept G7's Core AI palette as a control:

| Name | Saved form |
|---|---|
| FP16 | the codes times the FP16 scale, as FP16 constants |
| INT8 palette | four-bit indices into a 16-entry palette built from INT8 −8…7 and the scale; G7's W4A16 |
| FP16 palette | the same indices into a per-channel FP16 palette with the scale already applied |
| Direct INT4 | the codes as signed INT4 with the scale, no palette |

The saved constants of all four decode to bitwise-identical FP16 weights. Activations stay FP16, both runtimes prefer the Neural Engine, and inputs are seeded RMS-normalized tensors at <!-- claim:g8.positions-64@g8-001 -->64<!-- /claim --> and <!-- claim:g8.positions-1024@g8-002 -->1024<!-- /claim --> positions per call, not text. Each configuration ran in <!-- claim:g8.rounds@g8-003 -->3<!-- /claim --> new host processes for 120 s each.

## Direct INT4 narrows the gap

Three-round medians, positions/s. † marks the follow-up phase described [below](#two-phases).

| Work | Positions | Core ML FP16 | Core ML INT8 palette | Core ML FP16 palette | Core ML direct INT4 | Core AI INT8 palette |
|---|---:|---:|---:|---:|---:|---:|
| Gate | 64 | <!-- claim:g8.rate.gate.64.fp16@g8-004 -->135,086<!-- /claim --> | <!-- claim:g8.rate.gate.64.int8-lut@g8-005 -->60,192<!-- /claim --> | <!-- claim:g8.rate.gate.64.fp16-lut@g8-006 -->60,165<!-- /claim --> | <!-- claim:g8.rate.gate.64.int4@g8-007 -->208,706<!-- /claim --> | <!-- claim:g8.rate.gate.64.coreai@g8-008 -->216,296<!-- /claim --> |
| Gate | 1024 | <!-- claim:g8.rate.gate.1024.fp16@g8-009 -->253,816<!-- /claim --> | <!-- claim:g8.rate.gate.1024.int8-lut@g8-010 -->24,569<!-- /claim --> | <!-- claim:g8.rate.gate.1024.fp16-lut@g8-011 -->24,567<!-- /claim --> | <!-- claim:g8.rate.gate.1024.int4@g8-012 -->368,359<!-- /claim --> | <!-- claim:g8.rate.gate.1024.coreai@g8-013 -->379,146<!-- /claim --> |
| MLP | 64 | <!-- claim:g8.rate.mlp.64.fp16@g8-014 -->55,831<!-- /claim --> | <!-- claim:g8.rate.mlp.64.int8-lut@g8-015 -->22,055<!-- /claim --> | <!-- claim:g8.rate.mlp.64.fp16-lut@g8-016 -->22,054<!-- /claim --> | <!-- claim:g8.rate.mlp.64.int4@g8-017 -->97,713<!-- /claim --> | <!-- claim:g8.rate.mlp.64.coreai@g8-018 -->99,361<!-- /claim --> |
| MLP | 1024 | <!-- claim:g8.rate.mlp.1024.fp16@g8-019 -->35,442<!-- /claim --> † | <!-- claim:g8.rate.mlp.1024.int8-lut@g8-020 -->8,217<!-- /claim --> | <!-- claim:g8.rate.mlp.1024.fp16-lut@g8-021 -->8,217<!-- /claim --> † | <!-- claim:g8.rate.mlp.1024.int4@g8-022 -->107,942<!-- /claim --> † | <!-- claim:g8.rate.mlp.1024.coreai@g8-023 -->112,861<!-- /claim --> |

Paired by round, Core ML's direct INT4 runs at <!-- claim:g8.int4-over-coreai.same-phase.speed@g8-024 -->0.952–0.985×<!-- /claim --> Core AI's palette speed in the three configurations measured in the same phase, and at <!-- claim:g8.int4-over-coreai.mlp.1024.speed@g8-025 -->0.956–0.957×<!-- /claim --> for the 1024-position MLP, which compares the two phases. Against Core ML's own FP16 it is <!-- claim:g8.int4-over-fp16.other.speed@g8-026 -->1.451–1.751×<!-- /claim --> as fast for the gate and the 64-position MLP, and <!-- claim:g8.int4-over-fp16.mlp.1024.speed@g8-027 -->3.045–3.047×<!-- /claim --> for the 1024-position MLP. That largest ratio also reflects a slow FP16 baseline: the MLP's FP16 call takes <!-- claim:g8.api.mlp.1024.fp16@g8-028 -->28.9 ms<!-- /claim --> at 1024 positions, <!-- claim:g8.api.mlp-over-gate.1024.fp16@g8-029 -->7.2×<!-- /claim --> the gate's <!-- claim:g8.api.gate.1024.fp16@g8-030 -->4.0 ms<!-- /claim -->, while the direct INT4 MLP takes <!-- claim:g8.api.mlp.1024.int4@g8-031 -->9.5 ms<!-- /claim -->, <!-- claim:g8.api.mlp-over-gate.1024.int4@g8-032 -->3.4×<!-- /claim --> its gate. G7's Core AI FP16 MLP ran at a similar <!-- claim:g8.g7.rate.mlp.1024.coreai.fp16@g8-033 -->35,549<!-- /claim --> positions/s. [Every ratio](../../docs/MEASUREMENTS.md#g8-four-bit-representations-of-the-same-e4b-codes)

For every control input, Core ML's direct INT4 output is byte-identical to Core AI's palette output, and Core ML's FP16 output is byte-identical to both Core ML palette outputs. Identical bytes show the same arithmetic result, not the same kernel or placement.

## Scaling the palette in advance does not help

The FP16 palette runs at <!-- claim:g8.fp16-lut-over-int8-lut.speed@g8-034 -->0.997–1.004×<!-- /claim --> the INT8 palette's speed. Both run at <!-- claim:g8.palettes-over-fp16.gate.1024.speed@g8-035 -->0.097×<!-- /claim --> Core ML FP16's speed on the 1024-position gate, where the ANE counter reads <!-- claim:g8.ane-power.gate.1024.int8-lut@g8-036 -->2.9 W<!-- /claim --> for the INT8 palette against <!-- claim:g8.ane-power.gate.1024.fp16@g8-037 -->9.1 W<!-- /claim --> for FP16 and <!-- claim:g8.ane-power.gate.1024.int4@g8-038 -->12.1 W<!-- /claim --> for direct INT4. Core AI's graph also stores a palette, as a lookup fed by a shift-and-scale, and reads <!-- claim:g8.ane-power.gate.1024.coreai@g8-039 -->12.4 W<!-- /claim -->. The slow path goes with Core ML's palette forms, not with palette storage as such. Where its time goes is not located.

## Energy

Component energy per position follows speed. Direct INT4 uses <!-- claim:g8.int4-over-fp16.gate.1024.energy@g8-040 -->0.898–0.923×<!-- /claim --> the energy of Core ML FP16 for the 1024-position gate and <!-- claim:g8.int4-over-fp16.mlp.1024.energy@g8-041 -->0.704–0.712×<!-- /claim --> for the 1024-position MLP, and <!-- claim:g8.int4-over-int8-lut.gate.1024.energy@g8-042 -->0.271–0.279×<!-- /claim --> the INT8 palette's energy for the gate. These are CPU + GPU + ANE software counters over each 120 s block, admitted block by block, without idle subtraction.

## Two phases

The original run rejected the first round of the three 1024-position Core ML MLP configurations before timing them. Their numeric controls had passed; the placement check, which mapped each log line's wall-clock time onto the host clock, found no ANE request inside some control calls, and their later rounds were cancelled. Mapped through the Mach clocks that the log and the host share, each of those logs shows one successful request in every control call. A follow-up timed only those <!-- claim:g8.blocks.follow-up@g8-043 -->9<!-- /claim --> blocks, with a Mach-clock placement check at the same 100 µs tolerance and the same assets, inputs, host, timing and energy rules. None of the original <!-- claim:g8.blocks.original@g8-044 -->51<!-- /claim --> blocks was repeated. The Mach-clock recount also finds a request in every control call of all <!-- claim:g8.blocks@g8-045 -->60<!-- /claim --> timed blocks.

A comparison across phases uses blocks from different capture periods. Two observations are consistent with stable speed across the runs. The <!-- claim:g8.g7-agreement.count@g8-046 -->12<!-- /claim --> configurations G8 shares with G7 run at <!-- claim:g8.g7-agreement@g8-047 -->0.997–1.001×<!-- /claim --> G7's medians, including the follow-up's FP16 MLP at <!-- claim:g8.g7-agreement.mlp.1024.fp16@g8-048 -->1.000×<!-- /claim -->. The INT8 palette MLP from the original run and the FP16 palette MLP from the follow-up have byte-identical outputs and both ran at <!-- claim:g8.rate.mlp.1024.int8-lut@g8-055 -->8,217<!-- /claim --> positions/s. Energy is less comparable: the follow-up's opening idle capture drew <!-- claim:g8.idle.follow-up.start@g8-049 -->0.658 W<!-- /claim --> against <!-- claim:g8.idle.original.start@g8-050 -->0.157 W<!-- /claim --> before the original run.

## Reproduce

```sh
python scripts/verify_g8.py
```

This checks each saved representation's operators and codes and that the Core ML forms' decoded weights match, re-applies the admission rules to the recorded controls under each phase's clock mapping, recomputes every block's speed from its window and call count and its energy from the bundled power frames, and checks both against what the run recorded. The model exports, the host and a device run need the research workspace and the checkpoint's codes.

## What this does not show

- **Not four-bit arithmetic.** A faster direct-INT4 graph does not show that the Neural Engine computes in four bits; see [SCOPE.md](../../docs/SCOPE.md#what-it-does-not-establish).
- **Two palette forms, per-output-channel scales.** Grouped scales, per-tensor palettes and other palettization options were not tested; Core ML [sends K-grouped scales to the CPU](../coreml-grouped-scale-cpu/).
- **Not a model.** A gate and an MLP with synthetic FP16 activations; no attention, KV cache, A8, text or quality evaluation.
- **Not exclusive device time.** Speed is completed positions over each block's wall clock. Placement comes from the compute plan and ANE request counts, not per-operation traces.
- **One machine and toolchain.** M5 Pro, macOS 27.0 (26A428), coremltools 9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2.

## Evidence

- [g8-coreml-four-bit](../../results/historical/g8-coreml-four-bit/) — every prepared configuration with its asset audit and admission, the three rejected admissions, <!-- claim:g8.blocks@g8-051 -->60<!-- /claim --> blocks, <!-- claim:g8.calls@g8-052 -->6,018,688<!-- /claim --> calls, <!-- claim:g8.power-samples@g8-053 -->10,625<!-- /claim --> power samples in <!-- claim:g8.captures@g8-054 -->19<!-- /claim --> captures
- [verify_g8.py](../../scripts/verify_g8.py) · [the importer](../../results/historical/import_g8.py) · [PROVENANCE.md](../../docs/PROVENANCE.md#g8-four-bit-representations)

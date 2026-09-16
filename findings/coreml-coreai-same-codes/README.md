# Core ML and Core AI run the same codes at similar speeds, except as a four-bit palette

G7 tested whether the runtime explains the missing A8W4 speed-ups in earlier rounds, which all went through Core AI. The first gate projection (2560 → 10240) and the first MLP of [Gemma 4 E4B mobile QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) (revision `3624117c`) were built twice from the checkpoint's four-bit codes and per-channel scales, once with coremltools and once with coreai-torch, in four representations. Both runtimes ran them on the Neural Engine at <!-- claim:g7.gate.positions-64@g7-001 -->64<!-- /claim --> and <!-- claim:g7.gate.positions-1024@g7-002 -->1024<!-- /claim --> positions per call, with seeded RMS-normalized activations rather than text. Each real-weight configuration that passed admission ran in <!-- claim:g7.rounds@g7-003 -->3<!-- /claim --> new host processes for 120 s each.

## Similar speeds for FP16 and W8A8

At 1024 positions, three-round medians:

| Work | Representation | Core ML positions/s | Core AI positions/s |
|---|---|---:|---:|
| Gate | FP16 decoded from the codes | <!-- claim:g7.rate.gate.1024.coreml.fp16@g7-004 -->253,802<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.fp16@g7-005 -->258,787<!-- /claim --> |
| Gate | W8A8, the same codes stored as INT8 | <!-- claim:g7.rate.gate.1024.coreml.w8a8@g7-006 -->452,439<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.w8a8@g7-007 -->468,405<!-- /claim --> |
| Gate | W4A16, four-bit indices into an INT8 palette | <!-- claim:g7.rate.gate.1024.coreml.w4a16@g7-008 -->24,569<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.w4a16@g7-009 -->379,220<!-- /claim --> |
| Gate | A8W4, the W4A16 weights with the W8A8 activation QDQ | <!-- claim:g7.rate.gate.1024.coreml.a8w4@g7-010 -->47,980<!-- /claim --> | <!-- claim:g7.rate.gate.1024.coreai.a8w4@g7-011 -->515,465<!-- /claim --> |
| MLP | FP16 decoded from the codes | <!-- claim:g7.rate.mlp.1024.coreml.fp16@g7-012 -->35,448<!-- /claim --> | <!-- claim:g7.rate.mlp.1024.coreai.fp16@g7-013 -->35,549<!-- /claim --> |
| MLP | W4A16 | <!-- claim:g7.rate.mlp.1024.coreml.w4a16@g7-014 -->8,218<!-- /claim --> | <!-- claim:g7.rate.mlp.1024.coreai.w4a16@g7-015 -->112,847<!-- /claim --> |

Across both workloads and both sizes, paired by round, Core ML runs FP16 at <!-- claim:g7.ml-over-ai.fp16.speed@g7-016 -->0.979–0.998×<!-- /claim --> and the gate's W8A8 at <!-- claim:g7.ml-over-ai.w8a8.speed@g7-017 -->0.962–0.967×<!-- /claim --> Core AI's speed, and their FP16 and W8A8 outputs are byte-identical. The four-bit graphs are the exception: Core ML runs W4A16 at <!-- claim:g7.ml-over-ai.w4a16.speed@g7-018 -->0.065–0.280×<!-- /claim --> and A8W4 at <!-- claim:g7.ml-over-ai.a8w4.speed@g7-019 -->0.093–0.273×<!-- /claim --> Core AI's speed. The gap is widest at 1024 positions. [Every configuration](../../docs/MEASUREMENTS.md#g7-core-ml-and-core-ai-on-the-same-e4b-codes)

So an earlier A8W4 result is not explained by having used Core AI rather than Core ML: for these weights Core AI's four-bit path is the faster one, and its A8W4 gate [does speed up at 1024 positions](../quantized-speedup-conditions/#positions-per-call-on-a-real-projection).

## Core ML's four-bit graph is slower than its own FP16

In the saved Core ML graph, `constexpr_blockwise_shift_scale` applies one FP16 scale per output channel to the INT8 palette −8…7; `constexpr_lut_to_dense` then expands the scaled FP16 palette through the four-bit indices. Core ML's W4A16 runs at <!-- claim:g7.coreml.w4a16-over-fp16@g7-020 -->0.097–0.445×<!-- /claim --> the speed of Core ML's FP16 on the same codes, while Core AI's W4A16 runs <!-- claim:g7.coreai.w4a16-over-fp16@g7-021 -->1.465–3.175×<!-- /claim --> faster than Core AI's FP16.

Both four-bit graphs pass the numeric screen, every Core ML operation prefers the Neural Engine in the compute plan, and every control call makes one ANE request. Core ML's W4A16 output is byte-identical to its FP16 output. In the 1024-position gate blocks the ANE counter reads <!-- claim:g7.ane-power.gate.1024.coreml.w4a16@g7-022 -->2.9 W<!-- /claim --> for Core ML's W4A16 against <!-- claim:g7.ane-power.gate.1024.coreai.w4a16@g7-023 -->12.4 W<!-- /claim --> for Core AI's, and <!-- claim:g7.ane-power.gate.1024.coreml.fp16@g7-024 -->9.1 W<!-- /claim --> for Core ML's FP16. The extra time does not appear as CPU or GPU work: CPU stays below 0.4 W and GPU near zero in those blocks. Where it goes is not located.

**Corrected 2026-09-15.** The weight description above previously put lookup before scaling, as in the builder expression. The saved Core ML graph scales the palette before lookup.

## A8 graphs of the full MLP fail in both runtimes

All <!-- claim:g7.mlp.a8.configs@g7-025 -->8<!-- /claim --> W8A8 and A8W4 MLP configurations miss the reference by <!-- claim:g7.mlp.a8.l2@g7-026 -->338–339%<!-- /claim --> on ordinary rows, against a 5% screen, and <!-- claim:g7.mlp.a8.admitted@g7-027 -->0<!-- /claim --> were timed. The W8A8 outputs are byte-identical between Core ML and Core AI. The gross error and its response to a product clamp are consistent with the [QDQ multiply defect](../coreai-qdq-multiply-scale/), which Core ML also shows on the Neural Engine. The clamp was tested in <!-- claim:g7.mlp.clamp.configs@g7-108 -->4<!-- /claim --> configurations at <!-- claim:g7.mlp.clamp.positions@g7-109 -->64<!-- /claim --> positions, covering W8A8 and A8W4 in both runtimes; it leaves <!-- claim:g7.mlp.clamp.l2@g7-028 -->5.6–5.8%<!-- /claim -->, still above the screen. The residual has not been fully attributed; the MLP outputs have not been checked value by value against the probe's substitution rule.

## The synthetic chain through Core ML

On the 128-layer K512 chain, Core ML W8A8 runs <!-- claim:g7.synthetic.w8a8-over-fp16@g7-029 -->1.88×<!-- /claim --> faster than Core ML FP16, as in the [fresh positive control](../../docs/MEASUREMENTS.md#128-layer-positive-controls). Core ML's 128-layer A8W4 misses its reference at relative L2 <!-- claim:g7.synthetic.a8w4.l2@g7-030 -->0.794<!-- /claim --> and was not timed; its two-layer version passes, and Core AI's 128-layer A8W4 passed in the fresh suites. That failure is not diagnosed.

## Energy

Component energy per position follows speed. Core ML's W4A16 uses <!-- claim:g7.coreml.gate.1024.w4a16-over-fp16.energy@g7-031 -->3.311–3.335×<!-- /claim --> the energy of Core ML's FP16 for the 1024-position gate. Within Core AI, A8W4 uses <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.energy@g7-032 -->0.642–0.647×<!-- /claim --> the energy of W4A16 there. These are CPU + GPU + ANE software counters over each 120 s block, admitted block by block, without idle subtraction.

## Reproduce

```sh
python scripts/verify_g7.py
```

This re-applies each admission rule to the recorded controls, recomputes every block's speed from its window and call count and its energy from the bundled power frames, and checks both against what the run recorded. The model exports, the host and a device run need the research workspace and the checkpoint's codes.

## What this does not show

- **One Core ML four-bit representation.** Direct signed INT4 with a blockwise scale and other palettization forms were not tested; another may run at Core AI's speed.
  **Updated 2026-09-16.** G8 stored the same codes as direct INT4 and as a pre-scaled FP16 palette: [direct INT4 runs at <!-- claim:g8.int4-over-coreai.same-phase.speed@g8-056 -->0.952–0.985×<!-- /claim --> Core AI's four-bit speed, and the FP16 palette is as slow as the INT8 palette](../coreml-direct-int4/).
- **Not a model.** A gate and an MLP with synthetic activations; no attention, KV cache, text or quality evaluation.
- **Not exclusive device time.** Speed is completed positions over each block's wall clock, with synchronous `prediction` or awaited `run`. Placement comes from the compute plan and ANE request counts, not per-operation traces; physical INT8 execution is not shown.
- **One machine and toolchain.** M5 Pro, macOS 27.0 (26A428), coremltools 9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2.

## Evidence

- [g7-coreml-coreai](../../results/historical/g7-coreml-coreai/) — every prepared configuration with its admission, <!-- claim:g7.blocks@g7-033 -->78<!-- /claim --> blocks, <!-- claim:g7.calls@g7-034 -->9,237,033<!-- /claim --> calls, <!-- claim:g7.power-samples@g7-035 -->12,353<!-- /claim --> power samples in <!-- claim:g7.captures@g7-036 -->17<!-- /claim --> captures
- [verify_g7.py](../../scripts/verify_g7.py) · [the importer](../../results/historical/import_g7.py) · [PROVENANCE.md](../../docs/PROVENANCE.md#g7-and-the-core-ml-qdq-probe)

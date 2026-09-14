# Core AI runs the iOS 4-bit palettized Qwen3-4B preset on the GPU

## Symptom

coreai-models exports Qwen3-4B for iOS with the default compression preset `4bit_weight_palettized_group32`: every projection weight becomes 4-bit indices into a 16-entry table shared by each 32 output channels. Loaded through Core AI with the delegate option `preferredDevice=NeuralEngine`, the bundle passes the ANE compiler's validation, fails its compile, and specializes anyway. It executes on the GPU; the checked first output passes the CPU reference built from the exported codes.

```text
[proxy compileModel:...] returned success=0 : ... error=Error Domain=com.apple.appleneuralengine.compiler Code=1 ... NSUnderlyingError ... Code=22
MPSGraphANEUtils.mm:993: ANE compilation failed! Error = <private>
```

The log carries no fallback line and the host sees no error; a numerical check alone passes the result.

## A matched pair

Two fresh one-layer bundles from the same exporter, with the same capacity, decode functions and trace width; only the compression differs. Each ran in a new host with the unified log streamed from before the host started.

| Bundle | ANE validation | ANE compile | Direct ANE requests | Metal shader compilations | Decode width 4 versus 8 |
|---|---|---|---:|---:|---|
| FP16 | ends | succeeds | 8 | 0 | same logits |
| W4 palettized | ends | fails | 0 | 6 | same logits |

## The complete model

G6 loaded the same preset for the full model at 1K and 4K inputs. Both admission logs show <!-- claim:g6.w4.admission-ane-requests@g6-001 -->0<!-- /claim --> direct ANE requests and up to <!-- claim:g6.w4.metal-compiles@g6-002 -->259<!-- /claim --> Metal shader compilations; every power block records no ANE counter energy, so all six fail the energy admission's ANE response rule. Decode ran at <!-- claim:g6.w4.decode.1024.rate@g6-003 -->2.87 token/s<!-- /claim --> from <!-- claim:g6.n.1024@g6-004 -->1K<!-- /claim --> and <!-- claim:g6.w4.decode.4096.rate@g6-005 -->1.29 token/s<!-- /claim --> from <!-- claim:g6.n.4096@g6-006 -->4K<!-- /claim -->; FP16 on ANE is <!-- claim:g6.w4.fp16-faster-decode@g6-007 -->5.1–8.1×<!-- /claim --> faster in decode and <!-- claim:g6.w4.fp16-faster-prefill@g6-008 -->5.2–7.0×<!-- /claim --> in prefill. The system GPU counter records <!-- claim:g6.w4.decode.1024.gpu-energy@g6-009 -->5.1 J/token<!-- /claim --> per decode token at <!-- claim:g6.n.1024@g6-010 -->1K<!-- /claim -->. The first output matched a CPU model built from the exported codes within KL <!-- claim:g6.w4.reference-kl@g6-011 -->0.00004<!-- /claim -->; against the FP16 model the quantized first output differs by KL <!-- claim:g6.w4.quantization-kl-1k@g6-012 -->0.26<!-- /claim --> at <!-- claim:g6.n.1024@g6-013 -->1K<!-- /claim -->. [G6 scope](../../docs/SCOPE.md#g6-decode-query-w4-and-repeat) · [Bundle](../../results/historical/g6-qwen3-4b/)

These are GPU timings of a graph built for ANE, through the ANE-preferred host. They are not a 4-bit ANE measurement, and not the GPU path's own quantized asset.

## Boundary

One preset, group size 32, on one machine and toolchain: macOS 27.0 (26A428), coreai-models 7304c47, coreai-torch 0.4.2. The `group8` preset, the macOS `4bit` quantization preset and the mixed 4/8-bit YAML were not loaded. The compiler's error text is private in the log, so the rejected operation is unknown. The [grouped-LUT probe](../coreai-flattened-scale/) is a different case: a two-scale-per-row convolution under coreai-torch 0.4.1 that ran on ANE and returned wrong values.

## Reproduce

```sh
python findings/coreai-palettized-weights-gpu/repro/verify.py
python scripts/verify_g6.py
```

The first checks the recorded pair: matched export records, one compile result per bundle, the compile failure and zero ANE requests for W4, the ANE requests for FP16, and equal logits for both widths. The second recomputes the complete-model blocks. The [repro README](repro/README.md) gives the export and device steps.

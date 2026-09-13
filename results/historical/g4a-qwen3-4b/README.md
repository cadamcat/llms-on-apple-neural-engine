# G4 A: complete Qwen3-4B FP16 with matched ANE graphs

This imported bundle supports offline recomputation of one run: prefill and decode speed and software component energy on one M5 Pro, with an ANE graph sized to each of six inputs and the GPU on its 32K asset. Both paths use Core AI. It contains no model weights, compiled models or logits.

| File | Content |
|---|---|
| `protocol.json` | Inputs, capacities, continuation IDs, repetitions, prefill counts, timing and admission settings, source identities |
| `requests.jsonl.gz` | Every boundary, full and prefill request result with its command, by input and path |
| `arms.json` | Request order per session, warmup and 1K reference checks |
| `power.jsonl.gz` | Selected original plist fields, receipt anchors, recorded decoding and raw-frame hashes |
| `summary-recorded.json` | The summary the runner wrote, for comparison with the recomputation |
| `asset-identity.json` | Tier graph inventories, native model hashes, compile receipts, short ANE and GPU flow checks, GPU export record, model source configuration |
| `runtime-implementation.json` | Host engine selection, host binary hashes and the Swift changes from the G3 host |
| `capture-status.json`, `closures.json` | Capture audit, run, terminal and per-host closure |
| `disk.json` | Load-gate and observer free-space readings with host session times |
| `provenance.json` | Source and product hashes and transformations |

Input token IDs and the model structure are read from the [G3 bundle](../g3-qwen3-4b/); the recomputation checks their hashes.

```sh
python scripts/verify_g4a.py
python scripts/verify_ane_compiler_disk.py
python scripts/summarize.py
python scripts/render_figures.py --check
```

Run from the repository root. These commands use only the standard library and do not execute a device. [Scope](../../../docs/SCOPE.md#g4-a-matched-graph-observations) · [Method](../../../docs/METHODS.md#g4-a-matched-graph-stages) · [Article](../../../articles/06-qwen3-4b-matched-graphs.md).

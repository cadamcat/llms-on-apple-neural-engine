# G6: decode query width, W4 palettization and a G4 A repeat

This imported bundle supports offline recomputation of one night on one M5 Pro, on the G4 A inputs and energy rules: FP16 decode with query 8 and query 4 on six matched ANE graphs, the upstream iOS 4-bit palettized preset at 1K and 4K, and a repeat of every G4 A arm in a new host with the arm order reversed. It contains no model weights, W4 codes, compiled models, logs or logits.

| File | Content |
|---|---|
| `protocol.json` | Segment list by capture, continuation IDs, repetitions, prefill counts, timing, admission and reclaim settings, source identities |
| `requests.jsonl.gz` | Every boundary, full and prefill request result with its command, by segment |
| `sessions.json` | Request order per query session and repeat arm, query order, warmup and 1K reference checks, idle windows |
| `power.jsonl.gz` | Selected original plist fields per capture, receipt anchors, recorded decoding and raw-frame hashes |
| `summary-recorded.json` | The summary the runner wrote, for comparison with the recomputation |
| `asset-identity.json` | Tier graph inventories and hashes, W4 preset and code shapes, the dequantized-code reference summary, admission results with ANE-request and Metal-compile log counts |
| `runtime-implementation.json` | Host binary hashes and the two-line query allow-list change from the G4 A host |
| `capture-status.json`, `closures.json` | Audit of each capture, run, terminal and per-host closure |
| `disk.json` | Load-gate and observer free-space readings, reclaim waits between captures, host session times |
| `provenance.json` | Source and product hashes and transformations |

Input token IDs and the model structure are read from the [G3 bundle](../g3-qwen3-4b/); the repeat ratios read the [G4 A bundle](../g4a-qwen3-4b/). The recomputation checks their hashes.

```sh
python scripts/verify_g6.py
python scripts/summarize.py
```

Run from the repository root. These commands use only the standard library and do not execute a device. [Scope](../../../docs/SCOPE.md#g6-decode-query-w4-and-repeat) · [Method](../../../docs/METHODS.md#g6-query-width-w4-and-repeat).

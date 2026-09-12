# G3: complete Qwen3-4B FP16

This imported bundle supports offline recomputation of complete-model prefill/decode speed and software component energy on one M5 Pro. Both paths use Core AI. It contains no model weights or compiled models.

| File | Content |
|---|---|
| `protocol.json` | Model/revision, precision, contexts, work budgets, source identities and timing assumptions |
| `inputs.json`, `input-identity.json` | Initial token IDs and hashes of source documentation |
| `requests.jsonl.gz` | Complete request results and commands, qualified by run and host |
| `blocks.json` | Stage work inventories and capture references |
| `power.jsonl.gz` | Selected original plist fields, receipt anchors, recorded decoding and raw-frame hashes |
| `quality-records.json` | Recorded first-output and cache controls; no portable logits replay |
| `asset-identity.json`, `runtime-implementation.json` | Recorded asset hashes and the host's engine-selection excerpt |
| `capture-status.json`, `closures.json` | Original all-channel integrity and process-closure records |
| `provenance.json` | Source/product hashes and transformations |

r4 supplies the single-request coverage curves. r5 supplies the warmed stage matrix and shared power capture. r6 supplies the longer short-prefill pair. Original short blocks remain in the bundle; the primary pair selection and its response requirement are checked by the recomputation code.

```sh
python scripts/verify_g3.py
python scripts/summarize.py
python scripts/render_figures.py --check
```

Run from the repository root. These commands use only the standard library and do not execute a device. Raw request work and the selected power fields are rechecked; original model quality, asset preparation and all-channel capture audits remain separately identified runtime records. [Scope](../../../docs/SCOPE.md#g3-complete-model-observations) · [Method](../../../docs/METHODS.md#g3-complete-model-stages) · [Article](../../../articles/05-qwen3-4b-prefill-decode-energy.md).

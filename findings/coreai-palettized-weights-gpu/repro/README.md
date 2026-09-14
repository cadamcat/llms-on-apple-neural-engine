# Palettized placement reproduction

Supports the [palettized preset finding](../README.md). No upstream issue has been filed.

## Verify the recorded pair without a device

```sh
python findings/coreai-palettized-weights-gpu/repro/verify.py
```

## Export and run on an Apple silicon Mac

Recorded environment: Apple M5 Pro, macOS 27.0 (26A428), coreai-models 7304c47 with coreai-torch 0.4.2, Qwen3-4B revision `1cfa9a72`. The bundles were exported with the [short decode query](../../ane-short-decode-query/repro/) exporter:

```sh
python export_small.py --execute --out <dir>/l1-c1280-fp16 --layers 1 --capacities 1280 --max-context 1280 --queries 4 8 64 --compression none
python export_small.py --execute --out <dir>/l1-c1280-w4 --layers 1 --capacities 1280 --max-context 1280 --queries 4 8 64 --compression 4bit_weight_palettized_group32
```

For each bundle, start `log stream --level debug --style ndjson --predicate 'process == "<host binary name>"'` before the host, then run the probe with the widened host: `q1_probe.py --execute --capacity 1280 --input quality_short --queries 8 4 --same-host --steps 2 --host <host> --asset <bundle>`. Use bundles that have not been loaded before; once specialized into the Core AI cache, a load does not call the ANE compiler again and the failure is not logged.

## Files

`recorded/` is written by `results/historical/import_palettized_placement.py` from the two probe records and log streams. Source and records are provided under the MIT license in this directory.

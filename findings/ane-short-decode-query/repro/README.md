# Decode query width reproduction

Supports the [short decode query finding](../README.md). No upstream issue has been filed.

## Verify the recorded outcomes without a device

From the repository root, with Python 3.12 and no third-party packages:

```sh
python findings/ane-short-decode-query/repro/verify.py
```

## Export and run on an Apple silicon Mac

Recorded environment: Apple M5 Pro, macOS 27.0 (26A428), coreai-models 7304c47 with its `uv` environment (coreai-torch 0.4.2, Torch 2.9.0), Qwen3-4B revision `1cfa9a72`.

`export_small.py` wraps the coreai-models iOS exporter: it overrides `IOS_STATIC_QUERY_LENS`, the traced query width and the static shape table, and can truncate the model. The case bundles were exported with:

```sh
python export_small.py --execute --out <dir>/l1-c768-q1-8-64-t8 --layers 1 --capacities 768 --queries 1 8 64
python export_small.py --execute --out <dir>/l1-c768-q1-64-t8 --layers 1 --capacities 768 --queries 1 64
python export_small.py --execute --out <dir>/l1-c768-q1-2-4-8-64-t8 --layers 1 --capacities 768 --queries 1 2 4 8 64
python export_small.py --execute --out <dir>/l1-c768-q1-8-64-t1 --layers 1 --capacities 768 --queries 1 8 64 --trace-query 1
```

The last command fails in `torch.export`. The host is the coreai-models `g3-flow-host` tool with the fixed-capacity additions used in G4 A, rebuilt with the decode width allow-list widened from `[1,8]` to `[1,2,4,8]` in `G3Host.swift` and `CoreAIStaticShapeEngine.swift`; the diff is in `recorded/observations.json`. `q1_probe.py` starts one host per width, runs a greedy request with captured logits, streams the host PID's unified log and compares widths under the same history.

## Files

`export_small.py` and `q1_probe.py` are the research-workspace scripts at import time, unchanged. They import the workspace's host driver and expect its directory layout. The earlier cases ran with earlier revisions of `q1_probe.py` that differ only in added options (`--host`, `--tier-asset`, `--same-host`, `--reference-dir`) and in using full mode for truncated bundles. `recorded/` is written by `results/historical/import_short_query.py`. Source and records are provided under the MIT license in this directory.

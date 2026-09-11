# G2 imported service evidence

A separate native W4A16 component experiment on 2026-09-11, imported from the
closed workspace by [import_g2.py](../import_g2.py). It is not a fresh run of the
public package. The [finding](../../../findings/w4a16-service-tradeoffs/) explains
the results; [scope](../../../docs/SCOPE.md#g2-service-observations) defines their limits.

| File | Contents |
|---|---|
| `protocol.json` | Workload, frozen phase/group definitions, limits, source/binary/asset identities, toolchain and original audit scope |
| `p2.jsonl.gz` | All 45 cells with control, warmup and measured scalar records; 30 measured durations per cell |
| `slots.json`, `arrivals.jsonl.gz` | All 34 slot definitions and summaries; exact frozen offsets for each arrival stream |
| `P1-*.jsonl.gz`, `P3-*.jsonl.gz`, `SAT-*.jsonl.gz` | Every terminal service event; selected timing, status, input identity and numerical-status fields |
| `sensors.jsonl.gz`, `thermal.jsonl.gz`, `resources.jsonl.gz` | All audited r4 samples in their original numeric time coordinates; no downsampling |
| `p0.json` | Four host summaries and all natural 0/32/64/128/256/512 checkpoints |
| `thermal-starts.json`, `thermal-windows.json` | Original classifications and sensor summaries, checked against the included samples |
| `power-status.json` | Original failed strict acceptance and source identities; not the complete power byte stream |
| `environment.json` | Post-run screensaver report and descriptive process-presence evidence; no per-process GPU attribution |
| `expected-summary.json` | Workspace scalar expectations; `scripts/g2/evidence.py` compares its recomputation against them |
| `provenance.json` | Every read source and emitted data file, extraction rules and hashes |

JSONL gzip streams contain one JSON object per line and have a fixed gzip
mtime. For example, `gzip -dc p2.jsonl.gz` displays the stored cells. No weight
array, activation tensor, personal system process listing or compile cache is
included. Paths in metadata are relative to the source workspace; they are
provenance identifiers, not promised paths within this repository.

```sh
# From the repository root; standard library only.
python scripts/verify_g2.py
```

The verifier checks bundle identities and recomputes timing, all-arrival service
accounting, same-input pairing, thermal starts/platforms and sampled resource
peaks. It does not reconstruct outputs from weights or re-establish ANE placement
from the original log streams. Those remain imported audit evidence.

The source extractor requires an explicit closed-workspace path and a new
output directory. It never loads a model or launches a device. The thermal-start
classifier in `scripts/g2/thermal.py` is copied from the recorded workspace
`g2_thermal.py`; its origin hash is in `provenance.json`. The portable service
accounting and plotting modules are separate implementations over selected rows.

# Historical evidence: findings, method and audit

The initial timing bundle contains ten run records copied from a closed research
workspace; later imports add separate scalar evidence from other experiments. They are labelled `historical_import: true`, they are never
overwritten by a fresh run, and they do **not** validate the current code. The
current analysis lives in the [article series](../articles/README.md) and the
fresh numbers in [MEASUREMENTS.md](MEASUREMENTS.md). Boundaries that apply to
every number are in [SCOPE.md](SCOPE.md).

## What the imported records show

The INT8 control recorded Core AI W8A8 at **34.73 T source-equivalent ops/s**,
with a paired FP16 baseline giving **1.89×**. Two Core ML W8A8 processes
recorded 34.26 and 34.25 T ops/s against their own same-round FP16 baselines.
The source graph is 128 layers at `[1, 512, 64, 64]`, with 30 measured calls
after four controls and ten warmups in each process.

The W4 bridge recorded a wide K512 A8W4 p50 of **0.400 ms** and a K32 split p50
of **1.773 ms**, both two-layer and 4096-position controls, for a recomputed
ratio of **4.44×**. That is a structural cost for this synthetic bridge. It is
not a claim that slicing, narrow convolution, reduction or fusion alone caused
it.

Both are preserved with every measured timing row, output hash, process
identity, control-admission field, source hash and epoch range in
[`selected-evidence.json`](../results/historical/selected-evidence.json),
including the same-round FP16 baselines and both wide A8W4 processes, so each
speed-up pairs within its own round. Every headline recomputes from those rows
and the recorded `projection_ops`; no summary-only number is used.

These reports use modified Hadamard and low-entropy controls. They are not a
strict reproduction of arbitrary Gemma or Q4_0 weights, and a persisted INT4
graph or a preferred-device entry is not proof of physical INT8 on the ANE.

Two further experiments from the same workspace — the cross-model validation of
a derived arithmetic model, and the bisection that localized its first
unexplained residual — are imported separately and written up in
[findings/](../findings/). Their provenance is in
[PROVENANCE.md](PROVENANCE.md); they are rechecked by
`python results/historical/tests/verify_arithmetic.py`.

The initial Python MLP comparison and its IOSurface growth are preserved in
[ane-vs-gpu-prefill.json](../results/historical/ane-vs-gpu-prefill.json). They are
followed by [native-mlp-followup.json](../results/historical/native-mlp-followup.json):
G1-HOST / PIO / TILE host and pipeline progress, separate timing rounds, and
bounded C64/C256 memory passes. The later observations do not rewrite the old
failure or establish indefinite stability. No cross-round ratio is formed by
mixing an old ANE measurement with a new GPU denominator.

## How the records were selected

A standard-library extractor
([extract.py](../results/historical/extract.py)) reads each run's `result.json`
and its paired `run.json`, hashes both files, keeps every `phase == measured`
record and recomputes the median from `duration_ns`. Each record keeps its
start and end epoch nanoseconds, duration, measured index and output SHA-256;
process identity comes from the paired run metadata. Control admission is
retained exactly as recorded: numerical pass, ANE control participation,
preferred-device evidence, memory gate, repeat hashes and child return code.

The source experiments used `[1, 512, 64, 64]`, 128 layers for the INT8
throughput controls and two layers for the bridge comparison; the bridge report
states 4096 spatial positions. The throughput denominator is the source-graph
projection recorded by the original run, not a physical MAC count. The timing
interval is the Swift prediction or function-run interval, excluding output
copy, hashing and persistence.

The extractor imports no array, model or device library. It copies no weights,
assets, system logs or compiled packages, serialises no absolute path, and
reruns nothing. It reads a configurable `ANE_SCOPE_SOURCE_ROOT` and emits
workspace-relative paths only.

## Audit limits

`result.json.records` is the source for every retained timing row; the paired
`run.json` supplies process PID, reported p50 and p95, projected ops,
runtime and format, and the run-level gates. The optional per-run
`admission.json` hash and `benchmark_admitted` field are recorded where present.

The median is recomputed from the measured `duration_ns` rather than copied.
Reported throughput stays a run summary, because it depends on that run's
`projection_ops` and timing summary — but with the full 30-row record list the
headline can be recalculated independently.

The hardware string comes from the original workspace's `before/TOOLING.md`
and macOS manifest, not inferred from a benchmark number. The fixed shape comes
from the corresponding protocols and reports. These are provenance fields: they
describe the machine that produced the imported records, and say nothing about
where the current package has been run.

Two interpretation limits are worth repeating here because they are easy to
lose when reading old records: ANE control participation is process-and-call
evidence, not per-operation accounting; and a preferred compute unit is plan
evidence, not execution. See [PROVENANCE.md](PROVENANCE.md) for the per-record
source ledger.

## Complete Qwen3-4B measurements (G3)

The [G3 bundle](../results/historical/g3-qwen3-4b/) adds complete-model FP16 coverage, warmed prefill/decode blocks and software component energy. Its three run identities remain separate: coverage, main stages/capture and the longer short-prefill supplement. [Provenance](PROVENANCE.md#g3-complete-model-import) · [Scope](SCOPE.md#g3-complete-model-observations) · [Recompute](REPRODUCING.md#g3-recomputation-and-device-replay).

# Provenance ledger

All entries below are marked `historical_import: true`. Paths are workspace-relative and omit any user-specific absolute prefix. What these records show, and how they were extracted, is in [HISTORICAL.md](HISTORICAL.md).

| Evidence | Original source | Content | Integrity |
|---|---|---|---|
| Core AI W8A8 128 | `results/g1-int8-control-20260910/runs/coreai-heldout-w8a8-128/result.json` plus paired `run.json` | 30 measured rows, 34.7269958834T headline, controls and process metadata | SHA256 recorded in `selected-evidence.json` |
| Core ML W8A8 process 1 | `results/g1-int8-control-20260910/runs/heldout-w8a8-128-1/result.json` plus paired `run.json` | 30 measured rows, 34.2644425287T | SHA256 recorded in `selected-evidence.json` |
| Core ML W8A8 process 2 | `results/g1-int8-control-20260910/runs/heldout-w8a8-128-2/result.json` plus paired `run.json` | 30 measured rows, 34.2480767946T | SHA256 recorded in `selected-evidence.json` |
| Core AI FP16 baseline | `results/g1-int8-control-20260910/runs/coreai-heldout-fp16-128/result.json` plus paired `run.json` | Same-round FP16 denominator, 30 measured rows | SHA256 recorded in `selected-evidence.json` |
| Core ML FP16 pair | `results/g1-int8-control-20260910/runs/heldout-fp16-128-1/result.json` and `heldout-fp16-128-2/result.json` plus paired `run.json` | Same-round FP16 denominators, 30 rows/process | SHA256 recorded in `selected-evidence.json` |
| Wide K512 A8W4 pair | `results/g1-w4-bridge-20260910/runs/a8w4-128-1/result.json` and `a8w4-128-2/result.json` plus paired `run.json` | Same-round full wide baselines, 30 rows/process | SHA256 recorded in `selected-evidence.json` |
| Wide K512 A8W4 | `results/g1-w4-bridge-20260910/runs/a8w4-2-bench/result.json` plus paired `run.json` | 30 measured rows, p50 0.399729ms | SHA256 recorded in `selected-evidence.json` |
| Split K32 A8W4 | `results/g1-w4-bridge-20260910/runs/split32-2-bench/result.json` plus paired `run.json` | 30 measured rows, p50 1.773229ms | SHA256 recorded in `selected-evidence.json` |

Context sources retained for interpretation include `results/g1-int8-control-20260910/REPORT.md`, `results/g1-w4-bridge-20260910/REPORT.md`, the G1 V repro package, and the G1 group Core ML report/review. The reports cite the ANEMLL public benchmark gist as external context: https://gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c. That citation is attribution/context, not an independent validation of these imported rows.

Hardware and software version claims must be taken from each original run/prepared metadata; this ledger does not infer a machine model from a headline. No absolute paths, full system logs, model weights, or compile caches are included.

## Execution-model evidence

Two further experiments were imported by
[import_arithmetic.py](../results/historical/import_arithmetic.py), which reads
`results/g1-generalize-20260910` and `results/g1-localize-20260910` in the closed
workspace and hashes every file it reads.

| Evidence | Content |
|---|---|
| [cross-model-validation.json](../results/historical/cross-model-validation.json) | Per-model output counts, residual counts and maximum relative L2 for a derived arithmetic model, plus the 13 residual coordinates with their predicted and actual pre-quantization values |
| [dot-localization.json](../results/historical/dot-localization.json) | The seven-level bisection ladder with its guards, the 32 terms of the located dot product, the single-coordinate counterfactual trace, and the open subnormal hypothesis |

Only derived scalars are copied. **No weight tensor, activation array or model
file is published**, and the extractor imports no array, model or device library.
Both files carry `historical_import: true` and are rechecked, without a device,
by [verify_arithmetic.py](../results/historical/tests/verify_arithmetic.py). What
they show is written up in [findings/](../findings/).

## ANE-versus-GPU comparison

[import_prefill.py](../results/historical/import_prefill.py) reads
`results/g1-prefill-20260910` in the closed workspace and emits
[ane-vs-gpu-prefill.json](../results/historical/ane-vs-gpu-prefill.json):
latency percentiles for three paths, the ratios between them, process
footprints, per-variant IOSurface growth, the recorded gates, the external
sources cited for the leak, and the completeness record showing that three of
nine planned timed processes finished.

It copies derived scalars only — no weights, activations or model files — and is
rechecked without a device by
[verify_prefill.py](../results/historical/tests/verify_prefill.py). The write-up
is in [findings/ane-vs-gpu-prefill](../findings/ane-vs-gpu-prefill/) and
[findings/iosurface-per-call-growth](../findings/iosurface-per-call-growth/).

## Native MLP follow-up

[native-mlp-followup.json](../results/historical/native-mlp-followup.json) retains
G1-TILE timing and memory scalars with their source identities. HOST and PIO are
the preceding host/pipeline development context, not additional timing rounds
contained in this selection. TILE used that native persistent host: C64/C256
passed bounded memory checks, and C256 4K measured 582.19 ms against its
same-round GPU 152.25 ms. For the natural 16→32 C256 memory window,
Swift grew by 16,384 bytes and the Python controller by 114,688 bytes. Long-term
stability, energy and coexistence remain unmeasured; A256's failed memory gate
is not covered by the C256 pass.

The Python MLP SDK probe used Xcode 26.6 / macOS SDK 26.5. The fresh synthetic
suites use Xcode/SDK 27.0. Follow-up runtime and binary identities are retained
per source run, not inferred from either headline. Timing interfaces and
process counts must be read within each round; no old/new measurements are pooled.

## Implementation origins

[code-origins.json](code-origins.json) lists the selected local research recipes and original source hashes. They were refactored into generic fixture generation, Core ML/Core AI exporters, persisted-asset audits and Swift hosts. No original model weights or compiled assets are shipped. Imported records describe their original protocol; fresh package results have their own source identities and are separate observations.

## Arithmetic article evidence

[arithmetic-reference-evidence.json](../results/historical/arithmetic-reference-evidence.json) is a small historical excerpt added for [article 03](../articles/03-arithmetic-compatibility.md). It retains exact JSON pointers/values, source hashes, and literal protocol excerpts. A separately labeled streaming read of the old FP16/W8A8 outputs confirms the FP16 denominator of their relative-L2 comparison. This is a document-evidence recheck, with zero new device calls. The historical source files and existing result bundles were not overwritten.

## Documentation figure sources

[figures/manifest.json](figures/manifest.json) records each SVG's source paths, source and generator hashes, and its plotted values or semantic scope. The plots use the fresh or historical source identified for each figure; distinct experiments are never pooled, and the diagrams describe documented method. Generation and checking use the Python standard library alone, execute no device experiment, and are byte-reproducible. See [FIGURES.md](FIGURES.md) for the regeneration commands and what the checks do not cover.

## G2 native W4A16 service

[G2](../results/historical/g2-w4a16-night/) is marked `historical_import: true`:
its source is the separate closed-workspace r4 run on 2026-09-11, not one of the
public package's `fresh` suites. The r4 device phase reused only the original
completed baseline; earlier stopped device attempts are not pooled.

[import_g2.py](../results/historical/import_g2.py) records every source read in
[provenance.json](../results/historical/g2-w4a16-night/provenance.json). It exports
selected fields from all P2 control/warmup/measured records and all 578,711
P1/P3/saturated events, exact arrival offsets, all audited r4 sensor/resource
samples, P0 natural checkpoints, classifications and failure statuses. Gzip
streams have fixed metadata. No sample is discarded because of latency, thermal
mismatch or memory-baseline behaviour.

The r4 run recorded no toolchain versions. `protocol.json` takes them from the
same-day preflight package verification and Xcode recheck, and the OS build from
the r3 attempt; a post-run statement says that no software changed that day. The GPU host
ran MLX 0.32.2 from the workspace `performance/.venv`, which the frozen
`g2_client.py` launches.

Absolute workspace prefixes are removed from metadata paths; numeric clocks,
values and input/output hashes are unchanged. Paths in these records identify
workspace sources and are not promised repository files. The original source
hashes and the public product hashes are distinct. Model tensors, private
process listings, complete power bytes and compiled assets are not included.

The thermal classifier `scripts/g2/thermal.py` is copied from the workspace
`g2_thermal.py`, whose hash is recorded by the extractor. The new portable
accounting checks selected rows against imported expectations. It does not
re-execute the original 87-check audit, materialize numerical references or
re-establish per-PID placement from the private unified logs.

The screensaver note is a post-run statement, supported by process-presence snapshots; the classifier and original outcomes are unchanged. Energy remains
undetermined after the complete strict power parse failed. Detailed limits and
reproduction levels live in [SCOPE](SCOPE.md#g2-service-observations) and
[REPRODUCING](REPRODUCING.md#g2-recomputation-and-device-replay).

## G3 complete-model import

[import_g3.py](../results/historical/import_g3.py) extracts closed r4, r5 and r6 records into [g3-qwen3-4b](../results/historical/g3-qwen3-4b/). r4 contributes single-request coverage and recorded output controls; r5 contributes the stage matrix and complete shared capture; r6 contributes the longer short-prefill pair. Request IDs are qualified by run, and r6 points to the same r5 power stream.

The importer reads source request results and commands, not the analysis report. It re-serializes an explicit whitelist of original plist fields for elapsed time, timestamp, thermal state, invalid flags and component power/energy. Original frame byte ranges and hashes link the extracts to the archived capture; receipt clocks and recorded decoded values allow an independent parser check of the public fields. Unselected hardware/process fields are omitted. The public extract cannot reproduce an audit of every original telemetry channel; the all-channel audit and process closure are imported records.

Inputs are token IDs from a frozen snapshot of this repository's own documentation. Input and source-document hashes are bundled; the original text snapshots, model weights, compiled assets and logits arrays are not. The model revision, asset preparation and host source identity are research-run metadata, not evidence of a new public-suite device run. Both G3 arms use Core AI; G2's MLX version does not describe G3.

The [provenance manifest](../results/historical/g3-qwen3-4b/provenance.json) hashes every read source and every imported product, and lists transformations. The portable energy and token-contract code adapts the recorded research checks; origins are listed in [code-origins.json](code-origins.json). Bundle regeneration and document maintenance do not execute a device.

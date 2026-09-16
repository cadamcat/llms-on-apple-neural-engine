# Provenance ledger

All entries below are marked `historical_import: true`. Paths are workspace-relative and omit any user-specific absolute prefix. What these records show, and how they were extracted, is in [HISTORICAL.md](HISTORICAL.md).

| Evidence | Original source | Content | Record |
|---|---|---|---|
| Core AI W8A8 128 | `results/g1-int8-control-20260910/runs/coreai-heldout-w8a8-128/result.json` plus paired `run.json` | 30 measured rows, 34.7269958834T headline, controls and process metadata | Path in `selected-evidence.json` |
| Core ML W8A8 process 1 | `results/g1-int8-control-20260910/runs/heldout-w8a8-128-1/result.json` plus paired `run.json` | 30 measured rows, 34.2644425287T | Path in `selected-evidence.json` |
| Core ML W8A8 process 2 | `results/g1-int8-control-20260910/runs/heldout-w8a8-128-2/result.json` plus paired `run.json` | 30 measured rows, 34.2480767946T | Path in `selected-evidence.json` |
| Core AI FP16 baseline | `results/g1-int8-control-20260910/runs/coreai-heldout-fp16-128/result.json` plus paired `run.json` | Same-round FP16 denominator, 30 measured rows | Path in `selected-evidence.json` |
| Core ML FP16 pair | `results/g1-int8-control-20260910/runs/heldout-fp16-128-1/result.json` and `heldout-fp16-128-2/result.json` plus paired `run.json` | Same-round FP16 denominators, 30 rows/process | Path in `selected-evidence.json` |
| Wide K512 A8W4 pair | `results/g1-w4-bridge-20260910/runs/a8w4-128-1/result.json` and `a8w4-128-2/result.json` plus paired `run.json` | Same-round full wide baselines, 30 rows/process | Path in `selected-evidence.json` |
| Wide K512 A8W4 | `results/g1-w4-bridge-20260910/runs/a8w4-2-bench/result.json` plus paired `run.json` | 30 measured rows, p50 0.399729ms | Path in `selected-evidence.json` |
| Split K32 A8W4 | `results/g1-w4-bridge-20260910/runs/split32-2-bench/result.json` plus paired `run.json` | 30 measured rows, p50 1.773229ms | Path in `selected-evidence.json` |

Context sources retained for interpretation include `results/g1-int8-control-20260910/REPORT.md`, `results/g1-w4-bridge-20260910/REPORT.md`, the G1 V repro package, and the G1 group Core ML report/review. The reports cite the ANEMLL public benchmark gist as external context: https://gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c. That citation is attribution/context, not an independent validation of these imported rows.

Hardware and software version claims must be taken from each original run/prepared metadata; this ledger does not infer a machine model from a headline. No absolute paths, full system logs, model weights, or compile caches are included.

## Execution-model evidence

Two further experiments were imported by
[import_arithmetic.py](../results/historical/import_arithmetic.py), which reads
`results/g1-generalize-20260910` and `results/g1-localize-20260910` in the closed
workspace and lists every file it reads.

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

[import_native_followup.py](../results/historical/import_native_followup.py) writes
[native-mlp-followup.json](../results/historical/native-mlp-followup.json), which retains
the timing and memory scalars of the native tiled-asset round (named G1-TILE in the
record's scope field) with their source identities. The earlier native host and
persistent-pipeline development contributes no timing rounds to this selection. The
tiled-asset round used that native persistent host: C64/C256
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

[code-origins.json](code-origins.json) lists the selected local research recipes by origin path. They were refactored into generic fixture generation, Core ML/Core AI exporters, persisted-asset audits and Swift hosts. No original model weights or compiled assets are shipped. Imported records describe their original protocol; fresh package results have their own source identities and are separate observations.

## Arithmetic article evidence

[arithmetic-reference-evidence.json](../results/historical/arithmetic-reference-evidence.json) is a small historical excerpt added for [article 03](../articles/03-arithmetic-compatibility.md). It retains exact JSON pointers and values with their source paths. A separately labeled streaming read of the old FP16/W8A8 outputs confirms the FP16 denominator of their relative-L2 comparison. This is a document-evidence recheck, with zero new device calls. The historical source files and existing result bundles were not overwritten.

## Documentation figure sources

[figures/manifest.json](figures/manifest.json) records each SVG's source paths and its plotted values or semantic scope; `render_figures.py --check` regenerates every figure and compares the bytes. The plots use the fresh or historical source identified for each figure; distinct experiments are never pooled, and the diagrams describe documented method. Generation and checking use the Python standard library alone, execute no device experiment, and are byte-reproducible. See [FIGURES.md](FIGURES.md) for the regeneration commands and what the checks do not cover.

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
values and per-request input/output hashes are unchanged. Paths in these records identify
workspace sources and are not promised repository files. Model tensors, private
process listings, complete power bytes and compiled assets are not included.

The thermal classifier `scripts/g2/thermal.py` is copied from the workspace
`g2_thermal.py`. The new portable
accounting checks selected rows against imported expectations. It does not
re-execute the original 87-check audit, materialize numerical references or
re-establish per-PID placement from the private unified logs.

The screensaver note is a post-run statement, supported by process-presence snapshots; the classifier and original outcomes are unchanged. Energy remains
undetermined after the complete strict power parse failed. Detailed limits and
reproduction levels live in [SCOPE](SCOPE.md#g2-service-observations) and
[REPRODUCING](REPRODUCING.md#g2-recomputation-and-device-replay).

## G3 complete-model import

[import_g3.py](../results/historical/import_g3.py) extracts closed r4, r5 and r6 records into [g3-qwen3-4b](../results/historical/g3-qwen3-4b/). r4 contributes single-request coverage and recorded output controls; r5 contributes the stage matrix and complete shared capture; r6 contributes the longer short-prefill pair. Request IDs are qualified by run, and r6 points to the same r5 power stream.

The importer reads source request results and commands, not the analysis report. It re-serializes an explicit whitelist of original plist fields for elapsed time, timestamp, thermal state, invalid flags and component power/energy. Original frame byte ranges link the extracts to the archived capture; receipt clocks and recorded decoded values allow an independent parser check of the public fields. Unselected hardware/process fields are omitted. The public extract cannot reproduce an audit of every original telemetry channel; the all-channel audit and process closure are imported records.

Inputs are token IDs from a frozen snapshot of this repository's own documentation. Token counts and source-document paths are bundled; the original text snapshots, model weights, compiled assets and logits arrays are not. The model revision, asset preparation and host source identity are research-run metadata, not evidence of a new public-suite device run. Both G3 arms use Core AI; G2's MLX version does not describe G3.

The [provenance manifest](../results/historical/g3-qwen3-4b/provenance.json) lists every read source and the transformations. The portable energy and token-contract code adapts the recorded research checks; origins are listed in [code-origins.json](code-origins.json). Bundle regeneration and document maintenance do not execute a device.

## G1-W and G5 imports

[import_g1w.py](../results/historical/import_g1w.py) reads the closed G1-W rounds and writes [g1w-e4b-mobile-qat](../results/historical/g1w-e4b-mobile-qat/) plus the QDQ probe records in [the finding's reproduction](../findings/coreai-qdq-multiply-scale/repro/recorded/). Timings keep every measured call's duration for the zero-weight, depth, codebook, scale and E4B rounds; warmup, control and output hashes are dropped. Numerical results are copied as scalar fields: E4B relative L2 per graph, compile-fallback cases, the GELU zero control and the unit-scale QDQ rounding counts. Unified-log messages are reduced to the compile-error and fallback lines, without source-file prefixes or temporary paths. The summaries recorded when each round closed travel alongside, so the verifier checks its recomputation against them.

The probe records are the raw FP16 outputs, saved graph text and per-call ANE request counts of the executed package. `host.swift` is byte-identical to that package; `export.py` and `run.py` were reformatted, with parsed syntax trees identical to the executed files.

[import_g5.py](../results/historical/import_g5.py) writes [g5-attention](../results/historical/g5-attention/): per-operation timings for both speed rounds, the first round's recorded per-block means, the later round's absolute medians, both rounds' paired speed medians, and relative L2, row maximum and maximum absolute error per case. Replayed cases are collapsed after checking that they agree. FP16 inputs, device outputs and exported graphs stay in the research workspace.

Both importers refuse an existing output directory and regenerate byte-identical bundles from the same sources. Each `provenance.json` lists every source read.

## G4 A import

[import_g4a.py](../results/historical/import_g4a.py) reads one closed run and writes [g4a-qwen3-4b](../results/historical/g4a-qwen3-4b/). It refuses a run that did not close in one attempt, a host that did not exit cleanly, a capture whose audit failed, and mismatches in the selected identity files frozen at launch: ANE metadata and main.hash, the GPU export record, and the host sources and binaries. Launch-frozen files are checked against their launch hashes at import; those hashes are not published. Compiled bytecode sizes and graph inventories are retained. Request results and commands are kept for every boundary, full and prefill request; the boundary request's input IDs are checked against the recorded inputs and dropped, and no logits are kept. Inputs are the same token IDs as G3; the recomputation requires the input file to be among the G3 bundle's sources.

Asset identity keeps each ANE tier's graph inventory, compile receipt and short flow check, and the GPU asset's export record. Runtime identity keeps the host's engine-selection excerpt, the launch hashes of the host binaries, which G6 compares, and the diff of the three Swift sources from the G3 host. Power frames are reduced to the same whitelist as G3. The run's recorded summary travels with the bundle so the verifier can compare. Disk fields from the load gate and the observer are kept with host start and end times; process IDs and command lines are dropped.

With `--leak-output` the importer also writes the [disk observations](../findings/ane-compiler-service-disk/evidence/) from saved terminal output: lsof rows are parsed and the per-user temporary directory is rewritten as `$TMPDIR`; the prompt, inode numbers and session user name are removed, and the importer refuses output that still contains that name or a personal path. `provenance.json` lists every source in both outputs.

## G6 import

[import_g6.py](../results/historical/import_g6.py) reads one closed night and writes [g6-qwen3-4b](../results/historical/g6-qwen3-4b/). It refuses a run whose controller did not close without failed segments, a terminal with more than one attempt, a run configuration or launch record that differs from the frozen preparation, a host that did not exit cleanly, a capture whose audit failed and mismatches in identity files frozen at launch: tier metadata and main.hash, the W4 code inventory, the W4 reference summary, both host sources and binaries. Request results and commands are kept for every request of every query session and repeat arm; input IDs are checked and dropped, and no logits are kept. W4 codes are reduced to per-projection shapes, group size and maximum index; the admission logs are reduced to counts of direct ANE requests, request failures and Metal shader compilations. Power frames of all four captures use the G4 A whitelist, tagged by capture. Load-gate, observer and reclaim-wait disk fields are kept; process IDs and command lines are dropped.

## G7 and the Core ML QDQ probe

[import_g7.py](../results/historical/import_g7.py) reads one closed run and writes [g7-coreml-coreai](../results/historical/g7-coreml-coreai/). It refuses a run with cancelled or unrun blocks, a launch record whose hash differs from the run's, a frozen workspace launch file whose bytes changed, a block that did not close, pass its controls or keep its final output, a preparation record that disagrees with its admission file, a mismatch between a recorded and a recounted ANE request count, out-of-order call clocks and a capture whose audit failed. It keeps every prepared configuration's asset audit, numeric comparisons, placement result and control-output hashes, which stand in for the unbundled outputs in the zero, repeat and cross-runtime comparisons, with Core ML compute plans reduced to operator, first output and preferred device class; one configuration's first placement check missed the `ios18.conv` prefix, and its offline recheck of the same log is kept as the admission of record. Per-call clocks are reduced to p50, p90, maximum and busy fraction. Power frames of all seventeen captures use the G4 A whitelist, tagged by capture. Launch sources from this repository are named `ane-scope/<path>`. Their current working copies may have changed since the run and are not checked against the launch hashes; workspace launch files still must match. Process IDs, command lines, logs, models and output tensors are dropped.

[import_qdq_coreml.py](../results/historical/import_qdq_coreml.py) writes the [Core ML probe records](../findings/coreai-qdq-multiply-scale/repro/coreml/recorded/): the eight persisted graph texts, the six outputs of each of the sixteen 1,024 × 1,024 runs gzip-compressed with a zero mtime, each run's preferred devices and ANE request counts, and the recorded results of the 32 × 64 runs. The three probe sources in `coreml/` are the workspace files, unchanged. The portable verifier follows the saved QDQ connections and checks both quantize and dequantize scales, zero points and product-clamp bounds. The G7 verifier requires all six control-request counts before checking ANE participation. These checks validate the archived records; the recorded measurements and admission thresholds are unchanged.

## G8 four-bit representations

[import_g8.py](../results/historical/import_g8.py) reads the closed original run and its follow-up and writes [g8-coreml-four-bit](../results/historical/g8-coreml-four-bit/). It refuses a phase that did not close or left jobs unrun, a launch record whose hash differs from the phase's, a frozen workspace launch file whose bytes changed, an original inventory other than completed, rejected and cancelled jobs, cancellations that do not match the rejections, a follow-up whose jobs are not exactly the rejected and cancelled ones with their original fields and order, a rejection other than a numerically passing admission refused only for a missing ANE control request, a block that did not close, pass its controls or keep its final output, a preparation record that disagrees with its admission file, an ANE request recount under the phase's clock mapping that differs from the record, out-of-order call clocks, a block inventory that differs from the final audit's and a capture whose audit failed.

Asset audits keep operator counts and, per projection, the codes and decoded-FP16 hashes, which the verifier compares across representations and with the reference inputs' codes; scale hashes are dropped. Each admission keeps its numeric comparisons, its placement result with the clock mapping it used, the ANE request counts under both mappings and the control-output hashes, which stand in for the unbundled outputs; Core ML compute plans are reduced to operator, first output and preferred device class. Per-call clocks are reduced to p50, p90, maximum and busy fraction. Power frames of all nineteen captures use the G4 A whitelist, tagged by phase and capture. Idle windows keep the recorded idle means, and the process closures keep their counts. Capture audits retain sampler exit and reaping status but omit sampler PIDs. The second phase's records are named `recovery` at the source and `follow-up` in the bundle. The model revision and weight shapes come from G7's records, whose inputs G8 reused. Launch sources from this repository are not checked against the launch hashes. Process IDs, command lines, logs, models and output tensors are dropped.

## Hash reduction — 2026-09-15

Published records keep a hash only where a check or statement compares it and the hashed bytes are not in the repository: G2 per-request input and output hashes (repeated outputs, equal work), the G2 slot comparisons now made on the inputs and arrival offsets themselves, the native follow-up call output hashes, the G7 control-output hashes (zero, repeat and cross-runtime identity), the G6 W4 code-inventory hashes (reference built from the exported codes), the G4 A host launch hashes that G6 compares, the fresh device-suite records and their source identity, and the grouped-LUT reproduction, which is anchored to the fresh smoke record.

Removed without changing any value: every `products` map and its product-identity check; source hash maps in `provenance.json` and the older historical records, now lists of source paths; launch and runtime source hash maps; power-frame hashes; model, tier, metadata, header, export and asset-file hashes; G7 clock, block-output and weight-code hashes; hashes of bundled outputs and archived scripts in the finding reproductions; `docs/code-origins.json` hashes; and the figure manifest's source, generator and SVG hashes, replaced by regeneration and byte comparison. The same pass removed the workspace's Chinese status sentence from `g2-w4a16-night/expected-summary.json` and three pre-registration excerpts from `arithmetic-reference-evidence.json`.

The importers were changed to write the reduced records and rerun against the closed workspace; their output matched the reduced files except for transformation text and the sources no longer read. `arithmetic-reference-evidence.json`, `coreml-native-k64-rejection.json` and `quantization-provenance.json` have no importer and were edited directly.

## Grouped-LUT reproduction

[findings/coreai-flattened-scale/repro](../findings/coreai-flattened-scale/repro/) packages two cases of the fresh `smoke` suite. The files under `frozen/ane_scope/` are unchanged copies of the measured source snapshot, with hashes equal to that record's source identity; the recorded outputs, fixture arrays and MLIR text are copied unchanged, and `recorded/results.json` selects the two cases and their environment from the published record. `run_device.py`, `verify.py` and `archive.py` are new packaging helpers written for the reproduction, not part of the measured source.

## Short decode query import

[import_short_query.py](../results/historical/import_short_query.py) writes the [recorded outcomes](../findings/ane-short-decode-query/repro/recorded/) of five diagnostic probe runs. For each width it keeps the probe result, host return code and same-history comparison; the host's unified log is reduced to direct ANE request successes, failure status lines and loaded function names, and the MPSGraph assertion to its error code and counts. It keeps the decode functions' input signatures, the `torch.export` constraint line of the width-1 trace export and the two-line host allow-list diff. The archived export and probe scripts are checked against the hashes recorded at import.

## Palettized placement import

[import_palettized_placement.py](../results/historical/import_palettized_placement.py) writes the [recorded pair](../findings/coreai-palettized-weights-gpu/repro/recorded/) from two one-layer bundles. It keeps each export record and probe result; each process-name log stream is reduced to direct ANE request and Metal shader-compile counts plus its validation, compile, load, compile-failure and delegate-option lines, with pointer values and build paths removed.

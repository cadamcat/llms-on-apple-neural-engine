# What was run — 2026-09-10–11

An executed record of the checks behind the published numbers. No network
access, model download or CI dispatch took part in producing them. Boundaries
that apply to every number are in [SCOPE.md](SCOPE.md).

## Checks completed before the readability refactor

| Check | Outcome |
|---|---|
| Offline independent environment | Python 3.12.14, 32 packages from a local cache; the runtime belongs to this repository |
| Portable reference and evidence tests | 8 passed |
| Historical recalculation | Medians, throughput and paired ratios recompute from the bundled measured rows |
| Wheel build | Contains the Python modules, both Swift resources, LICENSE and NOTICE |
| Isolated wheel smoke | Fresh environment with the checkout's source directory moved aside; 14 cases and 56 controls completed |
| Full throughput | Five depth-2 controls, then five depth-128 arms in three independent processes: 450 measured predictions |
| SplitConv | Two controls, then three wide/split process pairs: 180 measured predictions |
| Evidence recheck | Data, asset, host and output identity, numeric references, PID-window placement and statistics all consistent; `source_matches_run` was true at the original recheck for the three selected fresh suites |

Raw artifacts for these checks stay in the ignored `runs/` tree and are not
published; what ships is the selected evidence in
[`results/fresh/`](../results/fresh/) plus each suite's verification fields.

The throughput and SplitConv timing arms passed their numeric, ANE
participation and footprint gates. **A completed smoke suite does not mean every
compatibility result passed numerically.** Core AI native K64 and the
unequal and reversed QDQ cases remain numerical mismatches, and the small Core
ML group cases select CPU. See [MEASUREMENTS.md](MEASUREMENTS.md) and
[findings/](../findings/).

A later documentation pass added the [SVG figures](FIGURES.md), derived from
stored results. A subsequent readability refactor also changed eight Python
modules. It did **not** rerun the device validation above.

## Current checkout versus the measured source

The current `_coreai.py`, `_coreml.py`, `cli.py`, `common.py`, `controller.py`,
`evidence.py`, `exporting.py` and `guard.py` differ from all three selected run
identities. `pyproject.toml` differs from the throughput and split identities after
description and project-name edits on 2026-09-11. `uv.lock` also differs from those
identities after the project rename; dependency versions and hashes are unchanged.
Both Swift hosts and the numerical reference still match.
The [current identity report](../results/validation/source-identity.json)
lists each recorded and current SHA256; recompute it with
`python scripts/check_source_identity.py --check`.

The stored `source_matches_run: true` fields describe the **original check**,
not today's changed Python code. Portable tests and scalar
recalculation can check arithmetic and recorded evidence; they cannot replace a
new device smoke run of this checkout. The source differences are not, by
themselves, evidence of a regression or of semantic equivalence. Fresh device
validation of the changed orchestration remains pending.

The local correction pass passed the portable tests, the historical and native
follow-up recalculations, the registered-number checks and figure regeneration.
No model was loaded and no device result was refreshed during that pass.

## Resources observed

Across the exploratory smoke, compatibility retry, throughput, split and
clean-wheel smoke runs, sampled owned-tree RSS peaked at about **0.436 GiB**,
the free-percentage metric stayed at **80% or above**, and swap stayed at
**2.38 MiB**. The largest individual result tree was about **1080 MiB**. All of
these stayed inside the frozen limits.

Transient peaks, and framework or compiler services outside the owned process
tree, are not fully measured. Power, whole-machine energy and long-term thermal
behaviour were not tested in these fresh suites.

The separate [native MLP follow-up](../results/historical/native-mlp-followup.json)
imports later research-workspace observations: one timing process per arm and
32-request short-memory controls. It is not a device rerun of the public
harness, does not complete the older Python-era 3-of-9 protocol, and does not
establish long-term residency.

## Corrections during validation

The first smoke attempt completed 12 cases. Two Core AI group exports were
rejected by the audit parser, which did not yet handle a hexadecimal constant
and an open-ended slice. The parser was corrected, the eight compatibility
cases were rerun, and then the full 14-case suite was rerun from the isolated
wheel. **References and numerical thresholds were unchanged.**

An independent code review noted that Core AI's requested compute unit cannot
supply per-operation placement proof; the documentation and the result fields
now carry that limit explicitly.

`verify` reports whether its current source matches the source that produced a
run, so evidence consistency after a later code change is never mistaken for
validation of that change.

## Limits of this record

Only the pinned stack was tested: Apple M5 Pro, macOS 27.0 build 26A428,
Xcode 27.0, coremltools 9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2. The CI
workflow is defined but had not been dispatched when this record was written,
and a fresh coreai-torch 0.4.2 run is not claimed — the historical 0.4.1/0.4.2
matrix is retained separately. Nothing here amounts to a
physical INT8 proof or a product-level claim about LLM inference.

## G2 import and portable checks — 2026-09-11

The separate [G2 workspace run](../results/historical/g2-w4a16-night/) was
imported after it closed. This documentation pass performed no new model load,
conversion, device experiment or power capture. The public runtime modules
were unchanged, so the eight-module device-smoke gap above still applies.

| Check | Outcome |
|---|---|
| Portable reference and evidence tests | 17 passed, including nine new G2 boundary cases |
| G2 scalar recomputation | 45 P2 cells, 34 service slots, 578,711 events and 14 thermal groups checked against imported expectations |
| Historical and native recalculation | Existing checks passed; old results were not rewritten |
| Registered document values | 63 checked against the bundled records |
| SVG identity and deterministic generation | 16 verified; regeneration was byte-identical; the original ten SVGs were unchanged |
| New SVG visual review | All six rendered and inspected in light and dark themes, at 900 pixels wide; no text overflow |

The G2 checks cover all-arrival denominators, observation versus drain
completions, missing/duplicate IDs, saturation censoring and insufficient
thermal coverage. They also check file identities, per-host timing, input
pairing, temperature summaries and sampled memory peaks. The thermal-start
classifier follows the recorded workspace implementation; this is not a
separate experimental validation of its engineering tolerances.

The original 87-check device/asset audit remains imported evidence. All six equal-work energy comparisons
remain undetermined; the display condition and thermal-start mismatches
apply to the findings. Run `python scripts/verify_g2.py` to repeat
the public scalar checks; [REPRODUCING](REPRODUCING.md#g2-recomputation-and-device-replay)
distinguishes them from full raw-audit and device replay. The CI configuration
now includes this command, but no GitHub workflow was dispatched.

## README references and maintenance checks — 2026-09-11

Both README openings now bind each key displayed quantity and unit to a named
evidence source and document location. Disposable-copy tests cover a wrong
card with a correct repeat elsewhere, changed units, larger-number collisions,
missing or duplicate locations and malformed markers. Rewording and moving the
surrounding prose still pass. The full portable suite, historical and G2
recalculations, registered-value checks and source-identity check passed.

Figure regeneration updated the source manifest; the complete SVG inventory
and bytes matched the saved pre-edit copy. Local browser previews confirmed
that the markers are invisible and the reproduction command table renders
correctly. The Python device orchestration and selected experiment records
were unchanged; its pending device smoke remains above. Manual workflow
dispatch is configured but has not been run on GitHub.

## Project name — 2026-09-11

The public project and Python distribution are now `llms-on-apple-neural-engine`,
displayed as **LLMs on Apple Neural Engine**. The synthetic suites retain the
`ane-scope` command and `ane_scope` module. An offline lock check and a wheel
built in a disposable directory verified the renamed metadata, command entry
point and unchanged Python and Swift resources. The command help was loaded
from that wheel without device execution. Historical records keep their original
names; this rename did not rerun the device suites.

## G3 complete-model import — 2026-09-12

The G3 portable check validates source-field decoding, raw request/token/graph clocks, KV progression, completed work, shared-capture references and primary-pair admission. It recomputes both coverage rates and warmed-stage speed, component power, J/token and timing-attribution bounds. The original all-channel capture audit, first-output numerical checks and runtime asset identity are imported records with the [scope stated separately](SCOPE.md#g3-complete-model-observations).

Disposable-bundle tests change input counts, phase clocks, power sample inventory, decoded power, work budgets and capture references. Selecting the original short GPU prefill block also fails its response check. A constant-power fixture independently checks the energy integral. Named document quantities bind each required numeric reference to the corresponding record; figures regenerate through the existing standard-library SVG canvas. Run `python scripts/verify_g3.py` and the portable checks above to check the current checkout.

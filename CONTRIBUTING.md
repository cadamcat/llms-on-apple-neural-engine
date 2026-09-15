# Contributing

Work from a new output directory, preserve existing evidence, and keep changes
small and reviewable.

Before a change, read [docs/SCOPE.md](docs/SCOPE.md) and
[docs/REPRODUCING.md](docs/REPRODUCING.md), and record which case, data, source
and runtime the change affects. A new result about the accelerator belongs in
[findings/](findings/) as its own directory, with the symptom, a way to
reproduce it, and any mechanism clearly marked as established or hypothesised.
Something that *works* belongs in [workarounds/](workarounds/) instead — with its
price and its boundary stated as plainly as the benefit.
Do not modify a historical result in place. A new result must identify itself as
`fresh_run`; an imported record must keep `historical_import: true`.

The [articles](articles/README.md) explain the experiments as a series. English is the default; Chinese versions live in `articles/zh/` with Chinese article filenames and a `README.md` index. Update both languages together, including evidence, qualifications, figure captions and numeric registrations, and keep their language links reciprocal. Keep each prose paragraph on one source line; preserve the structure of code, tables and equations.

Keep reference generation deterministic and independent of device execution.
Export, compile, device execution, verification and reporting stay separable
stages. Importing a module must never trigger conversion, compilation, model
loading, a device call, a download, or a write outside an explicit output path.

Device work is serial and must use the resource guard. A case completes its four
controls before any throughput measurement. Numerical admission, device
participation, compiler-plan evidence and benchmark eligibility are distinct
fields, and an unsupported or incomplete case must stay visible as one.

**Do not change a threshold after seeing a result.** Amend the protocol, name
it, and start a new case instead. Updating a figure is not permission to change
a reference or a historical result either.

Do not add model weights, compiled assets, full system logs, personal absolute
paths, credentials or user data. Use relative provenance paths; add a hash only
where a check compares it and the hashed bytes cannot be published.

Keep the **device suites model-free and reference-checkable** — that is what lets
anyone run them. Real-model results belong here as imported derived scalars, the
way the [comparison](findings/ane-vs-gpu-prefill/) and the
[execution model](findings/execution-model/) do, and not as a model loader or an
inference stack inside the package.

## Before opening a pull request

Run the [portable checks and figure regeneration](docs/REPRODUCING.md#checks-that-need-no-device)
used by CI. They require the standard library plus NumPy and no Apple device.
If you changed a published number, regenerate with
`python scripts/summarize.py --write`; if you changed a figure generator or a
document a figure cites, regenerate with
`python scripts/render_figures.py --write` and review the result. Include the
commands you ran, the validation scope and the limitations
in the change description, including any checks not run.

Key README quantities use `<!-- claim:SOURCE@LOCATION -->value unit<!-- /claim -->`
markers. Keep only the displayed quantity inside a marker; surrounding wording
and document order may change. [scripts/doc_claims.py](scripts/doc_claims.py)
defines the source fields, unit formatting and required locations. Each location
is checked independently, including repeated quantities. Other registered values
are listed in `scripts/summarize.py` and `scripts/g2/evidence.py`; their checks
require a complete numeric token somewhere in the document. A new checker rule
needs a named failing case in a disposable document copy.

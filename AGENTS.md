# Agent guidance

Prefer the smallest coherent change, and inspect the relevant files before
editing. Preserve unrelated work and preserve historical evidence.

Never run network downloads, publish, push, deploy or contact external services
as part of ordinary development. Do not trigger conversion, compilation, model
loading, device execution or large-array work implicitly during an import or a
documentation check.

Use relative paths in artifacts. Record source hashes, dependency versions,
requested devices, actual evidence and failure states. Keep historical imports
separate from fresh runs. **Do not infer ANE execution from graph presence or a
preferred compute device, and do not infer physical integer arithmetic from a
quantized representation.** [docs/SCOPE.md](docs/SCOPE.md) is the single place
where those boundaries are stated; link to it rather than restating it in a new
document.

Start from [findings/](findings/) for what this repository has established and
[workarounds/](workarounds/) for what works despite it,
[docs/SCOPE.md](docs/SCOPE.md) for what the numbers mean,
[docs/METHODS.md](docs/METHODS.md) for how they are produced and
[docs/REPRODUCING.md](docs/REPRODUCING.md) for the commands. Keep an established
result and a hypothesis visibly apart; a hypothesis needs its falsification
condition written down beside it.

Requirements follow the latest explicit instruction; implementation status follows the code
and Git; validation follows evidence for the corresponding source identity.
Investigate disagreements between those three rather than letting a compressed
summary settle them.

Use `python -m unittest discover -s tests -v` for portable changes and the
guarded CLI suite for device changes. After touching a published number, a
figure generator or a source document, run `python scripts/summarize.py` and
`python scripts/render_figures.py --check`; for imported evidence also run
`python results/historical/tests/verify_prefill.py` and
`python results/historical/tests/verify_arithmetic.py`. All are standard-library
only. Root-level documentation describes the public interface. Internal worker
commands are not a substitute for the guard.

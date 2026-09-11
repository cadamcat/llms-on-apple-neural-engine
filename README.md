# LLMs on Apple Neural Engine

English | [中文](README.zh-CN.md)

**Can a 4-bit LLM earn its place on the Apple Neural Engine instead of the GPU? On one
M5 Pro, the native ANE path runs the same first-layer MLP at about a quarter of the GPU's
speed. Its memory stays flat where the Python path leaked, and at equal load both engines
leave the fans at idle.**

<table>
<tr>
<td><b><!-- claim:g2.ane-share@speed-card -->24.8%<!-- /claim --></b><br><sub>ANE speed as a share of GPU at 1024 positions; three hosts each</sub></td>
<td><b><!-- claim:g2.native-calls@calls-card -->33,728 calls<!-- /claim --></b><br><sub>per native ANE host, ending <!-- claim:g2.native-shrink@memory-card -->60–62 MiB<!-- /claim --> smaller</sub></td>
<td><b><!-- claim:g2.temperature-gap@temperature-card -->1.3–3.3 °C<!-- /claim --></b><br><sub>warmer GPU sensor under GPU inference at equal load; fans idle on both</sub></td>
<td><b><!-- claim:arithmetic.q8-summary@arithmetic-card -->13 / 7,163,904<!-- /claim --></b><br><sub>final Q8 outputs a candidate arithmetic model does not match</sub></td>
</tr>
</table>

**Start here:** [Should I use it?](#the-short-answer) · [What actually works](workarounds/) ·
[Findings](findings/) · [Why it is slow](#why-it-is-slow) · [Reproduce](docs/REPRODUCING.md) ·
[What the numbers mean](docs/SCOPE.md) · [Articles](articles/README.md)

![G2 first-layer MLP: all three independent host curves at seven position counts, with per-engine medians. The ANE stays near 6,760 positions per second; the GPU runs between about 18,500 and 31,600.](docs/figures/g2-throughput.svg)

At 1024 positions the ANE path processes about **<!-- claim:g2.ane-rate@ane-rate -->6,760 positions/s<!-- /claim -->** and MLX on the GPU
about **<!-- claim:g2.gpu-rate@gpu-rate -->27,300 positions/s<!-- /claim -->**, the median of three independent hosts per engine. Above 256 positions the
ANE line is flat because every request is split into fixed 256-position tiles; that
describes this pipeline, not the hardware's ceiling. Sustained service puts the ANE at
**<!-- claim:g2.service-share@service-share -->27.0%<!-- /claim -->** of the GPU instead: the same few milliseconds of per-request checking and logging
weigh more on the faster engine.

![Completed requests per second against GPU sensor temperature and fan 0 speed. At equal load both engines stay at idle fan speed; only the saturated GPU blocks raise it.](docs/figures/g2-load-fans.svg)

At three equal arrival rates, up to **<!-- claim:g2.equal-rate-max@equal-rate -->4.9 requests/s<!-- /claim -->**, fans stayed at their idle speed of
about **<!-- claim:g2.fan-idle@fan-idle -->1,350 RPM<!-- /claim -->** on both engines, while the GPU sensor read **<!-- claim:g2.temperature-gap@temperature-detail -->1.3–3.3 °C<!-- /claim -->** warmer when
the GPU did the work. At full load the ANE, completing **<!-- claim:g2.ane-saturated@ane-saturated -->6.4 requests/s<!-- /claim -->**, still left the
fans at idle; the GPU completed **<!-- claim:g2.gpu-saturated@gpu-saturated -->23.7 requests/s<!-- /claim -->** and ran fan 0 at **<!-- claim:g2.fan-saturated@fan-saturated -->3,100–3,500 RPM<!-- /claim -->**. GPU loads in
between were not measured, so the point where its fans start rising is open. Energy is
undetermined: the power capture failed its integrity and clock checks.

## Measured on

| Machine | Memory | OS | Toolchain |
|---|---|---|---|
| **Apple M5 Pro** | 48 GiB | macOS 27.0 | G2: build 26A428 · Xcode 27.0 · MLX 0.32.2 (GPU host)<br>Fresh suites and native follow-up: build 26A428 · Xcode 27.0 (Swift 6.4); the suites add coremltools 9.0 · coreai-torch 0.4.1 · coreai-core 1.0.0b2<br>Historical Python MLP: Xcode 26.6 · MLX 0.32.2<br>Historical quantization matrix: build 26A5425a · coreai-torch 0.4.1 and 0.4.2 |

## The short answer

| If your model is… | On this toolchain, today |
|---|---|
| **The tested direct signed-INT4 Core ML K64 graph**, with two K32 scales per row | Numerically correct, CPU-selected. A related historical graph reports `ANE only support per-cout/per-tensor quantization`; this is not a test of every Q4 format or representation |
| **K32 decomposition** into per-output-channel scales | Core AI passes the small compatibility probe. A separate synthetic ablation is [about 4× slower](findings/split-decomposition-cost/); that is not a universal cost |
| **Grouped 4-bit through Core AI's native LUT path** | Accepted, shows real ANE activity, and [returns the wrong answer](findings/coreai-flattened-scale/) — 1921 of 4096 values, predictably |
| **W4A16 served from the native ANE host** | [About a quarter of the GPU's speed](findings/w4a16-service-tradeoffs/). At equal load the fans stay at idle on both engines; a matrix foreground shows a smaller tail penalty beside ANE inference. Energy undetermined |
| **Resident execution** | The historical Python gate retained 1,966,080 bytes per call. Two native G2 hosts ran 33,728 stage calls each and ended 60–62 MiB smaller; [indefinite residency is untested](findings/iosurface-per-call-growth/) |
| **W4A16 versus A8W4** | Historical 4K pair: [within 1.3%](findings/ane-vs-gpu-prefill/), without an established A8 speed benefit. G2 tests W4A16 only |
| **The tested W8A8 synthetic control** | Accelerates — Core AI is **1.86–1.87×** over its FP16 baseline on the controlled 128-layer chain |

Earlier rounds stay separate records. The historical Python MLP measured the GPU **2.87×**
faster at 64 positions, **5.09×** at 1024 and **4.41×** at 4096, with one timed process per
path and an early stop; native C256 later measured 582.19 ms against the GPU's 152.25 ms at
4096 positions, a native `run` time that still includes runtime and synchronization.
[Python round](findings/ane-vs-gpu-prefill/) ·
[native follow-up](results/historical/native-mlp-followup.json). Limits for each number
are in [SCOPE.md](docs/SCOPE.md).

## What is in here

Four parts, each usable on its own:

| Part | What it is |
|---|---|
| 🔧 **Three things that do work** | The rewrite that gets grouped 4-bit onto the accelerator at all, the graph expression that fixes an unequal-scale multiply, and the quantization scheme that simply accelerates — each with its price and its boundary written down, and each a case you can run. [workarounds/](workarounds/) |
| 📊 **Component comparisons** | [G2](findings/w4a16-service-tradeoffs/) measures seven-size native W4A16 speed, native host memory, equal-rate and saturated temperature and fan response, and GPU foreground tails. The [earlier A8W4/W4A16/GPU comparison](findings/ane-vs-gpu-prefill/) retains its numerical controls, per-PID evidence and early stop. Separate rounds, not pooled estimates. |
| 🐛 **Reproducible defects** | Two toolchain failures with minimal reproductions, expected wrong outputs and matched negative controls; one memory leak with four failed mitigations and an external corroboration; one structural cost measured in three paired process rounds. [findings/](findings/) |
| 🔬 **An arithmetic model** | A candidate arithmetic model checked against 7,163,904 final Q8 gate outputs from two real models, with 13 Q8 mismatches — and one localized 32-term dot product it cannot explain, checkable from published scalars with no Apple hardware. [The model](findings/execution-model/) · [the residual](findings/fp16-dot-residual/) |

## Who this is for

- 🟡 **You want to move inference off the GPU and are deciding whether it is worth it.**
  → [the short answer](#the-short-answer), then
  [the service comparison](findings/w4a16-service-tradeoffs/) with its caveats.
- 🔴 **Your grouped-quantized model compiles but runs on the CPU, or returns garbage.**
  → [what to do about it](workarounds/), then
  [Core ML's constraint](findings/coreml-grouped-scale-cpu/) and
  [Core AI's scale bug](findings/coreai-flattened-scale/) for why.
- 🟠 **Your long-running ANE process keeps growing.**
  → the historical Python binding retained [1.875 MiB per call](findings/iosurface-per-call-growth/),
  with what did not fix it and who else has reported it; two native Swift hosts ran 33,728
  stage calls each without that growth.
- 🔬 **You are debugging quantized numerics and cannot tell a rounding difference from a
  bug.** → [the execution model](findings/execution-model/) and
  [the residual](findings/fp16-dot-residual/).
- 🧪 **You want to reproduce or refute this.** → [REPRODUCING.md](docs/REPRODUCING.md).
  The bundled records support offline checks of the registered claims; full historical
  array replay still needs the original assets.

**Status.** A research artifact, not a supported product. No external replication
yet. Three conditions qualify G2: an animated
screensaver was running, found after the run; five of six coexistence groups had unmatched
thermal starts; and the power capture failed, so energy is undetermined.
[G2 scope](docs/SCOPE.md#g2-service-observations) · [RESEARCH.md](docs/RESEARCH.md)

## Why it is slow

The following constraints are measured. Their contribution to the real MLP's GPU gap
has **not** been isolated:

1. **A direct grouped-scale graph is CPU-selected.** Core ML's diagnostic concerns the
   tested convolution representation; it is not a ban on every four-bit format. [→](findings/coreml-grouped-scale-cpu/)
2. **A Core AI native LUT probe returns the wrong answer.** Its output matches a frozen
   flattened-scale error model. That supports a hypothesis, not an internal trace. [→](findings/coreai-flattened-scale/)
3. **Decomposition has a measured cost.** In a separate synthetic K512 ablation, sixteen
   K32 convolutions plus reduction take about four times as long. Similarity to the MLP
   gap does not establish a shared cause. [→](findings/split-decomposition-cost/)
4. **A8 did not show a stable speed advantage.** The historical 4K comparison differs by
   1.3%; the later native run is a separate observation, not a pooled estimate. [→](findings/ane-vs-gpu-prefill/)
5. **A candidate arithmetic model leaves residuals.** It misses 13 final Q8 outputs,
   with more differences before QDQ. One localized dot lies outside the binary16
   neighbours of its exact value: changing only the final rounding cannot explain it.
   Intermediate multiplication and accumulation remain unobserved. [→](findings/fp16-dot-residual/)

## Quick start

Choose the result to reproduce: the `ane-scope run` commands below execute synthetic
device suites. Recompute the G2 speed, memory, temperature and fan observations with
`python scripts/verify_g2.py`; G2 device replay still needs the research workspace and
model assets. [Commands by result](docs/REPRODUCING.md#run) ·
[current-source device validation](docs/VALIDATION.md#current-checkout-versus-the-measured-source).

```sh
uv sync --locked --extra apple
.venv/bin/ane-scope doctor
.venv/bin/ane-scope run --suite compatibility --output runs/my-groups
.venv/bin/ane-scope verify runs/my-groups
```

The `compatibility` suite reproduces the two toolchain findings. Python 3.12 and the Apple
SDK described in [REPRODUCING.md](docs/REPRODUCING.md); add `--offline` to `uv sync`
against a populated cache. `doctor` and `run` never fetch packages, models or data. Every
output directory must be new.

```sh
.venv/bin/ane-scope run --suite smoke      --output runs/my-smoke
.venv/bin/ane-scope run --suite throughput --output runs/my-throughput
.venv/bin/ane-scope run --suite split      --output runs/my-split
```

A completed compatibility suite can contain numerical failures and CPU-selected cases.
Those are the findings, not execution errors.

## Check the bundled evidence without a device

```sh
python results/historical/tests/verify_prefill.py     # the ANE/GPU ratios and the leak
python results/historical/tests/verify_native_followup.py  # native follow-up records
python results/historical/tests/verify_arithmetic.py  # the execution model and the dot product
python results/historical/tests/verify_historical.py  # imported timing records
python scripts/verify_g2.py                          # G2 event, thermal and resource accounting
python scripts/check_source_identity.py --check       # current source versus run identity
python scripts/summarize.py                           # registered tables and prose claims
python scripts/render_figures.py --check              # figure, source and generator identity
python -m unittest discover -s tests -v               # rounding, saturation, tampered evidence
```

Standard library plus NumPy, on any platform. `summarize.py` checks each named numeric
reference and its unit in both README openings, along with other registered values.
Unregistered prose remains outside these checks. These commands do not execute a device.

CI runs the checks above and also verifies figure regeneration:

```sh
python scripts/render_figures.py --write
git diff --exit-code -- docs/figures
```

The Git comparison requires tracked figure files. For an uncommitted draft, compare the
complete SVG inventory and bytes against a copy saved before editing.

## Repository map

| Path | Responsibility |
|---|---|
| `workarounds/` | The three things that do work, with their price and their boundary |
| `findings/` | One directory per finding: symptom, repro, evidence, and what is still a hypothesis |
| `results/historical/` | Imported records — historical comparisons, arithmetic evidence and the separate G2 service bundle |
| `results/fresh/` | Every measured call from the three device suites in this package |
| `src/ane_scope/_coreml.py`, `_coreai.py` | Export adapters and persisted graph/weight audits |
| `src/ane_scope/references/` | Deterministic fixtures and explicit arithmetic references |
| `src/ane_scope/native/` | Swift prediction hosts and observable runtime metadata |
| `src/ane_scope/controller.py`, `evidence.py`, `guard.py` | Admission, output/placement checks, resource limits |
| `scripts/summarize.py`, `render_figures.py` | Recompute the published numbers; regenerate the figures |
| `articles/` | Four research articles in English and Chinese |
| `docs/` | Scope, methods, provenance, reproduction, related work and open questions |
| `runs/` | Ignored: local fixtures, models, logs, compiled hosts and raw outputs |

## Corrections

- **(2026-09-11) The first G2 write-up said the ANE ran with lower fan speeds.**
  That held only at full load, where the GPU completed about 3.7 times the work. At equal
  load, the fans stayed at idle on both engines.
- **The G2 power capture failed.** The receiver hit its 1 GiB limit and the stream failed
  its clock-alignment rules, so all six equal-work energy comparisons are undetermined.
- **An animated screensaver ran during G2.** The Ventura dynamic screensaver loaded
  automatically and was noticed only after the run; all 775 background snapshots contain its
  process. There is no animation-off control, so its effect on either engine is unknown.
- **The prefill comparison is incomplete.** Three of nine planned timed processes finished
  before the run was stopped for memory safety. The ratios are a first observation, not a
  validated benchmark.
- **A 128-layer W8A8 control originally failed its reference at relative L2 0.233.** The
  frozen reference used ties-to-even where the device rounds ties away from zero. A new
  reference, new held-out inputs and the original thresholds were frozen first, then the
  experiment rerun. That correction became
  [the execution model](findings/execution-model/).
- **The strict cross-model byte gate did not pass.** E2B matched exactly; Qwen8B left 13
  residuals.
- **The historical split ratio was 4.44×; the fresh one is 3.96–4.10×.** Different runs,
  one process against three. Not pooled, and the difference is not attributed to any
  software change.
- **The first smoke attempt completed 12 of 14 cases.** Two Core AI group exports were
  rejected by the audit parser, which did not yet handle a hexadecimal constant and an
  open-ended slice. The parser was fixed and the cases rerun.
- **Core AI's requested compute unit was never per-operation proof.** An external code
  review pointed this out; the fields now record that limit.
- **Earlier figures declared a font most readers do not have**, so they rendered in a
  fallback serif everywhere except the machine that made them. Figures are now plain SVG
  text with a system font stack, and regeneration is byte-reproducible.

## What would change the answer

The next priority is complete-model prefill: the same model, precision and token input,
with attention and a KV cache, numerical checks first, then speed, mean power and joules
for the same completed work. Each engine runs at its own speed. A short pilot must
validate the power capture before any energy comparison. Replication on a second Apple
chip and hours of residency in one host would test whether the result transfers and lasts.

Further service experiments can test where the GPU's fans start rising between 4.9 and
23.7 requests/s, first covering 4.9 to about 6.4 requests/s, and whether the matrix-foreground
tail advantage survives an animation-off control, matched thermal starts and a CPU-only
busy baseline. [RESEARCH.md](docs/RESEARCH.md) · [RELATED_WORK.md](docs/RELATED_WORK.md)

## Contributing and licence

[CONTRIBUTING.md](CONTRIBUTING.md) · [LICENSE](LICENSE) · [NOTICE](NOTICE) ·
[code origins](docs/code-origins.json). Project code and generated experiment records are
MIT; dependencies keep their own licences, and no model weights, Apple framework binaries
or compiled models are redistributed. An independent project, with no Apple endorsement.

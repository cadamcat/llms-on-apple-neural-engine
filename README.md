# LLMs on Apple Neural Engine

English | [中文](README.zh-CN.md)

**How fast can a complete LLM run on the Apple Neural Engine, and what does it cost in energy? On one M5 Pro, Qwen3-4B FP16 runs on both ANE and GPU through Core AI, with an ANE graph sized to each input. The GPU is faster at every tested input from <!-- claim:g4a.all-contexts@g4a-001 -->500–16K<!-- /claim -->: <!-- claim:g4a.decode-gpu-faster@g4a-002 -->2.0–3.9×<!-- /claim --> in decode and <!-- claim:g4a.prefill-gpu-faster@g4a-003 -->2.9–15.9×<!-- /claim --> in prefill. For <!-- claim:g4a.short-contexts@g4a-004 -->500–2K<!-- /claim --> prefill, ANE uses <!-- claim:g6.two-run.short-prefill-energy-x@g6-001 -->0.68–0.81×<!-- /claim --> the GPU's component energy per input token across two runs; in decode it uses <!-- claim:g6.two-run.decode-energy-x@g6-002 -->0.86–1.18×<!-- /claim --> as much. Sizing the graph to the input is what keeps ANE usable on long inputs: against the earlier 256 / 2K / 32K graphs, ANE decode is <!-- claim:g4a.vs-g3.long-decode-speed@g4a-007 -->1.80–4.40×<!-- /claim --> faster from <!-- claim:g4a.n.2048@g4a-008 -->2K<!-- /claim --> and prefill <!-- claim:g4a.vs-g3.long-prefill-speed@g4a-009 -->2.76–6.40×<!-- /claim --> faster from <!-- claim:g4a.n.4096@g4a-010 -->4K<!-- /claim -->.**

<table>
<tr>
<td><b><!-- claim:g4a.decode.1024.ane-rate@g4a-011 -->14.56 token/s<!-- /claim --></b><br><sub>ANE decode from <!-- claim:g4a.n.1024@g4a-012 -->1K<!-- /claim --> KV; GPU <!-- claim:g4a.decode.1024.gpu-rate@g4a-013 -->30.09 token/s<!-- /claim --></sub></td>
<td><b><!-- claim:g4a.prefill.500.energy-x@g4a-014 -->0.69×<!-- /claim --></b><br><sub>ANE / GPU energy per input token at <!-- claim:g4a.n.500@g4a-015 -->500<!-- /claim --> prefill</sub></td>
<td><b><!-- claim:g4a.vs-g3.decode.2048.speed@g4a-016 -->4.40×<!-- /claim --></b><br><sub>ANE decode from <!-- claim:g4a.n.2048@g4a-017 -->2K<!-- /claim --> with a matched graph, against <!-- claim:g4a.g3.decode.2048.ane-rate@g4a-018 -->3.11 token/s<!-- /claim --> on the 32K graph</sub></td>
<td><b><!-- claim:g4a.prefill.1024.ane-tflops@g4a-019 -->6.5 TFLOP/s<!-- /claim --></b><br><sub>ANE prefill projection work at <!-- claim:g4a.n.1024@g4a-020 -->1K<!-- /claim --> input; GPU <!-- claim:g4a.prefill.1024.gpu-tflops@g4a-021 -->23.1 TFLOP/s<!-- /claim --></sub></td>
</tr>
</table>

**Start here:** [Complete-model results](findings/qwen3-4b-graph-capacity/) · [Should I use it?](#the-short-answer) · [What works](workarounds/) · [Findings](findings/) · [Reproduce](docs/REPRODUCING.md) · [Articles](articles/README.md)

![Qwen3-4B prefill and decode speed at six inputs: GPU, ANE with a graph sized to each input, and ANE on the earlier 256 / 2K / 32K graphs.](docs/figures/g4a-speed.svg)

Each path runs each input in its own host session: a warmup, <!-- claim:g4a.repetitions@g4a-048 -->3<!-- /claim --> teacher-forced requests of <!-- claim:g4a.decode-steps@g4a-049 -->256<!-- /claim --> decode steps, then a block of one-token prefill requests. Model loading, warmup and quiet periods are excluded. Speeds come from per-token clocks. The G3 comparison uses an equally long initial decode window for speed and energy.

At <!-- claim:g4a.n.1024@g4a-022 -->1K<!-- /claim --> input, prefill runs at <strong><!-- claim:g4a.prefill.1024.ane-rate@g4a-023 -->887.7 token/s<!-- /claim --> on ANE</strong> and <strong><!-- claim:g4a.prefill.1024.gpu-rate@g4a-024 -->3,177.1 token/s<!-- /claim --> on GPU</strong>: the GPU is <!-- claim:g4a.prefill.1024.gpu-faster@g4a-025 -->3.58×<!-- /claim --> faster. ANE slows more as the input grows, so at <!-- claim:g4a.n.16384@g4a-026 -->16K<!-- /claim --> the GPU is <!-- claim:g4a.prefill.16384.gpu-faster@g4a-027 -->15.94×<!-- /claim --> faster in prefill and <!-- claim:g4a.decode.16384.gpu-faster@g4a-028 -->3.87×<!-- /claim --> in decode. The dashed line is the earlier run, whose ANE path chose among 256 / 2K / 32K graphs: decode from <!-- claim:g4a.n.2048@g4a-029 -->2K<!-- /claim --> ran at <!-- claim:g4a.g3.decode.2048.ane-rate@g4a-030 -->3.11 token/s<!-- /claim --> there and <!-- claim:g4a.decode.2048.ane-rate@g4a-031 -->13.70 token/s<!-- /claim --> here. Both runs used the same weights and inputs; G4 A adapted the G3 host for fixed capacities. Their GPU rates agree within <!-- claim:g4a.vs-g3.gpu-speed@g4a-032 -->0.97–1.02×<!-- /claim -->. Timing includes the host and framework; it measures these implementations, not the hardware ceiling.

![Implied prefill TFLOP/s and modelled decode read GB/s at six inputs on both paths, split into projection and attention work, and weight and KV reads.](docs/figures/g4a-implied.svg)

The token rates can be expressed as model work. Each prefill token passes <!-- claim:g4a.projection-parameters@g4a-033 -->3.63 billion<!-- /claim --> projection weights; counting causal attention too, the GPU sustains <!-- claim:g4a.gpu-total-tflops-span@g4a-034 -->20.5–27.2 TFLOP/s<!-- /claim --> across the inputs, while ANE falls through <!-- claim:g4a.ane-total-tflops-span@g4a-035 -->1.7–7.0 TFLOP/s<!-- /claim --> as the input grows. Assuming one read of <!-- claim:g4a.weight-bytes@g4a-036 -->8.04 GB<!-- /claim --> of FP16 weights and the existing KV per decode step, the GPU's modelled read rate stays within <!-- claim:g4a.gpu-read-span@g4a-037 -->227–250 GB/s<!-- /claim --> from <!-- claim:g4a.n.500@g4a-038 -->500<!-- /claim --> to <!-- claim:g4a.n.16384@g4a-039 -->16K<!-- /claim -->, consistent with bandwidth-dominated decode. ANE's falls through <!-- claim:g4a.ane-read-span@g4a-040 -->59–122 GB/s<!-- /claim -->, more than the extra KV reads explain; attention compute, fixed-graph work and host overhead have not been separated. These are model-based rates: device counters and memory traffic were not measured, and the byte model excludes padding and current-token writes.

![Component energy per token by CPU, GPU and ANE counter at six inputs, GPU path and ANE path side by side.](docs/figures/g4a-energy.svg)

Across G4 A and G6, <!-- claim:g6.short-contexts@g6-005 -->500–2K<!-- /claim --> prefill on ANE uses <!-- claim:g6.two-run.short-prefill-energy-x@g6-006 -->0.68–0.81×<!-- /claim --> the GPU's component energy per input token; <!-- claim:g6.long-contexts@g6-007 -->4K–16K<!-- /claim --> prefill spans <!-- claim:g6.two-run.long-prefill-energy-x@g6-008 -->0.90–1.33×<!-- /claim --> and decode <!-- claim:g6.two-run.decode-energy-x@g6-009 -->0.86–1.18×<!-- /claim -->. These ranges combine the tested input lengths and two runs; the chart shows G4 A. [Paired results](findings/qwen3-4b-graph-capacity/#repeat-and-decode-query-width) show the change at each input.

In the plotted G4 A run, during ANE-path decode, <!-- claim:g4a.ane-arm-gpu-decode-share@g4a-046 -->32–58%<!-- /claim --> of the component energy is recorded by the system GPU counter; its process and operation sources are unassigned. These are CPU + GPU + ANE software energy estimates without idle subtraction, admitted block by block; the CPU counter was raised by other activity during the ANE <!-- claim:g4a.n.4096@g4a-047 -->4K<!-- /claim --> blocks. The † marks elevated CPU power relative to neighbouring inputs. Whiskers describe sample-timing bounds, not sensor accuracy; these are not wall-power measurements.

[Method and scope](docs/SCOPE.md#g4-a-matched-graph-observations) · [Repeat and query width](findings/qwen3-4b-graph-capacity/#repeat-and-decode-query-width) · [Detailed article](articles/06-qwen3-4b-matched-graphs.md) · [The earlier 256 / 2K / 32K results](findings/qwen3-4b-prefill-decode/)

## Measured on

| Machine | Memory | OS | Xcode | Toolchains |
|---|---|---|---|---|
| **Apple M5 Pro** | 48 GiB | macOS 27.0 | 27.0, 26.6 | Core AI · MLX 0.32.2 · coremltools 9.0 · coreai-torch 0.4.1, 0.4.2 |

Every round records the versions it ran with; [SCOPE.md](docs/SCOPE.md) lists them by round.

## The short answer

| If your model is… | On this toolchain, today |
|---|---|
| **Complete Qwen3-4B FP16** | [GPU is faster at every tested input](findings/qwen3-4b-graph-capacity/), even with an ANE graph sized to each input. ANE saves component energy on short prefill; decode energy per token is close. On a [graph ladder that jumps to 32K](findings/qwen3-4b-prefill-decode/), long inputs are much slower and costlier on ANE |
| **The tested Qwen3 static-shape exports with added one- or two-position decode functions** | [The ANE request fails](findings/ane-short-decode-query/) with `0xe00002c2` and the host aborts; four- and eight-position controls run. Four positions instead of eight left Qwen3-4B decode speed at <!-- claim:g6.q4-q8-speed@g6-014 -->1.001–1.008×<!-- /claim --> |
| **Several ANE model loads in a row** | [The ANE compiler service can keep each deleted compile input open](findings/ane-compiler-service-disk/): one run held <!-- claim:g4a.disk.held@g4a-050 -->178.4 GiB<!-- /claim --> until the service exited. [Reclaim it](workarounds/#4-reclaim-disk-space-held-by-the-ane-compiler-service) |
| **Qwen3-4B with the iOS 4-bit palettized preset** through Core AI, ANE preferred | [The ANE compile fails and the model runs on the GPU](findings/coreai-palettized-weights-gpu/), with no error to the host; its checked first output passes the CPU reference built from the exported codes: <!-- claim:g6.w4.decode.1024.rate@g6-015 -->2.87 token/s<!-- /claim --> decode from <!-- claim:g6.n.1024@g6-016 -->1K<!-- /claim -->, while FP16 on ANE is <!-- claim:g6.w4.fp16-faster-decode@g6-017 -->5.1–8.1×<!-- /claim --> faster |
| **The tested direct signed-INT4 Core ML K64 graph**, with two K32 scales per row | Numerically correct, CPU-selected. A related historical graph reports `ANE only support per-cout/per-tensor quantization`; this is not a test of every Q4 format or representation |
| **K32 decomposition** into per-output-channel scales | Core AI passes the small compatibility probe. A separate synthetic ablation is [about 4× slower](findings/split-decomposition-cost/); that is not a universal cost |
| **Grouped 4-bit through Core AI's native LUT path** | Accepted, shows real ANE activity, and [returns the wrong answer](findings/coreai-flattened-scale/) — 1921 of 4096 values, predictably |
| **W4A16 served from the native ANE host** | [About a quarter of the GPU's speed](findings/w4a16-service-tradeoffs/). At equal load the fans stay at idle on both engines; a matrix foreground shows a smaller tail penalty beside ANE inference. Energy undetermined |
| **Resident execution** | The historical Python gate retained 1,966,080 bytes per call. Two native G2 hosts ran 33,728 stage calls each and ended 60–62 MiB smaller; [indefinite residency is untested](findings/iosurface-per-call-growth/) |
| **W4A16 versus A8W4** | Depends on positions per call. One real E4B gate through Core AI runs A8W4 at <!-- claim:g7.coreai.gate.64.a8w4-over-w4a16.speed@g7-101 -->0.976–0.990×<!-- /claim --> W4A16 speed at 64 positions and [<!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.speed@g7-102 -->1.359–1.360×<!-- /claim --> at 1024](findings/quantized-speedup-conditions/#positions-per-call-on-a-real-projection). The historical 4K pair, looped over 64-position calls, [ran at 1.013×](findings/ane-vs-gpu-prefill/) |
| **The tested W8A8 synthetic control** | Accelerates — Core AI is **1.86–1.87×** over its FP16 baseline on the controlled 128-layer chain. [The gain depends on work per call and on the weights](findings/quantized-speedup-conditions/): A8W4 runs at <!-- claim:g1w.depth.both-1.speed@g1w-001 -->0.86×<!-- /claim --> W4A16 speed as one layer and <!-- claim:g1w.depth.both-128.speed@g1w-002 -->1.33×<!-- /claim --> as 128; exact zero weights make FP16 itself <!-- claim:g1w.density.old.speed@g1w-003 -->1.88×<!-- /claim --> faster |
| **Gemma 4 E4B mobile QAT A8W4**, first MLP | [Wrong, and not faster](findings/quantized-speedup-conditions/#a-released-qat-checkpoint). A QDQ multiply on the Neural Engine [uses another QDQ's scale](findings/coreai-qdq-multiply-scale/), through Core ML as well as Core AI: <!-- claim:g1w.e4b.1.native.l2@g1w-004 -->339%<!-- /claim --> off in one MLP. With a product clamp, eight repeated MLPs are <!-- claim:g1w.e4b.8.clip-product.l2@g1w-005 -->21.6%<!-- /claim --> off and run at <!-- claim:g1w.e4b.clip-product.speed@g1w-006 -->0.976×<!-- /claim --> W4A16 speed |
| **Core ML instead of Core AI**, same E4B codes on ANE | [Similar speeds for FP16 and gate W8A8](findings/coreml-coreai-same-codes/) (<!-- claim:g7.ml-over-ai.fp16.speed@g7-103 -->0.979–0.998×<!-- /claim --> and <!-- claim:g7.ml-over-ai.w8a8.speed@g7-104 -->0.962–0.967×<!-- /claim --> of Core AI). Core ML's four-bit palette graphs run at <!-- claim:g7.ml-over-ai.four-bit.speed@g7-105 -->0.065–0.280×<!-- /claim --> Core AI's speed, slower than Core ML's own FP16 |

Earlier rounds stay separate records. The historical Python MLP measured the GPU **2.87×**
faster at 64 positions, **5.09×** at 1024 and **4.41×** at 4096, with one timed process per
path and an early stop; native C256 later measured 582.19 ms against the GPU's 152.25 ms at
4096 positions, a native `run` time that still includes runtime and synchronization.
[Python round](findings/ane-vs-gpu-prefill/) ·
[native follow-up](results/historical/native-mlp-followup.json). Limits for each number
are in [SCOPE.md](docs/SCOPE.md).

## What is in here

The complete-model comparison and the component experiments have separate evidence:

| Part | What it is |
|---|---|
| 🔧 **Four things that do work** | The rewrite that gets grouped 4-bit onto the accelerator at all, the graph expressions that fix a QDQ multiply, the quantization scheme that accelerates on a deep enough chain, and reclaiming disk space the ANE compiler service keeps — each with its price and its boundary written down. [workarounds/](workarounds/) |
| 📊 **Complete-model measurements** | [G4 A](findings/qwen3-4b-graph-capacity/) measures Qwen3-4B FP16 prefill, decode and component energy with an ANE graph sized to each of six inputs, and G6 repeats every arm and halves the decode query width; [G3](findings/qwen3-4b-prefill-decode/) measures the same model on a 256 / 2K / 32K graph ladder. |
| 📊 **Component comparisons** | [G2](findings/w4a16-service-tradeoffs/) measures seven-size native W4A16 speed, native host memory, equal-rate and saturated temperature and fan response, and GPU foreground tails. The [earlier A8W4/W4A16/GPU comparison](findings/ane-vs-gpu-prefill/) retains its numerical controls, per-PID evidence and early stop. [G7](findings/coreml-coreai-same-codes/) runs the same E4B codes through Core ML and Core AI. Separate rounds, not pooled estimates. |
| 🐛 **Reproducible defects** | Three toolchain failures with minimal reproductions, expected wrong outputs and matched negative controls, including a QDQ multiply that takes another QDQ's scale on the Neural Engine in both Core AI and Core ML; a compile failure that silently moves a whole graph to the GPU; one memory leak with four failed mitigations and an external corroboration; a system compiler service that keeps deleted compile inputs open; a decode query shape that ANE rejects at one and two positions; a palettized 4-bit preset whose ANE compile fails without an error to the host; one structural cost measured in three paired process rounds. [findings/](findings/) |
| 🔬 **An arithmetic model** | A candidate arithmetic model checked against 7,163,904 final Q8 gate outputs from two real models, with 13 Q8 mismatches — and one localized 32-term dot product it cannot explain, checkable from published scalars with no Apple hardware. [The model](findings/execution-model/) · [the residual](findings/fp16-dot-residual/) |

## Who this is for

- 🟡 **You want to move inference off the GPU and are deciding whether it is worth it.**
  → [the short answer](#the-short-answer), then
  [the complete-model comparison](findings/qwen3-4b-graph-capacity/).
- 🔴 **Your grouped-quantized model compiles but runs on the CPU, or returns garbage.**
  → [what to do about it](workarounds/), then
  [Core ML's constraint](findings/coreml-grouped-scale-cpu/) and
  [Core AI's scale bug](findings/coreai-flattened-scale/) for why.
- 🟠 **Your long-running ANE process keeps growing.**
  → the historical Python binding retained [1.875 MiB per call](findings/iosurface-per-call-growth/),
  with what did not fix it and who else has reported it; two native Swift hosts ran 33,728
  stage calls each without that growth.
- 🟠 **Your disk fills up while you run ANE models, and `du` cannot find the files.**
  → [the compiler service that holds deleted inputs](findings/ane-compiler-service-disk/) and
  [how to reclaim the space](workarounds/#4-reclaim-disk-space-held-by-the-ane-compiler-service).
- 🔬 **You are debugging quantized numerics and cannot tell a rounding difference from a
  bug.** → [the execution model](findings/execution-model/),
  [the residual](findings/fp16-dot-residual/) and
  [small attention products](findings/attention-product-precision/).
- 🧪 **You want to reproduce or refute this.** → [REPRODUCING.md](docs/REPRODUCING.md).
  The bundled records support offline checks of the registered claims; full historical
  array replay still needs the original assets.

**Status.** A research artifact, not a supported product. No external replication
yet; G6 repeated every G4 A arm in new host sessions on the same machine. G3, G4 A and G6 add complete-model speed and software component energy; broad model-quality evaluation remains open. Three conditions qualify G2: an animated
screensaver was running, found after the run; five of six coexistence groups had unmatched
thermal starts; and the power capture failed, so energy is undetermined.
[G2 scope](docs/SCOPE.md#g2-service-observations) · [RESEARCH.md](docs/RESEARCH.md)

## Earlier component measurements (G2)

G2 measures one native W4A16 MLP, with MLX as its GPU baseline. Its figures below use component positions, not model tokens.

<table>
<tr>
<td><b><!-- claim:g2.ane-share@speed-card -->0.248×<!-- /claim --></b><br><sub>ANE / GPU speed at 1024 positions; three hosts each</sub></td>
<td><b><!-- claim:g2.native-calls@calls-card -->33,728 calls<!-- /claim --></b><br><sub>per native ANE host, ending <!-- claim:g2.native-shrink@memory-card -->60–62 MiB<!-- /claim --> smaller</sub></td>
<td><b><!-- claim:g2.temperature-gap@temperature-card -->1.3–3.3 °C<!-- /claim --></b><br><sub>warmer GPU sensor under GPU inference at equal load; fans idle on both</sub></td>
<td><b><!-- claim:arithmetic.q8-summary@arithmetic-card -->13 / 7,163,904<!-- /claim --></b><br><sub>final Q8 outputs a candidate arithmetic model does not match</sub></td>
</tr>
</table>

![G2 first-layer MLP: all three independent host curves at seven position counts, with per-engine medians. The ANE stays near 6,760 positions per second; the GPU runs between about 18,500 and 31,600.](docs/figures/g2-throughput.svg)

At 1024 positions the ANE path processes about **<!-- claim:g2.ane-rate@ane-rate -->6,760 positions/s<!-- /claim -->** and MLX on the GPU
about **<!-- claim:g2.gpu-rate@gpu-rate -->27,300 positions/s<!-- /claim -->**, the median of three independent hosts per engine. Above 256 positions the
ANE line is flat because every request is split into fixed 256-position tiles; that
describes this pipeline, not the hardware's ceiling. Sustained service puts the ANE at
**<!-- claim:g2.service-share@service-share -->0.270×<!-- /claim -->** the GPU speed: the same few milliseconds of per-request checking and logging
weigh more on the faster engine.

![Completed requests per second against GPU sensor temperature and fan 0 speed. At equal load both engines stay at idle fan speed; only the saturated GPU blocks raise it.](docs/figures/g2-load-fans.svg)

At three equal arrival rates, up to **<!-- claim:g2.equal-rate-max@equal-rate -->4.9 requests/s<!-- /claim -->**, fans stayed at their idle speed of
about **<!-- claim:g2.fan-idle@fan-idle -->1,350 RPM<!-- /claim -->** on both engines, while the GPU sensor read **<!-- claim:g2.temperature-gap@temperature-detail -->1.3–3.3 °C<!-- /claim -->** warmer when
the GPU did the work. At full load the ANE, completing **<!-- claim:g2.ane-saturated@ane-saturated -->6.4 requests/s<!-- /claim -->**, still left the
fans at idle; the GPU completed **<!-- claim:g2.gpu-saturated@gpu-saturated -->23.7 requests/s<!-- /claim -->** and ran fan 0 at **<!-- claim:g2.fan-saturated@fan-saturated -->3,100–3,500 RPM<!-- /claim -->**. GPU loads in
between were not measured, so the point where its fans start rising is open. Energy is
undetermined: the power capture failed its integrity and clock checks.

## Constraints in the quantized component

The following constraints are measured. Their contribution to the real MLP's GPU gap
has **not** been isolated:

1. **A direct grouped-scale graph is CPU-selected.** Core ML's diagnostic concerns the
   tested convolution representation; it is not a ban on every four-bit format. [→](findings/coreml-grouped-scale-cpu/)
2. **A Core AI native LUT probe returns the wrong answer.** Its output matches a frozen
   flattened-scale error model. That supports a hypothesis, not an internal trace. [→](findings/coreai-flattened-scale/)
3. **Decomposition has a measured cost.** In a separate synthetic K512 ablation, sixteen
   K32 convolutions plus reduction take about four times as long. Similarity to the MLP
   gap does not establish a shared cause. [→](findings/split-decomposition-cost/)
4. **A8's speed gain depends on the work in each call.** The historical 4K A8W4 path, looped over
   64-position calls, runs at 1.013× W4A16 speed; the later native run is a separate observation, not a pooled estimate. [→](findings/ane-vs-gpu-prefill/)
   Eight repeated Gemma 4 E4B QAT MLPs run at <!-- claim:g1w.e4b.native.speed@g1w-007 -->0.977×<!-- /claim --> W4A16 speed, and synthetic chains gain
   only with depth: <!-- claim:g1w.depth.both-1.speed@g1w-008 -->0.86×<!-- /claim --> at one layer, <!-- claim:g1w.depth.both-128.speed@g1w-009 -->1.33×<!-- /claim --> at 128. One real E4B gate through Core AI gains
   at 1024 positions, <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.speed@g7-106 -->1.359–1.360×<!-- /claim -->, and not at 64. [→](findings/quantized-speedup-conditions/)
5. **A candidate arithmetic model leaves residuals.** It misses 13 final Q8 outputs,
   with more differences before QDQ. One localized dot lies outside the binary16
   neighbours of its exact value: changing only the final rounding cannot explain it.
   Intermediate multiplication and accumulation remain unobserved. [→](findings/fp16-dot-residual/)
6. **A QDQ multiply on the Neural Engine dequantizes with another QDQ's scale.** A model-free probe returns 2, 4
   and 8 where the answer is 1, through Core AI and through Core ML, and a released QAT MLP is <!-- claim:g1w.e4b.1.native.l2@g1w-010 -->339%<!-- /claim --> off. A product clamp that leaves
   the true value unchanged avoids it. On the CPU, Core ML returns the right values. [→](findings/coreai-qdq-multiply-scale/)

## Quick start

Choose the result to reproduce: the `ane-scope run` commands below execute synthetic
device suites. Recompute the complete-model results with `python scripts/verify_g4a.py`, `python scripts/verify_g6.py` and `python scripts/verify_g3.py`, and G2 speed, memory, temperature and fan observations with
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
python scripts/verify_g4a.py                         # G4 A matched-graph speed and component energy
python scripts/verify_g6.py                          # G6 decode query width, W4 and the G4 A repeat
python scripts/verify_g7.py                          # G7 Core ML and Core AI on the same E4B codes
python scripts/verify_ane_compiler_disk.py           # disk space held by the ANE compiler service
python findings/ane-short-decode-query/repro/verify.py  # one- and two-position decode queries
python findings/coreai-palettized-weights-gpu/repro/verify.py  # palettized preset placement
python scripts/verify_g3.py                          # G3 complete-model speed and component energy
python scripts/verify_g2.py                          # G2 event, thermal and resource accounting
python scripts/verify_g1w.py                         # quantized speed conditions and the E4B QAT MLP
python scripts/verify_g5.py                          # attention precision and chunking timings
python findings/coreai-qdq-multiply-scale/repro/verify.py  # the QDQ multiply outputs
python findings/coreai-qdq-multiply-scale/repro/coreml/verify.py  # the same probe through Core ML
python scripts/check_source_identity.py --check       # current source versus run identity
python scripts/summarize.py                           # registered tables and prose claims
python scripts/render_figures.py --check              # figures regenerate byte-identically
python -m unittest discover -s tests -v               # rounding, saturation, tampered evidence
```

Standard library plus NumPy, on any platform. `summarize.py` checks each named numeric
reference and its unit in both README openings, along with other registered values.
Unregistered prose remains outside these checks. These commands do not execute a device,
and CI runs all of them.

## Repository map

| Path | Responsibility |
|---|---|
| `workarounds/` | The four things that do work, with their price and their boundary |
| `findings/` | One directory per finding: symptom, repro, evidence, and what is still a hypothesis |
| `results/historical/` | Imported records — historical comparisons, arithmetic evidence and separate G2 service / G3, G4 A and G6 complete-model / G7 runtime bundles |
| `results/fresh/` | Every measured call from the three device suites in this package |
| `src/ane_scope/_coreml.py`, `_coreai.py` | Export adapters and persisted graph/weight audits |
| `src/ane_scope/references/` | Deterministic fixtures and explicit arithmetic references |
| `src/ane_scope/native/` | Swift prediction hosts and observable runtime metadata |
| `src/ane_scope/controller.py`, `evidence.py`, `guard.py` | Admission, output/placement checks, resource limits |
| `scripts/summarize.py`, `render_figures.py` | Recompute the published numbers; regenerate the figures |
| `articles/` | Research articles in English and Chinese |
| `docs/` | Scope, methods, provenance, reproduction, related work and open questions |
| `runs/` | Ignored: local fixtures, models, logs, compiled hosts and raw outputs |

## Corrections

- **(2026-09-15) The QDQ multiply defect was attributed to Core AI.** The finding was titled
  "A Core AI QDQ multiply dequantizes with another QDQ's scale". The same graph built with
  coremltools returns the same wrong values through Core ML on the Neural Engine and correct
  values on the CPU, so the [finding](findings/coreai-qdq-multiply-scale/) now names the Neural Engine path.
- **(2026-09-15) A single gate was called the wrong place to look for an A8 gain.** The
  quantized speed-up finding said so from a 512-channel synthetic chain and named depth in its
  title. A real E4B gate at 1024 positions gains; the [finding](findings/quantized-speedup-conditions/)
  now names work per call.
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

G4 A answered the question G3 left open: a graph sized to the input removes most of ANE's long-input penalty, but not the GPU's lead. G6 then changed one thing at a time on the same graphs. A four-position decode query instead of eight ran at <!-- claim:g6.q4-q8-speed@g6-010 -->1.001–1.008×<!-- /claim --> the speed, so narrowing the logical query brought little speed benefit on these graphs; the cost of compiled padding and the benefit of narrower queries remain unmeasured; a one- or two-position query [does not execute](findings/ane-short-decode-query/). The upstream iOS 4-bit palettized preset ran entirely on the GPU, at <!-- claim:g6.w4.decode.1024.rate@g6-011 -->2.87 token/s<!-- /claim --> for <!-- claim:g6.n.1024@g6-012 -->1K<!-- /claim --> decode, so decode with 4-bit weights on ANE is still unmeasured; on one E4B projection and MLP, Core AI's four-bit palette weights on ANE ran [<!-- claim:g7.coreai.w4a16-over-fp16@g7-107 -->1.465–3.175×<!-- /claim --> faster than FP16](findings/coreml-coreai-same-codes/). The GPU energy on the ANE path stays at <!-- claim:g6.decode-gpu-energy@g6-013 -->0.257–0.265 J/token<!-- /claim --> with either query width; its source is open. Replication on a second Apple chip, hours of residency in one host and a broader quality evaluation would test whether the result transfers and lasts.

Further service experiments can test where the GPU's fans start rising between 4.9 and
23.7 requests/s, first covering 4.9 to about 6.4 requests/s, and whether the matrix-foreground
tail advantage survives an animation-off control, matched thermal starts and a CPU-only
busy baseline. [RESEARCH.md](docs/RESEARCH.md) · [RELATED_WORK.md](docs/RELATED_WORK.md)

## Contributing and licence

[CONTRIBUTING.md](CONTRIBUTING.md) · [LICENSE](LICENSE) · [NOTICE](NOTICE) ·
[code origins](docs/code-origins.json). Project code and generated experiment records are
MIT; dependencies keep their own licences, and no model weights, Apple framework binaries
or compiled models are redistributed. An independent project, with no Apple endorsement.

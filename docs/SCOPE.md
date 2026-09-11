# Scope: what these numbers mean, and what they do not

This is the one place where the boundaries are stated in full. Other documents
link here.

## What is measured

A synthetic convolution chain, exported through two Apple runtimes, executed by
a small Swift host, and timed around one synchronous prediction. Weights come
from a fixed seed; no model is downloaded and none is included. Every timed
configuration first passes numerical controls, a persisted-asset audit and
observed ANE participation.

Two kinds of evidence sit side by side and must not be confused. The
**synthetic suites** in this package are model-free and timed. The **imported
records** in `results/historical/` come from a closed research workspace and
include gate outputs from two real models — no weights or activations are
published, only derived scalars: counts, residual coordinates, the 32 terms of
one dot product, and a bisection ladder. Those are labelled
`historical_import: true`, and a fresh run never overwrites them.

The arithmetic findings rest on the imported records. The **controlled**
performance findings — the 128-layer positive control and the wide/split
ablation — rest on the synthetic suites in this package. The **ANE-versus-GPU
comparison** is a third thing again: an imported first observation from one
timed process per path, on a Python, 64-position chunked implementation, whose
1024- and 4096-position cases loop saved activations rather than running an
end-to-end prefill. A later [native MLP follow-up](../results/historical/native-mlp-followup.json)
adds separate HOST / PIO / TILE observations, including bounded C64/C256 memory
passes. Historical Python and native timing rounds are not pooled. None of these
experiments validates the others.

## The throughput unit

Source-equivalent operations per second: `2 × depth × 512² × 4096` divided by a
process's p50 prediction time. A MAC counts as two operations. The numerator
counts the source convolutions only — QDQ, slice and reduction operations cost
real time inside the measured interval but add nothing to it, so a graph with
more operations is not credited with more work.

This unit answers one question: *how fast does this synthetic graph finish?* It
is not a physical instruction count and it does not convert into model tokens
per second.

## What the fresh synthetic timing evidence establishes

- The saved asset really contains the representation it claims: weights,
  scales and zero points are decoded independently from the persisted file,
  not read back out of the in-memory program.
- The output matches a reference frozen before the run, under stated
  intermediate precision and rounding rules.
- ANE requests occurred inside every control window of the timed process, and
  for Core ML every audited convolution preferred the Neural Engine.
- The measured latency was produced with those checks already passed, in three
  independent processes per arm.

## What it does not establish

These limits distinguish the evidence sets rather than assigning one protocol to all of them.

1. **No physical INT8.** A quantized representation and an observed ANE request
   do not prove that integer hardware instructions ran. Nothing here observes
   instruction mix or utilisation.
2. **No exclusive placement.** Core AI exposes no per-operation mapping. The
   evidence is control-window participation plus the requested specialization,
   which is weaker than a placement trace.
3. **No model quality claim.** FP16, W8A8 and A8W4 start from identical weight
   values but are not quality-equivalent models. Task quality needs NLL, KL or
   a downstream evaluation; the synthetic suites do not provide it. Imported gate
   agreement is also not a task-quality result.
4. **The synthetic suites are not distributional benchmarks.** Their weights are Hadamard, sign-flipped and
   permuted; each code is ±1 times a shared scale, and 16 distinct spatial
   vectors are repeated across 4096 positions. That keeps a reference tractable
   and is deliberately unlike a real activation distribution.
5. **No accepted energy result.** G2 adds finite component thermal and
   coexistence observations, with the conditions below. Its failed power capture
   does not support joules or an energy-efficiency ranking.
6. **One host, distinct run identities.** Apple M5 Pro. The fresh synthetic suites
   use macOS 27.0 build 26A428, Xcode/SDK 27.0, coremltools 9.0, coreai-torch
   0.4.1 and coreai-core 1.0.0b2. The historical Python MLP record used Xcode
   26.6 / macOS SDK 26.5; its SDK blocker is not the native follow-up's status.
   G2 ran on build 26A428 with Xcode 27.0, and its GPU host on MLX 0.32.2. These
   come from same-day preflight records and a post-run statement that no software
   changed that day.
   Version and source identities belong to each run, not to a shared headline.
   No independent host replication is included.

Two more apply to the arithmetic findings specifically:

7. **A matching output is not an implementation.** A written-down model that
   reproduces the device byte-exact is consistent with the hardware doing those
   steps. It does not show that it does. The same goes for an error model that
   predicts a wrong answer exactly.
8. **Gate outputs, not models.** The cross-model validation covers one gate's
   outputs. The 13/7,163,904 mismatch count is after output QDQ; before it,
   E2B has 16 differences and Qwen8B 1,194. It says nothing about full-model correctness, perplexity or any
   downstream task.

And two to the historical Python ANE-versus-GPU comparison:

9. **Implementations, not a ceiling.** The initial latencies describe the Python
   path. A native host subsequently ran, and C64/C256 passed short memory gates.
   Native `run` includes framework, scheduling and synchronization, so it still
   does not isolate pure device cost. Long-term stability remains unmeasured.
10. **A first observation, not a benchmark.** Three of nine planned timed
    processes completed before the run stopped for memory safety, and the GPU
    side is one backend, one kernel and one version.

## How to read a spread

For the fresh throughput and split suites: three independent processes per arm,
30 measured calls in each. Reported ranges
are the minimum and maximum across the three process medians. They are **not**
confidence intervals: 30 calls inside one process are correlated observations,
not 30 replications, and OS scheduling, thermal state and background activity
are not fully controlled.

## Where each kind of evidence lives

| Question | Where to look |
|---|---|
| What was measured, and how | [METHODS.md](METHODS.md) |
| The current numbers | [MEASUREMENTS.md](MEASUREMENTS.md) |
| Which cases fail, and how they fail | [findings/](../findings/) |
| How to reproduce or recompute | [REPRODUCING.md](REPRODUCING.md) |
| Where imported records came from | [PROVENANCE.md](PROVENANCE.md) · [HISTORICAL.md](HISTORICAL.md) |
| What was actually run locally | [VALIDATION.md](VALIDATION.md) |
| What is still open | [RESEARCH.md](RESEARCH.md) |

Incorrect outputs, CPU-selected cases and incomplete evidence are recorded as
observations without a benchmark time.

## G2 service observations

G2 is a separate imported native W4A16 experiment on one M5 Pro. It uses Gemma
4 12B's first-layer full MLP (H3840/I15360), same-source Q4_0 weights and FP16
I/O. Three saved inputs are cycled; some shapes repeat saved activations. N is
MLP positions, not LLM context or tokens. Attention, KV cache, layer-to-layer
transfer, full-model quality and application deployment are outside this run.

Each backend has its own frozen numerical reference. Same source weights do not
require cross-backend byte identity. Actual ANE participation was audited in
same-target-PID control windows; the portable bundle retains that audit's
metadata, not a new device replay or physical instruction counter.

- **Two timing denominators.** P2 rates use N × 30 divided by accumulated client
  call time, then a median across three hosts. Saturated service uses completions
  strictly inside the observation window divided by window duration. Hashing,
  checks and service orchestration remain in the latter; no overhead is deducted.
- **Finite residency.** Multiple fresh hosts complete separate blocks. Neither
  the original eight-hour protocol budget nor a sampled memory pass means one
  complete model remained resident all night. Initial swap was nonzero; no
  observed growth is the reported result. The two native ANE P0 hosts ended
  smaller than they began; that shows the historical Python per-call growth did
  not appear in those runs, not that it cannot.
- **Thermal observations.** CPU/GPU sensor means are not case or ANE junction
  temperature; RPM is not acoustic noise. Saturated blocks complete unequal
  work. Equal-rate means cover six minutes, and most blocks do not reach the
  recorded platform rule. They are not equilibrium temperature differences.
  Fan speeds did not differ at equal load; saturated fan differences come with
  unequal work, and GPU loads between the highest equal rate and GPU saturation
  were not measured.
- **Coexistence.** Only one of six starting groups matches. Results include all eighteen slots and the unmatched
  groups; the memory-access alone baselines are anomalous. In all
  four matrix groups the ANE-condition p95 is slightly below the alone baseline,
  so that baseline is not a zero-interference reference. Tails
  describe completed jobs; violations include every arrival. A fixed-rate
  throughput-retention ratio is not maximum spare GPU capacity or a GUI frame rate.
- **Environment.** Normal personal-PC system background is accepted and recorded.
  An unintended Ventura dynamic screensaver loaded during the run and was noticed after completion;
  all 775 r4 background snapshots contained its process. That does not establish
  continuous rendering in every slot or per-process GPU overhead. No animation-off
  control exists. Effects on the two backends need not cancel.
- **Power.** Capacity truncation and strict clock-alignment failures leave all six
  equal-work energy pairs undetermined. Power sampling ended during the first
  0.75R GPU block, changing observer overhead. **Correction (2026-09-11):** earlier
  text attributed implausible system-power spikes to SMC. In the recorded
  [macmon source](https://github.com/vladkens/macmon/blob/6919d7781b6c55a6e3bedff83a210435837e1dfe/src_lib/metrics.rs#L561-L564),
  a successful SMC read produces `sys_power = max(PSTR, CPU + GPU + ANE)`.
  Component-power spikes can therefore enter this column. PSTR was not recorded
  separately, so these readings cannot identify an SMC fault or establish energy
  use. The `SMC_over_1000W` field in the imported `expected-summary.json` retains
  the original label for these mixed readings; its name is not a sensor diagnosis.

The public records support scalar recomputation; the original 87-check device
and asset audit remains imported evidence. Repeating it or the device protocol
requires additional workspace assets. See [reproduction](REPRODUCING.md#g2-recomputation-and-device-replay).

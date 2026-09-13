# Scope: what these numbers mean, and what they do not

This is the one place where the boundaries are stated in full. Other documents
link here.

## What is measured

The fresh device suites measure a synthetic convolution chain, exported through two Apple runtimes, executed by
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
   and is deliberately unlike a real activation distribution. The measured rate
   and the paired quantized ratio belong to this fixture as much as to its
   format: at the same shapes, 73.63% exact-zero weights make FP16 1.88× faster
   and cut the A8W4 gain over FP16 to 1.36×, and a single layer runs slower.
   [Quantized speed-up conditions](../findings/quantized-speedup-conditions/).
5. **G2 has no accepted energy result.** G2 adds finite component thermal and
   coexistence observations, with the conditions below. Its failed power capture
   does not support joules or an energy-efficiency ranking. G3 has separate software component energy estimates, described below.
6. **One host, distinct run identities.** Apple M5 Pro, 48 GiB, macOS 27.0. Each
   round ran with its own versions:

   | Round | macOS build | Xcode | Other |
   |---|---|---|---|
   | G3 complete model | 26A428 | 27.0 (27A266a) | Core AI on both paths; coreai-torch 0.4.2 for the export |
   | G2 component service | 26A428 | 27.0 | MLX 0.32.2 on the GPU host |
   | Fresh synthetic suites | 26A428 | 27.0 / SDK 27.0 | coremltools 9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2 |
   | Native MLP follow-up | 26A428 | 27.0 (Swift 6.4) | — |
   | Historical Python MLP | — | 26.6 / SDK 26.5 | MLX 0.32.2; its SDK blocker is not the native follow-up's status |
   | Historical quantization matrix | 26A5425a | — | coreai-torch 0.4.1 and 0.4.2 |
   | G1-W quantized speed and QDQ probes | 26A428 | 27.0 (27A266a) | coreai-torch 0.4.1, Torch 2.11.0 |
   | G5 weight-free attention | not recorded | not recorded | coreai-torch 0.4.2, Torch 2.9.0 |

   G2's versions come from same-day preflight records and a post-run statement
   that no software changed that day. Version and source identities belong to
   each run, not to a shared headline. No independent host replication is
   included.

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

## G3 complete-model observations

G3 measures complete Qwen3-4B FP16 on one M5 Pro through Core AI on both arms: a static-shape ANE path and a sequential GPU path. This differs from the G2 MLP comparison against MLX. The main plots use warmed stage blocks at inputs <!-- claim:g3.contexts@g3-001 -->500 / 1,024 / 2,048 / 4,096 / 8,192 / 16,384<!-- /claim -->; the ANE graph contexts are <!-- claim:g3.graph-contexts@g3-002 -->256 / 2K / 32K<!-- /claim -->. Actual graph calls are recorded, but no per-operation placement trace was collected in these blocks. The power-domain response supports activity on the requested accelerator, not exclusive execution of every operation there.

Both arms use identical source weights and initial token IDs. Decode performs <!-- claim:g3.decode-steps@g3-003 -->1,024<!-- /claim --> forwards from a prepared cache, including steps after EOS. Output tokens are freely generated, not forced to match. The recorded short-answer, cache and first-output reference checks do not establish equal task quality or perplexity. Inputs come from a single family built from the repository documentation; model assets and original logits are not in the portable package.

The main speed, mean power and J/token refer to the same work block, including host and request-handling time. Model load, pre-block warmup and post-block recovery are excluded. The initial single-request coverage results are separate measurements, each with <!-- claim:g3.coverage-steps@g3-004 -->256<!-- /claim --> decode forwards, and are not a cold-start benchmark. Consecutive requests or decode segments are not independent process repetitions.

Energy is the CPU + GPU + ANE software component sum during the work interval, without idle subtraction. It excludes unreported domains and is not wall-input energy. Thermal starts were not matched, observer overhead was not independently isolated, and estimator accuracy and latency are uncalibrated. The plotted bounds describe sample-time attribution under the stated timing assumptions, not statistical confidence or a calibrated sensor error bar. [Energy calculation](METHODS.md#g3-complete-model-stages).

The short GPU prefill block at <!-- claim:g3.n.500@g3-005 -->500<!-- /claim --> had too few interior power samples for its response rule. The primary pair uses the longer follow-up from r6, with <!-- claim:g3.supplement-count@g3-006 -->192<!-- /claim --> requests per arm and the same r5 capture. Other primary pairs come from r5. Original short blocks remain listed with their response status.

The implied compute and read rates use the measured token rates and the model's structure record. Prefill counts two FLOPs per projection weight per token; attention is listed separately. Decode assumes one read of every FP16 weight and the existing KV at each step, summing the cache growth across the block. The byte model excludes current-token writes and fixed-graph padding. These are effective model-work rates, not device counters; no DRAM traffic was measured. A nearly flat modelled byte rate is consistent with bandwidth-dominated execution, but cannot identify the bottleneck or distinguish ANE memory-access limits from graph and host costs. The same-machine synthetic ANE FP16 reference uses a different graph on one fixed shape, not a hardware peak. [Calculation](METHODS.md#g3-complete-model-stages).

Larger contexts select larger fixed ANE functions. The function selection and curve transitions are observed together; a controlled change of graph shapes is still needed to isolate their cost. These rates do not directly measure UMA bandwidth, and G3 FP16 does not establish a quantized-path benefit. [Article](../articles/05-qwen3-4b-prefill-decode-energy.md) · [Bundle](../results/historical/g3-qwen3-4b/).

## G1-W and G5 component observations

G1-W measures quantized speed on synthetic 1×1 convolution chains and on the first MLP of Gemma 4 E4B mobile QAT, and isolates the QDQ multiply defect with model-free graphs. G5 measures a weight-free attention graph. Neither runs a complete model or measures energy.

Times are awaited `function.run` calls from a native Swift host, in two or three fresh processes per arm; ranges are process medians, not confidence intervals. An A8 graph that is faster is not evidence of a physical INT8 datapath. ANE request logs show participation per call, not per-operation placement. The E4B stack repeats one real layer eight times behind a shared RMS norm, and its inputs are synthetic control rows. Relative L2 there, and in G5, compares a device output with its own CPU or FP64 reference: it is implementation error, not model quality.

G5's inputs are synthetic uniform, random and small-constant attention; the complete-model G3 path was not re-evaluated against it. Its relative L2 values are imported scalars, while its timings, like all G1-W timings, are recomputed from per-call records. The model checkpoint, activations and FP16 attention arrays are not distributed. [Speed conditions](../findings/quantized-speedup-conditions/) · [QDQ multiply](../findings/coreai-qdq-multiply-scale/) · [Attention precision](../findings/attention-product-precision/).

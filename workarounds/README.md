# What actually works

The [findings](../findings/) are mostly about what does not. This is the other
half: three things that do run on the Neural Engine, each with its price and its
boundary, plus one that keeps an ANE machine's disk usable. The first three are cases
in the test suite, so you can check them on your own machine.

| Want to… | Do this | Price |
|---|---|---|
| Try the tested grouped-scale rewrite | [Split into per-output-channel scales](#1-split-grouped-weights-into-per-output-channel-scales) | About **4× slower** in a separate synthetic ablation |
| Get a correct QDQ multiply | [Clamp the product, or write the quantize step explicitly](#2-fix-a-qdq-multiply) | Clamp: no measurable time; neither form makes a real QAT MLP match |
| Just get acceleration | [Stay with per-tensor or per-output-channel 8-bit](#3-stay-with-per-tensor-or-per-output-channel-8-bit) | Not the format your checkpoint ships in; needs a deep enough chain |
| Get back disk space that ANE model loads leave behind | [End `ANECompilerService` when it holds deleted compile inputs](#4-reclaim-disk-space-held-by-the-ane-compiler-service) | A compile request running at that moment may fail once |

---

## 1. Split grouped weights into per-output-channel scales

**The problem.** Core ML's compiler permits a quantization scale that varies per
output channel or per tensor, not along the reduction dimension —
[its diagnostic states this](../findings/coreml-grouped-scale-cpu/). The tested direct signed-INT4 graph
varies its scale along that dimension and is CPU-selected. Other formats and
graph representations have not been exhaustively tested.

**The rewrite.** Cut the weight along K into groups, give each slice its own
convolution with a single per-output-channel scale, and add the partials back:

```
W x  =  (s₀ q₀) x₀  +  (s₁ q₁) x₁  +  …
```

In exact arithmetic this is the same function. One K512 convolution becomes
sixteen K32 convolutions and a balanced fifteen-add FP16 tree per layer.

**What it buys.** On the small K64 compatibility fixture, the Core AI split path is numerically exact against its
frozen reference — relative L2 **0**, zero mismatches — with ANE participation in
all four control windows. The native path on the same weights is
[wrong](../findings/coreai-flattened-scale/).

**A separate cost measurement.** In a synthetic K512 fixture with shared scales,
[about 4× the latency](../findings/split-decomposition-cost/):
the wide path is **3.96–4.10× faster** than the split one across three paired
process rounds. Partition, partial sums, the reduction tree, kernel selection and
fusion all change together, so no single component is charged for it.

**The catch.** This worked on Core AI at the tested shape.
On **Core ML** the 64-output split was numerically correct and still chose
`MLCPUComputeDevice`, with zero ANE events — while the historical 15360-output
Gemma-derived split *did* select ANE. Output-channel count has not been isolated
as the cause, and those two fixtures differ in more than shape. **Splitting is a tested option,
not a universal requirement or placement guarantee: check your own shape.**

**Check it:**

```sh
.venv/bin/ane-scope run --suite compatibility --output runs/my-groups
.venv/bin/ane-scope verify runs/my-groups
```

Compare `coreai-group-split32` against `coreai-group-native64`, and read the
`placement.json` in each run directory for supported versus preferred devices and
the per-PID ANE event windows. The exporter that builds both shapes is in
[`_coreai.py`](../src/ane_scope/_coreai.py); the Core ML side is in
[`_coreml.py`](../src/ane_scope/_coreml.py).

---

## 2. Fix a QDQ multiply

**The problem.** On an exact grid with no midpoint tie and no saturation, a
standard QDQ multiplication returns the wrong product as soon as the two branches
carry different scales — relative L2 **0.75**, maximum absolute error **12**, on a
target of 16. Merely reversing the order in which the two branches are built
changes the error to 3.0. All of these show ANE participation in every control.
The wrong values fit [one rule](../findings/coreai-qdq-multiply-scale/): a branch is
dequantized with another QDQ's scale. Core ML returns the same wrong values when it runs
the graph on the Neural Engine, so the rewrites below apply to both runtimes.

**The cheaper rewrite: clamp the product.** When the product feeds another QDQ, clamp it
to that QDQ's representable range before quantizing:

```python
p = (a * qdq(b, s_in)).clamp(min=-128 * s_out, max=127 * s_out)
return qdq(p, s_out)
```

Quantization saturates at those bounds anyway, so the clamp never changes the QDQ's output. It restored the correct output
in all four arms of the model-free probe, through Core AI and through Core ML, kept one ANE request per call, and cost no
measurable time in an eight-MLP stack (<!-- claim:g1w.e4b.clip-product.speed@g1w-001 -->0.976×<!-- /claim --> against <!-- claim:g1w.e4b.native.speed@g1w-002 -->0.977×<!-- /claim --> without it). On a real
Gemma 4 E4B QAT MLP it removed the gross error but left <!-- claim:g1w.e4b.1.clip-product.l2@g1w-003 -->6.31%<!-- /claim --> for one layer and <!-- claim:g1w.e4b.8.clip-product.l2@g1w-004 -->21.6%<!-- /claim -->
for eight repeated layers, so it is not a complete fix there.

**The explicit rewrite.** Express the quantize half yourself — divide by the scale, round,
clamp, cast to integer — and keep the dequantize as it was:

```python
q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
return torch.ops.coreai.dequantize.default(q, scale, zero, None, 0, torch.int8)
```

**What it buys.** Relative L2 **0**, maximum absolute error **0**, with ANE
participation in all four controls — same inputs, same scales, same target, on the
same runtime version that gets it wrong the other way.

**What is unknown.** This is verified on **one minimal probe**: inputs 2 and 8,
scales 0.5 and 2, an exact grid. Whether it holds for wider input ranges, other
scale pairs, saturation boundaries or larger graphs is **not tested**, and neither
is its performance — the probe is not timed. Treat it as a lead worth trying on
your own case, not a general fix.

**Keep the division in FP16 before compressed weights.** An FP32 division, cast back to
FP16 and fed to a W4 or W8 convolution, makes ANE compilation fail and runs the whole
graph on the GPU, with output that still passes a numerical check. Look for
`Falling back to full compile on GPU` in the target-process log.
[The fallback table](../findings/coreai-qdq-multiply-scale/#the-explicit-quantize-workaround-has-a-boundary).

**Check it:** the same `compatibility` suite. Compare `coreai-qdq-explicit_q`
against `coreai-qdq-unequal` and `coreai-qdq-reverse_order`; the four graph
expressions are in [`_coreai.py`](../src/ane_scope/_coreai.py) under `Multiply`.

---

## 3. Stay with per-tensor or per-output-channel 8-bit

**What works.** A quantization scheme the compiler already accepts — one scale
per tensor or per output channel — accelerates on a long enough chain. On a
controlled 128-layer chain, Core AI W8A8 runs **1.86–1.87×** faster than its FP16
baseline. Core ML also accelerates, with every audited convolution preferring the
Neural Engine; both runtimes show ANE participation in every control window.

**Why it is here.** It answers "does quantization buy anything on
this accelerator at all". It does. The problem is that it is not the format your
4-bit checkpoint ships in, so getting there means requantizing — which is a model
quality question this repository has not answered, and cannot answer with
model-free fixtures.

**The boundary.** The gain [depends on the work in each call and on the weights](../findings/quantized-speedup-conditions/).
A8W4 runs at <!-- claim:g1w.depth.both-1.speed@g1w-005 -->0.86×<!-- /claim --> W4A16 speed as a single layer and <!-- claim:g1w.depth.both-128.speed@g1w-006 -->1.33×<!-- /claim --> at 128 layers; a real E4B gate
runs it at <!-- claim:g7.coreai.gate.1024.a8w4-over-w4a16.speed@g7-001 -->1.359–1.360×<!-- /claim --> at 1024 positions and gains nothing at 64. Time the call size you will serve
before concluding anything. Exact zero weights make the FP16
baseline itself <!-- claim:g1w.density.old.speed@g1w-007 -->1.88×<!-- /claim --> faster. A released per-channel A8W4 checkpoint, eight repeated
Gemma 4 E4B QAT MLPs, gained nothing (<!-- claim:g1w.e4b.native.speed@g1w-008 -->0.977×<!-- /claim -->).

**Check it:**

```sh
.venv/bin/ane-scope run --suite throughput --output runs/my-throughput
python scripts/summarize.py
```

Numbers and their unit are in [MEASUREMENTS.md](../docs/MEASUREMENTS.md); what
"source-equivalent ops/s" does and does not mean is in
[SCOPE.md](../docs/SCOPE.md).

---

## 4. Reclaim disk space held by the ANE compiler service

**The problem.** On the tested M5 Pro with macOS 27.0 and Qwen3-4B FP16 assets, each ANE model load left its multi-gigabyte compile input open in
`ANECompilerService` after the loading process exits and the file is deleted. The
space returns only when the service exits, and `du` cannot see it.
[The finding](../findings/ane-compiler-service-disk/) has the listings and free-space readings.

**Check it.** List what the service holds; `+L1` keeps deleted files and `-a` restricts
the listing to the service:

```sh
sudo lsof -nP -a +L1 -c ANECompiler
```

**Reclaim it by hand.** When no ANE model is loading or running, end the service. It
did not respond to SIGTERM on the observed machine, so send SIGKILL; launchd starts it
again on the next compile request.

```sh
sudo kill -9 <PID from the listing>
```

**Or install a periodic task.** [`ane-compiler-reclaim/`](ane-compiler-reclaim/) holds a
root LaunchDaemon that runs every five minutes and ends the service only when all of these
hold: no `powermetrics` process, no live
process owns an `mpsgraph-*` scratch directory or any file the service holds, and the
deleted inputs total at least 2 GiB. It tries SIGTERM, then SIGKILL after 10 seconds, and
logs each action with free space before and after to `/Library/Logs/ane-compiler-reclaim.log`.
These checks reduce interference with ongoing work.

```sh
sudo sh workarounds/ane-compiler-reclaim/install.sh
sudo ANE_RECLAIM_DRY_RUN=1 sh /usr/local/libexec/ane-compiler-reclaim.sh   # current decision
```

The installer copies the script to root-owned `/usr/local/libexec/`, because root runs
it. To remove it: `sudo launchctl bootout system/local.ane-compiler-reclaim`, then delete
`/Library/LaunchDaemons/local.ane-compiler-reclaim.plist` and the script.

**The catch.** The checks do not reserve an idle window; a compilation starting during
reclamation may be interrupted. While `powermetrics` is detected, reclamation is skipped,
so a run with many ANE loads still needs disk for all of them. Only macOS 27.0 with Qwen3-4B assets has been
observed. The decision rules are covered by dry-run tests (`tests/test_ane_compiler_reclaim.py`);
the signalling path ran once for real, releasing the space it logged.

---

## Diagnosing your own case

The CLI runs fixed, model-free suites — you cannot point it at your checkpoint.
The *method* transfers, though. For any operation you suspect:

1. **Read the persisted asset, not the program you built.** Reload the saved
   package and decode weights, scales and zero points independently. An
   unrecognised representation should fail your audit.
   ([how](../src/ane_scope/_coreml.py))
2. **Read the compute plan, and keep `supported` apart from `preferred`.** They
   disagree, and neither substitutes for actual execution evidence.
3. **Capture target-PID unified logs around each call.** Requesting a compute
   unit is not evidence; a successful ANE request inside the call window is.
   Compiler refusals show up here too — that is how the Core ML constraint was
   found, verbatim.
4. **Freeze a reference before you look at the output**, and write down its
   intermediate precision and rounding rule. A reference disagreement can explain a discrepancy; it does not
   repair the device output or establish model quality.
5. **When something is wrong, write down what you think it is doing instead and
   test that too.** A wrong-on-purpose reference that the broken case satisfies
   exactly, and the working case violates, is much stronger evidence than a large
   error.

## What has no workaround here

- **Long-term residency remains unverified.** The historical Python path grew by
  [1,966,080 attributed bytes per gate call](../findings/iosurface-per-call-growth/).
  Later native host and tiled-asset rounds passed bounded C64/C256 checks; the old SDK
  blocker no longer describes current progress. [Follow-up](../results/historical/native-mlp-followup.json).
- **The GPU gap remains.** Historical Python ratios are separate from the later
  native C256 result: 582.19 ms versus its same-round GPU 152.25 ms at 4K.
  No matched real-MLP ablation establishes how much splitting contributes to it.
  G2 adds finite thermal and coexistence observations, with display and thermal-start
  limits; energy remains open. [Service comparison](../findings/w4a16-service-tradeoffs/).

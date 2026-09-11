# What actually works

The [findings](../findings/) are mostly about what does not. This is the other
half: three things that do run on the Neural Engine, each with its price and its
boundary. Every one of them is a case in the test suite, so you can check it on
your own machine.

| Want to… | Do this | Price |
|---|---|---|
| Try the tested grouped-scale rewrite | [Split into per-output-channel scales](#1-split-grouped-weights-into-per-output-channel-scales) | About **4× slower** in a separate synthetic ablation |
| Get a correct QDQ multiply with unequal scales | [Write the quantize step explicitly](#2-write-the-quantize-step-explicitly) | Unknown — verified on one exact-grid probe only |
| Just get acceleration | [Stay with per-tensor or per-output-channel 8-bit](#3-stay-with-per-tensor-or-per-output-channel-8-bit) | Not the format your checkpoint ships in |

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

## 2. Write the quantize step explicitly

**The problem.** On an exact grid with no midpoint tie and no saturation, a
standard QDQ multiplication returns the wrong product as soon as the two branches
carry different scales — relative L2 **0.75**, maximum absolute error **12**, on a
target of 16. Merely reversing the order in which the two branches are built
changes the error to 3.0. All of these show ANE participation in every control.

**The rewrite.** Express the quantize half yourself — divide by the scale, round,
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

**Check it:** the same `compatibility` suite. Compare `coreai-qdq-explicit_q`
against `coreai-qdq-unequal` and `coreai-qdq-reverse_order`; the four graph
expressions are in [`_coreai.py`](../src/ane_scope/_coreai.py) under `Multiply`.

---

## 3. Stay with per-tensor or per-output-channel 8-bit

**What works.** A quantization scheme the compiler already accepts — one scale
per tensor or per output channel — accelerates, and does so reliably. On a
controlled 128-layer chain, Core AI W8A8 runs **1.86–1.87×** faster than its FP16
baseline. Core ML also accelerates, with every audited convolution preferring the
Neural Engine; both runtimes show ANE participation in every control window.

**Why it is here.** It answers "does quantization buy anything on
this accelerator at all". It does. The problem is that it is not the format your
4-bit checkpoint ships in, so getting there means requantizing — which is a model
quality question this repository has not answered, and cannot answer with
model-free fixtures.

**Check it:**

```sh
.venv/bin/ane-scope run --suite throughput --output runs/my-throughput
python scripts/summarize.py
```

Numbers and their unit are in [MEASUREMENTS.md](../docs/MEASUREMENTS.md); what
"source-equivalent ops/s" does and does not mean is in
[SCOPE.md](../docs/SCOPE.md).

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
  Later native HOST / PIO / TILE work passed bounded C64/C256 checks; the old SDK
  blocker no longer describes current progress. [Follow-up](../results/historical/native-mlp-followup.json).
- **The GPU gap remains.** Historical Python ratios are separate from the later
  native C256 result: 582.19 ms versus its same-round GPU 152.25 ms at 4K.
  No matched real-MLP ablation establishes how much splitting contributes to it.
  G2 adds finite thermal and coexistence observations, with display and thermal-start
  limits; energy remains open. [Service comparison](../findings/w4a16-service-tradeoffs/).

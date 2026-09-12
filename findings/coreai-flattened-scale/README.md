# A Core AI grouped-scale probe matches a flattened-scale error model

## Symptom

A `[64, 64, 1, 1]` signed-INT4 convolution with two K32 scales per output row
and an identity input. The persisted asset audits clean — four-bit indices, LUT,
the two-dimensional scale array and the operand connections are all what they
should be — and the call shows ANE participation in all four control windows.

The output is still wrong: **1921 of 4096 values** miss the intended reference,
at a relative L2 of **0.450247**.

## The error model and its control

A candidate error was generated separately and frozen before comparison:
*flatten the two-dimensional scale array and reuse its first 64 entries as one
scale per output row.* Comparing the device output against that wrong-on-purpose
reference gives a control pair:

| Case | vs the intended reference | vs the flattened-scale prediction |
|---|---|---|
| `coreai-group-native64` | fails, L2 0.450247, 1921 mismatches | **passes exactly, L2 0, 0 mismatches** |
| `coreai-group-split32` | passes exactly, L2 0 | fails, L2 0.450393, 1921 mismatches |

The broken case satisfies the wrong model to the value; the correct case
violates it, which makes `coreai-group-split32` a matched negative control.

Both cases show ANE participation in every control window. **A wrong answer and
a real ANE request coexist here** ; placement evidence
and numerical evidence are therefore recorded as separate fields.

## Reproduce

The [archived reproducer](repro/README.md) contains the original exporter and Swift host, synthetic inputs and recorded outputs. Verify those outputs offline from the repository root:

```sh
python findings/coreai-flattened-scale/repro/verify.py
```

The archive README gives the environment, device commands and upstream issue status. Its small device runner has not been rerun; the bundled output belongs to the recorded smoke experiment.

## What this does not show

An output that matches a wrong scale-indexing model is consistent with that
model. It is not a trace of the compiler or the kernel, and no internal code
path has been observed. Core AI and Core ML use different weight
representations, so this cannot be attributed to a shared backend stage — see
[the Core ML behaviour](../coreml-grouped-scale-cpu/), which fails differently.

Scoped to coreai-torch 0.4.1 on the recorded host. The historical 0.4.1/0.4.2
matrix is retained separately in
[quantization-expected-results.json](../../results/historical/quantization-expected-results.json);
no fresh 0.4.2 run is claimed.

## Evidence

- [smoke.json](../../results/fresh/smoke.json) — `coreai-group-*`, including
  the per-case `flattened-scale-hypothesis` block
- [MEASUREMENTS.md](../../docs/MEASUREMENTS.md) — the smoke observation table
- [The exporter that builds these graphs](../../src/ane_scope/_coreai.py)

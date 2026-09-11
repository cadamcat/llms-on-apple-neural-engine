# Core ML routes K-grouped scales to the CPU, and says why

## Symptom

The same grouped-INT4 weights, expressed through Core ML's direct signed-INT4
blockwise scaling rather than a four-bit LUT, are **numerically correct** — and
never touch the Neural Engine.

Two shapes, two different refusals:

| Case | Output | Devices the plan lists as supported | Preferred | ANE events in the four control windows |
|---|---|---|---|---:|
| `coreml-group-native64` | correct, L2 0 | **CPU only** | `MLCPUComputeDevice` | 0 |
| `coreml-group-split32` | correct, L2 0 | CPU **and** Neural Engine | `MLCPUComputeDevice` | 0 |

The native case is not eligible at all. The split case *is* eligible, is not
chosen, and produces no ANE request. So "ANE supported", "ANE preferred" and
"ANE actually ran" are three different things, and only the third is evidence of
execution.

## The compiler states the constraint outright

On the original Gemma-derived fragment — a `[15360, 64, 1, 1]` weight with a
`[15360, 2, 1, 1]` group scale — the Core ML compiler emitted, twice:

```
Unsupported op 4 (ios18.conv): ANE only support per-cout/per-tensor quantization
```

That is the vendor toolchain naming the limitation: quantization scale may vary
per output channel or per tensor, but not along the reduction dimension K.
The tested group scales vary along K. This diagnostic constrains that direct
convolution representation; it does not cover every checkpoint format, group size
or alternative graph.

Decomposition into per-output-channel scales is one tested workaround. The
diagnostic does not establish that it is the only possible implementation.
What it costs is
[measured separately](../split-decomposition-cost/).

## Reproduce

```sh
.venv/bin/ane-scope run --suite compatibility --output runs/my-groups
.venv/bin/ane-scope verify runs/my-groups
```

Each run records the compute plan's supported and preferred devices per
convolution, plus the target-PID unified-log window. Without a device, the
stored plan entries are in
[smoke.json](../../results/fresh/smoke.json) and summarised in
[MEASUREMENTS.md](../../docs/MEASUREMENTS.md).

## What this does not show

The historical diagnostic came from a 15360-output fragment carrying real Gemma
weights and scale distribution. The model-free probe here has 64 outputs. **They
differ in more than output-channel count**, and the older K32 split did select
ANE where this small one does not — so output-channel count has *not* been
isolated as the cause, and no general claim about all Core ML INT4 graphs
follows. A CPU choice also does not prove K32 is unsupported; it shows this
graph, at this shape, on this toolchain, was not selected.

## Evidence

- [coreml-native-k64-rejection.json](../../results/historical/coreml-native-k64-rejection.json)
  — the two diagnostics with timestamps, PID scope, both shapes and the source hash
- [smoke.json](../../results/fresh/smoke.json) — `coreml-group-*` plans and
  placement windows
- [The exporter and its persisted-asset audit](../../src/ane_scope/_coreml.py)

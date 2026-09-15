# QDQ multiply scale reproduction

An all-ones input through `Q_out(a × Q_1/16(b))` should return 1 at every output scale. On the tested Core AI → ANE path it returns 1, 2, 4 and 8 for output scales 1/16, 1/8, 1/4 and 1/2. A product clamp that leaves the true value unchanged restores 1 in every arm. Core ML returns the same values on the Neural Engine; see [Core ML](#core-ml). No model, checkpoint or downloaded data is needed.

This supports the [QDQ multiply finding](../README.md). The Core AI probe is reported as [apple/coreai-torch #112](https://github.com/apple/coreai-torch/issues/112).

## Verify the recorded outputs without a device

From the repository root, with Python 3.12 and no third-party packages:

```sh
python findings/coreai-qdq-multiply-scale/repro/verify.py
```

The verifier checks each recorded output's size, the zero and repeat controls, one ANE request per call, the correct value for the clamped arms and the scale-substitution prediction for the unclamped arms. Pass `--outputs <run>/host-output` to report the same comparisons for a new run; a corrected runtime is reported as data, not as a failure.

## Export and run on an Apple silicon Mac

Recorded environment: Apple M5 Pro, macOS 27.0 (26A428), Xcode 27.0 (27A266a), coreai-torch 0.4.1, Torch 2.11.0 and NumPy 2.3.5. The exporter needs coreai-torch with Core AI authoring; the host uses the macOS 27 Core AI SDK.

```sh
python export.py /tmp/qdq-mul-result
python run.py /tmp/qdq-mul-result
python verify.py --outputs /tmp/qdq-mul-result/host-output
```

`run.py` compiles `host.swift`, runs the original input, a zero input and the original input again on each of the eight graphs, and keeps the raw FP16 outputs and the target-process unified log. Use a new result directory for each export.

## Core ML

`coreml/` runs the same graph through Core ML. `export_coreml.py` builds the eight arms with the coremltools MIL builder; `--half 512 --positions 1024` gives the recorded 1,024 × 1,024 size, at which Core ML places the graph on the Neural Engine (at the default 32 × 64 it selects the CPU). `run_probe.py` runs each arm with `cpuAndNeuralEngine` and with `cpuOnly`, one host per run: the original, zero, sign-flipped, repeated, `b` = 0.5 and repeated `b` = 0.5 inputs, with the compute plan and the host-PID unified log. `host.swift` is the G7 same-code host with a `compute_units` field added.

Recorded environment: Apple M5 Pro, macOS 27.0 (26A428), Xcode 27.0, coremltools 9.0, NumPy 2.3.5. From `coreml/`:

```sh
mkdir -p bin && xcrun swiftc -parse-as-library -O -target arm64-apple-macos27.0 host.swift -o bin/probe-host
python export_coreml.py /tmp/qdq-coreml-export --half 512 --positions 1024
python run_probe.py /tmp/qdq-coreml-export /tmp/qdq-coreml-run
```

Check the recorded outputs without a device:

```sh
python findings/coreai-qdq-multiply-scale/repro/coreml/verify.py
```

It checks every output's size and single value, the zero and repeat controls, the graph text, the preferred devices and ANE request counts, the substitution values on the unclamped Neural Engine arms and the correct values everywhere else. `coreml/recorded/` is written by `results/historical/import_qdq_coreml.py`; the three sources are the research-workspace files, unchanged.

## Files

`host.swift` is byte-identical to the executed package. `export.py` and `run.py` were reformatted for reading; their parsed syntax trees are identical to the executed files. `recorded/` holds the raw outputs, the saved graph text and the per-call ANE request counts, imported by `results/historical/import_g1w.py`. Source and records are provided under the MIT license in `LICENSE`; no Apple framework binaries or compiled models are included.

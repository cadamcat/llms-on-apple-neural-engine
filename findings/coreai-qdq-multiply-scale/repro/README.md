# Core AI QDQ multiply scale reproduction

An all-ones input through `Q_out(a × Q_1/16(b))` should return 1 at every output scale. On the tested Core AI → ANE path it returns 1, 2, 4 and 8 for output scales 1/16, 1/8, 1/4 and 1/2. A product clamp that leaves the true value unchanged restores 1 in every arm. No model, checkpoint or downloaded data is needed.

This supports the [QDQ multiply finding](../README.md). No upstream issue has been filed yet.

## Verify the recorded outputs without a device

From the repository root, with Python 3.12 and no third-party packages:

```sh
python findings/coreai-qdq-multiply-scale/repro/verify.py
```

The verifier checks each recorded output against its hash, the zero and repeat controls, one ANE request per call, the correct value for the clamped arms and the scale-substitution prediction for the unclamped arms. Pass `--outputs <run>/host-output` to report the same comparisons for a new run; a corrected runtime is reported as data, not as a failure.

## Export and run on an Apple silicon Mac

Recorded environment: Apple M5 Pro, macOS 27.0 (26A428), Xcode 27.0 (27A266a), coreai-torch 0.4.1, Torch 2.11.0 and NumPy 2.3.5. The exporter needs coreai-torch with Core AI authoring; the host uses the macOS 27 Core AI SDK.

```sh
python export.py /tmp/qdq-mul-result
python run.py /tmp/qdq-mul-result
python verify.py --outputs /tmp/qdq-mul-result/host-output
```

`run.py` compiles `host.swift`, runs the original input, a zero input and the original input again on each of the eight graphs, and keeps the raw FP16 outputs and the target-process unified log. Use a new result directory for each export.

## Files

`host.swift` is byte-identical to the executed package. `export.py` and `run.py` were reformatted for reading; their parsed syntax trees are identical to the executed files, whose hashes are in `recorded/results.json`. `recorded/` holds the raw outputs, the saved graph text and the per-call ANE request counts, imported by `results/historical/import_g1w.py`. Source and records are provided under the MIT license in `LICENSE`; no Apple framework binaries or compiled models are included.

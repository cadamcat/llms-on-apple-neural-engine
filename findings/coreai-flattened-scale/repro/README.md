# Core AI grouped-LUT convolution reproduction

A signed-INT4 LUT convolution with two K32 scales per output channel returns 1921 incorrect values out of 4096 in the recorded run, at a relative L2 of 0.450247. Splitting it into two K32 convolutions with per-output scales returns the intended values. The native output matches a flattened-scale prediction: use the first 64 entries of the scale array as one scale per output channel.

This archive supports the [grouped-scale finding](https://github.com/cadamcat/llms-on-apple-neural-engine/blob/cce9b49dcd5ad49070c1b9dab09b64e22e40042a/findings/coreai-flattened-scale/README.md) and upstream bug investigation. It provides a standalone numerical reproduction with an identity input, saved outputs and the measured source files. No model download or timed benchmark loop is needed.

## Upstream tracking

**Issue:** [apple/coreai-torch #93](https://github.com/apple/coreai-torch/issues/93), submitted on 2026-09-12. The report links the [reproduction snapshot](https://github.com/cadamcat/llms-on-apple-neural-engine/tree/ba79d22a0df214e3dcb9a1e9baeff02caaa137d1/findings/coreai-flattened-scale/repro). No Apple Feedback reference has been assigned.

## Verify the recorded output without a device

Run from this directory (or the extracted archive):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy==2.3.5
.venv/bin/python verify.py
```

The verifier regenerates the seeded codes, checks the exact input QDQ grid, independently computes the intended and flattened-scale outputs, and compares all saved outputs with the published records. It also checks fixture and frozen-source hashes. Numeric comparison treats positive and negative zero as equal; the original and repeated outputs must be byte-identical.

## Export and run on an Apple silicon Mac

The recorded environment was M5 Pro, 48 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), SDK 27.0 and Python 3.12.14. The exporter uses version-specific private authoring interfaces in coreai-torch 0.4.1 and coreai-core 1.0.0b2.

```sh
uv pip install --python .venv/bin/python -r requirements-device.txt
.venv/bin/python run_device.py --output run-01
```

This explicitly exports the two assets, compiles the Swift host and runs the original, zero, negative and repeated input on each asset. The host requests `.neuralEngine`. The historical placement counts and asset audits are in `recorded/results.json`; the small runner does not collect new placement logs. See the [measurement scope](https://github.com/cadamcat/llms-on-apple-neural-engine/blob/cce9b49dcd5ad49070c1b9dab09b64e22e40042a/docs/SCOPE.md) for the limits of device preferences and participation evidence.

`run-01/comparison.json` reports the new numerical results, including a corrected native output if the discrepancy no longer reproduces. Existing output directories are refused. A failure leaves its logs and assets in that directory.

## Source and records

The files under `frozen/ane_scope/` are unchanged copies from the measured source snapshot. Their hashes match the [published smoke record](https://github.com/cadamcat/llms-on-apple-neural-engine/blob/cce9b49dcd5ad49070c1b9dab09b64e22e40042a/results/fresh/smoke.json). `references/prepare.py` preserves the original fixture generator for inspection; the small runner uses the saved files in `fixture/`. The recorded outputs, fixture arrays and MLIR text are copied unchanged. `recorded/results.json` selects the two cases and their environment from the published record.

`run_device.py` and `verify.py` are new packaging helpers. The package has been verified offline; the small device runner has not been executed, and there is no new result for coreai-torch 0.4.2. Matching the flattened-scale prediction identifies an output pattern, not the compiler or runtime stage responsible.

Source and generated records are provided under the MIT license in `LICENSE`. Dependencies are installed separately under their own licenses. No Apple framework binaries or compiled models are included.

## Build a ZIP copy

The repository directory is the maintained source. From the repository root:

```sh
python findings/coreai-flattened-scale/repro/archive.py --output dist/coreai-grouped-lut-repro.zip
```

The archive contains the documented files, frozen source, fixtures and recorded evidence. Local environments and new device outputs are excluded. CI checks the repository copy and an extracted ZIP without executing a device. When citing the reproduction in an issue, use a link pinned to its repository commit; the ZIP is an optional download of the same files.

# A candidate arithmetic model for group-quantized gate outputs

A closed accelerator does not document its arithmetic. We derived a candidate execution model, froze it, and then checked it
against outputs the device had already produced.

## The model

For each group of 32 input channels:

1. Decode the original Q4_0 weights to binary16; apply the original input QDQ.
2. Take the **exact** dot product of the 32 decoded weights and resulting inputs.
3. Round to nearest binary16, with midpoint ties **away from zero** (RZA).
4. Apply the same rounding at every add of the original balanced reduction tree.
5. Apply the original group-32 output QDQ.

Step 3 is the part that is not obvious. The declared arithmetic and the
documented behaviour of the surrounding framework both suggest ties-to-even;
the tested outputs are better predicted by the ties-away model. A later probe
of a single Core AI QDQ on ANE, fed all <!-- claim:g1w.qdq.finite-values@g1w-001 -->63,488<!-- /claim --> finite FP16 values at unit scale,
matched ties-away at every value; ties-to-even would have differed at 128 half-integers.
[Imported counts](../../results/historical/g1w-e4b-mobile-qat/evidence.json).

## What it predicts

| Model | Final Q8 gate outputs | Q8 residuals under this model | Q8 residuals under the original reference | Reduction |
|---|---:|---:|---:|---:|
| E2B | 1,449,984 | **0** | 13,903 | 100% |
| Qwen8B | 5,713,920 | **13** | 31,155 | 99.958% |

After output QDQ, E2B is byte-exact in this round; Qwen8B has 13 final Q8
numeric mismatches — **13 unexplained values out of 7,163,904**. Maximum final-Q8
relative L2 falls from 5.29e-3 to 0 for E2B, and from 8.83e-3 to 2.98e-4 for Qwen8B.

Before output QDQ, the model misses **16 E2B values and 1,194 Qwen8B values**.
Quantization hides many small differences. The final-Q8 agreement is not a claim
of byte-exact dot products, intermediate arithmetic or pre-QDQ gate outputs.

Both predictions were computed twice by independent implementations — one using
exact integer arithmetic in units of 2⁻²⁴ with proven bounds against int64
overflow, one using FP64 dot products with an adjacent-binary16 midpoint method
and a different tree traversal — and cross-checked byte for byte.

## Reproduce

No device and no model weights needed:

```sh
python results/historical/tests/verify_arithmetic.py
```

This recomputes the agreement fractions and residual reduction from the stored
counts, and checks that the published residual list matches them.

## What this establishes, and what it does not

It establishes that a *specific, written-down* arithmetic model reproduces this
accelerator's final Q8 gate output on the recorded inputs from two real models,
with the residual counts above. That match is what makes the remaining 13 residuals interesting rather than noise.

It does **not** establish the internal implementation. Matching outputs is
consistent with the model; it does not prove the hardware performs those steps
in that order. It is also scoped to these assets, these inputs and this
toolchain version: the strict cross-model byte gate did **not** pass.

These are gate outputs, not full-model results. No quality, token-rate or
energy claim follows from any of it.

## Evidence

- [cross-model-validation.json](../../results/historical/cross-model-validation.json)
  — per-model counts, the 13 residual coordinates, and source hashes
- [verify_arithmetic.py](../../results/historical/tests/verify_arithmetic.py)
  — the recomputation, standard library only
- The frozen ties-away reference in
  [prepare.py](../../src/ane_scope/references/prepare.py) is the same rule,
  reimplemented model-free for the synthetic suites
- Where the model breaks: [fp16-dot-residual](../fp16-dot-residual/)

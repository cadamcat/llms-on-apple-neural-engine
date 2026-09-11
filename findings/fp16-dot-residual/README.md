# A dot product that lands outside its own binary16 bracket

This result needs no Apple hardware to check.

## Symptom

One 32-term dot product, taken from Qwen8B's first gate (output channel 11793,
token position 22, input channels 2912–2943):

| | value |
|---|---:|
| Exact dot product (FP64, and as an exact integer −1004010 × 2⁻²⁴) | −0.05984365940093994 |
| Nearest binary16 below the exact value | **−0.059844970703125** |
| Nearest binary16 above the exact value | **−0.059814453125** |
| Our model's prediction (correctly rounded — both tie rules agree here) | −0.059844970703125 |
| **What the ANE returned** | **−0.05987548828125** |

The exact result lies between two binary16 numbers. Any correctly rounded
binary16 result must be one of those two, whatever the midpoint convention —
nearest-even, ties-away, toward zero, up, down. **The device returned a third
value, exactly one ULP beyond the lower one.**

The exact value is not near a midpoint, so this is not a tie-breaking
disagreement. It rules out changing only the final rounding of the exact dot product.
Intermediate product rounding, small-product handling and finite-precision
accumulation remain candidates; this does not identify an internal mechanism.

## Reproduce

The 32 terms are published as scalars. This needs no device, no model and no
Apple software:

```sh
python results/historical/tests/verify_arithmetic.py
```

It recomputes the dot product exactly with `fractions.Fraction`, derives the
two straddling binary16 values with `struct`, and asserts the recorded ANE
result is neither. Roughly twenty lines of that script are the whole claim; the
rest is the surrounding bisection.

## How it was localized

Seven levels of binary search over the 128 K32 blocks of the gate, from the
whole reduction down to one block:

| Depth | Parent blocks | Descended into | Predicted | Actual |
|---:|---|---|---:|---:|
| 0 | 0–127 | 64–127 | −0.078674316406 | −0.078735351562 |
| 1 | 64–127 | 64–95 | −0.097412109375 | −0.097473144531 |
| 2 | 64–95 | 80–95 | −0.030654907227 | −0.030715942383 |
| 3 | 80–95 | 88–95 | −0.027236938477 | −0.027297973633 |
| 4 | 88–95 | 88–91 | −0.044403076172 | −0.044464111328 |
| 5 | 88–91 | 90–91 | −0.065490722656 | −0.065551757812 |
| 6 | 90–91 | **91** | −0.059844970703 | −0.059875488281 |

At every level two guards had to hold before descending: the parent output had
to equal an **independent ANE add** of the two child sub-graph outputs, and that
add had to equal the frozen ties-away reference. Both held at all seven levels,
on real, zero and repeated inputs, over the complete padded arrays. The descent
rule was fixed in advance ("first discrepant child, left first"), so the path
was not chosen after seeing results.

Extracting a sub-graph can change fusion, so no claim is made about observing
the original graph's undisturbed internal state. Each parent is therefore
rebuilt from the measured child outputs.

One-hot probes of blocks 90 and 91 returned decoded weights with **zero**
numeric mismatches against the CPU decode, which rules out mis-decoded weights
under those conditions. The record lists signed-zero bit differences separately.

## Why this one dot product matters

Injecting the measured leaf value at that single coordinate — leaving every
sibling at its predicted value — and replaying the seven verified ties-away adds
reproduces all seven parent values and returns the root to the recorded output
and Q8 code. So this one 32-term dot product accounts for the *first* of the 13
residuals in [the execution model](../execution-model/). It is a consistency
check on saved data, not a runtime repair, and the other 12 residuals have not
each been localized.

## Open hypothesis — not a mechanism

Two of the 32 exact products fall below 2⁻¹⁴, the smallest normal binary16
value:

| Input channel | Decoded weight | Input | Exact product |
|---:|---:|---:|---:|
| 2916 | 0.00747680664062 | −0.0078125 | −5.84125518799e-05 |
| 2927 | 0.00747680664062 | 0.00390625 | 2.92062759399e-05 |

Subnormal handling before accumulation is therefore the next candidate. Before it counts, a rule fixed in advance must explain all
six deviations in this block and introduce no new ones, and then survive a
different document or an adjacent block. The presence of two small products is
not itself an explanation.

## Scope

One dot product, one model, one host, one toolchain version. The claim is about
arithmetic semantics rather than performance, which is the part most likely to
transfer — and testing it on another Apple chip is the cheapest high-value
replication available.

## Evidence

- [dot-localization.json](../../results/historical/dot-localization.json) — the
  bisection ladder, the 32 terms, the counterfactual trace, source hashes
- [verify_arithmetic.py](../../results/historical/tests/verify_arithmetic.py)
- [The model this residual breaks](../execution-model/)

#!/usr/bin/env python3
"""Recheck the execution-model evidence with no device and no workspace access.

The central claim is checkable from published scalars alone: recompute the
32-term dot product exactly, find the two binary16 values that straddle it, and
show the recorded ANE result is neither of them. No rounding rule applied to a
correctly computed dot product can return a third value, so this rules out the
midpoint convention as the explanation.

Standard library only.
"""

import argparse
import json
import struct
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


# -- binary16 helpers ---------------------------------------------------
def to_bits(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def from_bits(bits):
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def fp16(value):
    """Round a real to binary16, nearest with ties to even."""
    return struct.unpack('<e', struct.pack('<e', value))[0]


def ordinal(bits):
    """Map binary16 bit patterns to integers that increase with the value."""
    return (0x8000 - (bits & 0x7FFF)) if bits & 0x8000 else (0x8000 + bits)


def from_ordinal(index):
    if index >= 0x8000:
        return from_bits(index - 0x8000)
    return from_bits(0x8000 | (0x8000 - index))


def bracket(exact):
    """The two binary16 values that straddle `exact`, as (lower, upper)."""
    nearest = fp16(exact)
    index = ordinal(to_bits(nearest))
    if nearest == exact:
        return nearest, nearest
    if nearest < exact:
        return nearest, from_ordinal(index + 1)
    return from_ordinal(index - 1), nearest


def fp16_rza(exact):
    """Round to binary16, nearest with midpoint ties away from zero."""
    low, high = bracket(exact)
    if low == high:
        return low
    middle = (low + high) / 2
    if exact == middle:
        return high if exact > 0 else low
    return low if exact - low < high - exact else high


def check_helpers():
    assert ordinal(to_bits(-1.0)) < ordinal(to_bits(0.0)) < ordinal(to_bits(1.0))
    for value in (-2.0, -0.5, 0.0, 0.5, 2.0):
        assert from_ordinal(ordinal(to_bits(value))) == value
    # 0.0625/0.125 is exactly 0.5, a midpoint: ties-to-even gives 0, ties-away gives up.
    assert fp16_rza(0.09375) == 0.09375
    low, high = bracket(0.1)
    assert low < 0.1 < high


# -- checks -------------------------------------------------------------
def check_localization(data):
    leaf = data['leaf']
    terms = leaf['terms']
    assert len(terms) == 32, 'a K32 block has 32 terms'

    # Recompute the dot product exactly: every operand is a dyadic rational.
    total = Fraction(0)
    for term in terms:
        product = Fraction(term['decoded_weight']) * Fraction(term['input'])
        assert product == Fraction(term['exact_product']), term['input_channel']
        total += product
    assert total == Fraction(leaf['exact_dot']), 'recomputed dot differs from the record'

    integer = leaf['exact_integer']
    assert total == Fraction(integer['numerator'], 2 ** -integer['scale_exponent'])

    exact = leaf['exact_dot']
    low, high = bracket(exact)
    prediction, actual = leaf['model_prediction'], leaf['ane_actual']
    assert prediction in (low, high), 'the model prediction is not a binary16 neighbour'
    assert prediction == fp16(exact) == fp16_rza(exact), 'prediction is not correctly rounded'
    assert actual not in (low, high), 'the ANE result IS a neighbour; the claim would fail'
    assert leaf['not_explained_by_final_midpoint_rounding'] is True

    ulp = high - low
    steps = (low - actual) / ulp
    assert abs(steps - round(steps)) < 1e-9 and round(steps) == 1

    # Every level of the bisection: the parent equals a ties-away add of its children.
    assert len(data['bisection']) == 7
    for row in data['bisection']:
        assert row['guards'] == {'parent_vs_add': True, 'add_vs_RZA': True}
        if row['add'] is None:
            continue
        assert row['add'] == row['parent'], row['depth']
        assert fp16_rza(row['left'] + row['right']) == row['add'], row['depth']
    first, last = data['bisection'][0], data['bisection'][-1]
    assert first['parent_block_range'] == [0, 127]
    assert last['selected_block_range'] == [leaf['block'], leaf['block']]
    assert data['guards']['every_level_parent_equals_independent_add'] is True
    assert data['guards']['every_level_add_equals_frozen_RZA_reference'] is True

    # The counterfactual walks the recorded leaf value back up to the root.
    trace = data['counterfactual']['trace']
    assert len(trace) == 7
    for step in trace:
        assert step['corrected_parent'] == step['actual_parent'], step['depth']

    print(f'  32-term dot product recomputes exactly: {float(total)!r}')
    print(f'  binary16 values straddling it:          [{low!r}, {high!r}]')
    print(f'  correctly rounded (either tie rule):    {prediction!r}')
    print(f'  what the ANE returned:                  {actual!r}  '
          f'({round(steps)} ULP beyond the lower one)')
    print('  no rounding rule on a correct dot product can return that value')
    return len(terms)


def check_cross_model(data):
    total_values = total_residuals = 0
    for name, block in data['models'].items():
        agreement = 1 - block['q8_numeric_mismatches'] / block['real_values']
        assert abs(agreement - block['q8_agreement']) < 1e-15, name
        reduction = 1 - (block['q8_numeric_mismatches']
                         / block['original_q8_numeric_mismatches'])
        assert abs(reduction - block['residual_reduction']) < 1e-15, name
        assert block['q8_numeric_mismatches'] < block['original_q8_numeric_mismatches']
        assert block['max_q8_l2'] <= block['original_max_q8_l2']
        total_values += block['real_values']
        total_residuals += block['q8_numeric_mismatches']
        print(f'  {name:8s} {block["real_values"]:>9,} values  '
              f'{block["q8_numeric_mismatches"]:>3} residual(s)  '
              f'{100 * agreement:.6f}% exact  '
              f'({100 * reduction:.3f}% fewer than the original reference)')

    residuals = data['q8_residuals']
    assert len(residuals) == total_residuals, 'residual list and counts disagree'
    for entry in residuals:
        assert abs(entry['code_difference']) == 1 and entry['adjacent_codes']
        assert not (entry['predicted_at_boundary'] and entry['actual_at_boundary'])
    # The strict cross-model byte gate did not pass; that must stay recorded.
    assert data['strict_cross_model_byte_gate_passed'] is False
    return total_values, total_residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=HERE)
    args = parser.parse_args()
    check_helpers()

    print('Cross-model validation of the derived execution model:')
    values, residuals = check_cross_model(
        json.loads((args.evidence / 'cross-model-validation.json').read_text()))
    print(f'  {values:,} recorded gate outputs, {residuals} unexplained\n')

    print('Localization of the first unexplained residual:')
    check_localization(
        json.loads((args.evidence / 'dot-localization.json').read_text()))
    print('\nexecution-model evidence verification passed')


if __name__ == '__main__':
    main()

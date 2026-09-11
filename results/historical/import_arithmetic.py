"""Import the execution-model evidence from the closed research workspace.

Two experiments are imported: the cross-model validation of a derived
arithmetic model for group-quantized matmul, and the bisection that localised
its first unexplained residual to a single 32-term dot product.

Only derived scalars are copied — per-model counts, residual coordinates, the
32 terms of one dot product and the bisection ladder. No weight tensor, no
activation array, no model file and no absolute path is serialised. Like
`extract.py`, this imports no array, model or device library and reruns
nothing.
"""

import argparse
import hashlib
import json
from pathlib import Path

CROSS_MODEL = 'results/g1-generalize-20260910'
LOCALIZATION = 'results/g1-localize-20260910'

# Fields copied verbatim from each model's summary block.
MODEL_FIELDS = ('real_values', 'pre_numeric_mismatches', 'q8_numeric_mismatches',
                'original_q8_numeric_mismatches', 'max_pre_l2', 'max_q8_l2',
                'original_max_q8_l2', 'all_full_q8_bytes_equal')
LADDER_FIELDS = ('pred_left', 'pred_right', 'pred_parent', 'parent', 'left', 'right', 'add')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, rel):
    return json.loads((root / rel).read_text())


def sources(root, *rels):
    """Record every file this import read, by workspace-relative path and hash."""
    return {rel: sha(root / rel) for rel in rels}


def cross_model(root):
    summary = load(root, CROSS_MODEL + '/summary.json')
    residuals = load(root, CROSS_MODEL + '/q8-residual-details.json')
    models = {name: {k: block[k] for k in MODEL_FIELDS}
              for name, block in summary['models'].items()}
    for name, block in models.items():
        block['q8_agreement'] = 1 - block['q8_numeric_mismatches'] / block['real_values']
        block['residual_reduction'] = (
            1 - block['q8_numeric_mismatches'] / block['original_q8_numeric_mismatches'])
    return {
        'historical_import': True,
        'scope': 'A derived arithmetic model for group-quantized matmul, compared with '
                 'recorded ANE output on two real models. Gate outputs only; not a '
                 'full-model, quality or performance result.',
        'formula': 'Per 32 input channels: decode the original Q4_0 weights to FP16, take '
                   'the exact dot product, round to nearest with midpoint ties away from '
                   'zero, apply the same rounding at every add of the original balanced '
                   'tree, then apply the original group-32 input and output QDQ.',
        'device_calls': summary['new_ANE_calls'],
        'all_replays_equal_previous': summary['all_replays_equal_previous'],
        'independent_verification_passed': summary['all_independent_verification_passed'],
        'strict_cross_model_byte_gate_passed': summary['strict_all_models_Q8_bytes_gate_passed'],
        'models': models,
        'q8_residuals': residuals,
        'sources': sources(root, CROSS_MODEL + '/summary.json',
                           CROSS_MODEL + '/q8-residual-details.json',
                           CROSS_MODEL + '/PROTOCOL.json'),
    }


def localization(root):
    summary = load(root, LOCALIZATION + '/summary.json')
    terms = load(root, LOCALIZATION + '/leaf/target-32-terms.json')
    counterfactual = load(root, LOCALIZATION + '/leaf/single-coordinate-counterfactual.json')

    ladder, read = [], [LOCALIZATION + '/summary.json',
                        LOCALIZATION + '/leaf/target-32-terms.json',
                        LOCALIZATION + '/leaf/single-coordinate-counterfactual.json',
                        LOCALIZATION + '/PROTOCOL.json']
    for depth in range(7):
        rel = f'{LOCALIZATION}/depth{depth}/decision.json'
        decision = load(root, rel)
        read.append(rel)
        points = decision['target_points']
        ladder.append({
            'depth': decision['depth'],
            'parent_block_range': [decision['lo'], decision['lo'] + decision['count'] - 1],
            'discrepant_side': decision['side'],
            'selected_block_range': [decision['selected']['lo'],
                                     decision['selected']['lo']
                                     + decision['selected']['count'] - 1],
            'guards': decision['guards'],
            **{k: points.get(k) for k in LADDER_FIELDS},
        })

    return {
        'historical_import': True,
        'scope': "A seven-level bisection of one model's first unexplained Q8 residual, "
                 'down to a single 32-term dot product. The device was not repaired and no '
                 'reference or quality result was changed.',
        'target': {
            'model': 'qwen8b',
            'document': 'chinese/tokenize',
            'output_channel': summary['target_channel'],
            'token_position': summary['target_position'],
        },
        'guards': {
            'every_level_parent_equals_independent_add': summary[
                'all7_full_array_composition_guards_passed'],
            'every_level_add_equals_frozen_RZA_reference': summary[
                'all7_ANE_add_vs_RZA_guards_passed'],
            'onehot_weight_numeric_mismatches': summary['onehot_weight_numeric_mismatches'],
        },
        'bisection': ladder,
        'leaf': {
            'block': summary['selected_block'],
            'input_channels': summary['input_channels'],
            'exact_dot': summary['target_exact_dot'],
            'exact_integer': {'numerator': -1004010, 'scale_exponent': -24},
            'model_prediction': summary['target_predicted_dot'],
            'ane_actual': summary['target_actual_dot'],
            'not_explained_by_final_midpoint_rounding': summary[
                'target_error_not_final_midpoint_rounding'],
            'terms': terms,
        },
        'counterfactual': counterfactual,
        'open_hypothesis': {
            'statement': 'Two of the 32 exact products fall below the smallest normal '
                         'binary16 value 2^-14, so subnormal handling before accumulation '
                         'is the next candidate rule.',
            'status': 'unverified',
            'requirement': 'A pre-registered rule must explain all six deviations in this '
                           'block and introduce no new ones before it counts as a mechanism.',
        },
        'unlocalized_residuals': 12,
        'device_calls': summary['new_ANE_calls'],
        'sources': sources(root, *read),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, build in (('cross-model-validation.json', cross_model),
                        ('dot-localization.json', localization)):
        path = out / name
        if path.exists():
            raise SystemExit(f'refusing to overwrite existing output: {path}')
        path.write_text(json.dumps(build(root), indent=2) + '\n')
        print(f'wrote {path}')


if __name__ == '__main__':
    main()

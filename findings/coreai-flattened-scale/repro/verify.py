"""Recompute the grouped-LUT comparison without Core AI or a device."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

def require(condition, name):
    if not condition:
        raise ValueError(name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def references():
    data = ROOT / 'fixture'
    codes = np.load(data / 'group_codes.npy', allow_pickle=False)
    scales = np.load(data / 'group_scales.npy', allow_pickle=False)
    input_scales = np.load(data / 'group_input_scales.npy', allow_pickle=False)
    x = np.fromfile(data / 'group_input.raw', dtype='<f2').reshape(64, 64)
    require(codes.shape == (64, 64, 1, 1), 'fixture.codes.shape')
    require(scales.shape == (64, 2, 1, 1), 'fixture.scales.shape')
    require(np.array_equal(x, np.eye(64, dtype=np.float16)), 'fixture.identity_input')
    require(np.array_equal(codes, np.random.default_rng(20260910).integers(-8, 8, codes.shape, dtype=np.int8)), 'fixture.seeded_codes')
    wanted_scales = np.array([[.125, .25] if i % 2 == 0 else [.25, .125] for i in range(64)], dtype=np.float16)
    require(np.array_equal(scales.reshape(64, 2), wanted_scales), 'fixture.group_scales')
    require(np.array_equal(input_scales, np.array([.25] * 32 + [.5] * 32, dtype=np.float16)), 'fixture.input_scales')
    q = x.astype(np.float64) / input_scales[:, None]
    require(np.array_equal(q, np.rint(q)) and abs(q).max() <= 127, 'fixture.exact_input_qdq')
    weights = codes.reshape(64, 64).astype(np.float64)
    intended = ((weights * np.repeat(scales.reshape(64, 2), 32, axis=1)) @ x.astype(np.float64)).astype(np.float16)
    flattened = ((weights * scales.reshape(-1)[:64, None]) @ x.astype(np.float64)).astype(np.float16)
    require(np.array_equal(intended.reshape(-1), np.load(data / 'reference-group.npy', allow_pickle=False).reshape(-1)), 'fixture.intended_reference')
    require(np.array_equal(flattened.reshape(-1), np.load(data / 'reference-group-wrong.npy', allow_pickle=False).reshape(-1)), 'fixture.flattened_reference')
    return intended, flattened

def compare(actual, expected):
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    denominator = np.linalg.norm(e)
    return {'numeric_mismatches': int(np.count_nonzero(a != e)),
            'relative_L2': float(np.linalg.norm(a - e) / denominator) if denominator else float(np.linalg.norm(a - e))}

def verify(outputs=None):
    record = json.loads((ROOT / 'recorded/results.json').read_text())
    require(sorted(case['case_id'] for case in record['runs']) == ['coreai-group-native64', 'coreai-group-split32'], 'recorded.cases')
    for name in record['bundled_source_files']:
        require(sha(ROOT / 'frozen/ane_scope' / name) == record['source_identity'][name], 'source.' + name)
    for name, digest in json.loads((ROOT / 'fixture/manifest.json').read_text())['files'].items():
        require(sha(ROOT / 'fixture' / name) == digest, 'fixture.sha256.' + name)
    intended, flattened = references()
    summary = {}
    for case in record['runs']:
        case_id = case['case_id']
        phases = ['control_original', 'control_zero', 'control_negative', 'control_original_repeat']
        require([r['phase'] for r in case['numerical']['comparisons']] == phases, case_id + '.intended.phases')
        require([r['phase'] for r in case['flattened-scale-hypothesis']['comparisons']] == phases, case_id + '.flattened.phases')
        directory = (Path(outputs) if outputs else ROOT / 'recorded') / case_id
        results = []
        for saved, wrong in zip(case['numerical']['comparisons'], case['flattened-scale-hypothesis']['comparisons']):
            phase = saved['phase']
            path = directory / saved['output_file']
            if outputs is None:
                require(sha(path) == saved['output_sha256'], case_id + '.' + phase + '.sha256')
            actual = np.fromfile(path, dtype='<f2')
            require(actual.size == 4096 and np.isfinite(actual).all(), case_id + '.' + phase + '.finite_shape')
            actual = actual.reshape(64, 64)
            sign = 0 if phase == 'control_zero' else -1 if phase == 'control_negative' else 1
            observed = compare(actual, intended * sign)
            predicted = compare(actual, flattened * sign)
            if outputs is None:
                for label, measured, published in [('intended', observed, saved), ('flattened', predicted, wrong)]:
                    require(measured['numeric_mismatches'] == published['numeric_mismatches'], case_id + '.' + phase + '.' + label + '.mismatches')
                    require(abs(measured['relative_L2'] - published['relative_L2']) < 1e-12, case_id + '.' + phase + '.' + label + '.relative_L2')
            results.append({'phase': phase, 'intended': observed, 'flattened': predicted})
        require((directory / 'control_original.raw').read_bytes() == (directory / 'control_original_repeat.raw').read_bytes(), case_id + '.repeat_byte_equal')
        summary[case_id] = results
    return {'mode': 'new_outputs' if outputs else 'recorded_evidence', 'comparisons': summary}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, help='Compare a new run; report changes instead of requiring the historical discrepancy.')
    args = parser.parse_args()
    print(json.dumps(verify(args.outputs), indent=2))

"""Import the G5 weight-free attention precision and chunking evidence.

Relative L2 results are copied as scalar fields with their source paths; the
FP16 inputs and device outputs stay in the workspace. Per-operation timings
travel with the bundle so the speed comparisons can be recomputed. No array,
model or device library is imported, and no absolute path is serialised.
"""

import argparse
import gzip
import json
from pathlib import Path

BASE = 'workplans/g5-long-context-attention-20260912'
REVIEW = 'workplans/g5-independent-review-20260912'


class Reader:
    def __init__(self, root):
        self.root, self.sources = Path(root), set()

    def json(self, path):
        data = (self.root / path).read_bytes()
        self.sources.add(path)
        return json.loads(data)


def metric(block):
    return {k: block[k] for k in ('relative_l2', 'row_relative_l2_max', 'max_abs')}


def by_arm(cases, fixtures):
    """One entry per arm and fixture, in fixture order; replays must agree."""
    seen = {}
    for case in cases:
        if case['fixture'] in fixtures:
            entry = metric(case['numeric'])
            key = case['arm'], case['fixture']
            assert seen.setdefault(key, entry) == entry, key
    arms = sorted({arm for arm, _ in seen})
    return {arm: [seen[arm, f] for f in fixtures] for arm in arms}


SEEDS = ['0-random-n4096', '1-random-n4096', '2-random-n4096']


def numerics(read):
    dense = read.json(f'{BASE}/runs/p2-q64-analysis-r2/RESULT.json')
    post = read.json(f'{BASE}/runs/post-pv-q64-analysis-r1/RESULT.json')
    amplitude = read.json(f'{BASE}/runs/v3-amplitude-analysis-r1/RESULT.json')
    scale = read.json(f'{BASE}/runs/pv-scale-analysis-r1/RESULT.json')
    inverse = read.json(f'{BASE}/runs/pv-inverse-analysis-r1/RESULT.json')
    constant = read.json(f'{BASE}/runs/v4-pv-analysis-r1/RESULT.json')
    review = read.json(f'{REVIEW}/RESULT.json')
    cpu = read.json(f'{REVIEW}/CPU-EXPRESSION.json')
    factors = {}
    for row in scale['rows']:
        factors.setdefault(str(row['factor']), set()).add(row['relative_l2'])
    assert all(len(v) == 1 for v in factors.values()), 'repeated scale rows disagree'
    inverse_values = {row['relative_l2'] for row in inverse['rows']}
    assert len(inverse_values) == 1
    constants = {}
    for row in constant['rows']:
        entry = {'relative_l2': row['attention_metrics']['legacy']['relative_l2'],
                 'output_min': row['output_minmax'][0], 'output_max': row['output_minmax'][1]}
        key = row['arm'] + '/' + row['fixture']
        assert constants.setdefault(key, entry) == entry, 'replayed rows disagree'
    diagnostics = review['rounding_diagnostics_row0']
    return {
        'random_three_seeds': {**by_arm(dense['cases'], SEEDS),
                               **{'post_pv_' + k: v for k, v in
                                  by_arm(post['cases'], SEEDS).items()}},
        'near_zero_half': {arm: v[0] for arm, v in by_arm(amplitude['cases'],
                                                          ['near_zero_half']).items()},
        'isolated_pv': {k: review['PV'][k] for k in
                        ('relative_l2', 'same_bytes_as_zero_QK_dense', 'P_range_nonzero')},
        'pv_scale_cpu_division': {k: v.pop() for k, v in factors.items()},
        'pv_scale_64_device_division': inverse_values.pop(),
        'constant_v_pv': constants,
        'cpu_fp16_same_expression_seed0': {k: v['relative_l2'] for k, v in cpu['results'].items()},
        'simple_fp16_models_row0': {name: {side: block[side]['relative_l2']
                                           for side in ('vs_truth', 'vs_device')}
                                    for name, block in diagnostics.items()},
    }


def timings(read):
    rounds = {'dense_vs_post_pv_b1024': 'post-pv-bench-process{}-r1',
              'post_pv_b4096_vs_b1024': 'v4-base-bench-process{}-r1'}
    out = {}
    for name, pattern in rounds.items():
        blocks = []
        for process in range(3):
            for block in read.json(f'{BASE}/runs/{pattern.format(process)}/host/TIMINGS.json'):
                assert block['completed_calls'] == [block['calls_per_operation']] * block['repeats']
                blocks.append({k: block[k] for k in ('process', 'pair', 'arm', 'mode',
                                                     'calls_per_operation', 'operation_ns')})
        out[name] = blocks
    return out


def reported(read):
    v2 = read.json(f'{BASE}/runs/post-pv-speed-summary-r1/RESULT.json')
    v4 = read.json(f'{BASE}/runs/v4-base-speed-summary-r1/RESULT.json')
    return {
        'dense_vs_post_pv_b1024': {mode: {arm: block['median_pair_speedup']
                                          for arm, block in arms.items()}
                                   for mode, arms in v2['comparison'].items()},
        'post_pv_b4096_vs_b1024': {mode: v4['comparisons'][mode]['post4096_over_post1024']['median']
                                   for mode in v4['comparisons']},
        'dense_round_block_means': [
            {'process': process['process'], 'pair': pair['pair'], 'mode': pair['mode'],
             'arm': arm, 'ms_per_operation': value}
            for process in v2['processes'] for pair in process['pairs']
            for arm, value in sorted(pair['ms_per_operation'].items())],
        'absolute_ms': {mode: {arm: block['median_block_mean_ms'] for arm, block in arms.items()}
                        for mode, arms in v4['absolute_ms'].items()},
    }


def write(path, text):
    with path.open('x') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new bundle directory')
    args = parser.parse_args()
    read = Reader(args.source_root)
    config = read.json(f'{BASE}/CONFIG.json')
    asset = read.json(f'{BASE}/runs/v3-post4096-export-r1/kv_unrolled/ASSET.json')
    args.output.mkdir(parents=True)
    evidence = {
        'schema_version': 1, 'historical_import': True, 'fresh_run': False,
        'scope': 'Weight-free attention graphs with synthetic FP16 Q, K and V; relative L2 '
                 'against an independent FP64 reference. No real QKV, model layer, token/s '
                 'or quality result.',
        'shape': {**config['structure'], **config['primary']},
        'screen': config['numeric'],
        'packages': asset['packages'],
        'numerics': numerics(read),
        'reported': reported(read),
    }
    write(args.output / 'evidence.json', json.dumps(evidence, indent=2) + '\n')
    with gzip.GzipFile(args.output / 'timings.json.gz', 'xb', mtime=0) as handle:
        handle.write(json.dumps(timings(read), separators=(',', ':')).encode())
    provenance = {
        'importer': 'results/historical/import_g5.py',
        'sources': sorted(read.sources),
        'transformations': [
            'relative L2, row maximum and maximum absolute error copied per case; arrays omitted',
            'replayed rows collapsed after checking that they agree',
            'per-operation durations kept for every timed block; output files omitted',
        ],
    }
    write(args.output / 'provenance.json', json.dumps(provenance, indent=2) + '\n')
    print('wrote', args.output)


if __name__ == '__main__':
    main()

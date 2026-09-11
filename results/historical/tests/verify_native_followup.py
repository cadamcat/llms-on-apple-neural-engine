"""Recompute the native follow-up from its bundled per-call scalar observations."""
import json
import math
import statistics
from pathlib import Path


def quantile(values, p):
    x = sorted(values)
    i = (len(x) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return x[lo] + (x[hi] - x[lo]) * (i - lo)


def verify(data):
    assert data['historical_import'] is True
    assert data['evidence_checks_passed'] is True
    assert data['formal_three_process_acceptance'] is False
    assert data['long_term_residency_proven'] is False
    assert data['power_thermal_coexistence_measured'] is False
    medians = {}
    for arm, block in data['timing_arms'].items():
        assert block['independent_processes'] == 1
        assert len(block['calls']) == 80
        for row in block['reported_rows']:
            n = row['tokens']
            calls = [c for c in block['calls'] if c['positions'] == n]
            warm = [c for c in calls if c['phase'] == 'warmup']
            measured = [c for c in calls if c['phase'] == 'measure']
            assert len(warm) == row['warmup'] == 10
            assert len(measured) == row['measurements'] == 30
            assert sorted(c['index'] for c in calls) == list(range(40))
            assert all(c['numerical_passed'] for c in calls)
            for c in calls:
                assert c['outer_ns'] > 0 and c['worker_ns'] > 0
                assert len(c['output_sha256']) == 64
            for i in {c['input_index'] for c in calls}:
                assert len({c['output_sha256'] for c in calls if c['input_index'] == i}) == 1
            for field, key, p in [('outer_ns', 'p50_outer_ms', .5),
                                   ('outer_ns', 'p95_outer_ms', .95),
                                   ('worker_ns', 'p50_worker_ms', .5)]:
                actual = quantile([c[field] for c in measured], p) / 1e6
                assert math.isclose(actual, row[key], rel_tol=1e-12, abs_tol=1e-9), (arm, key)
            medians[arm, n] = row['p50_outer_ms']
    for arm, block in data['memory'].items():
        assert block['request_positions'] == 4096 and block['requests'] == 32
        assert block['long_term_residency_proven'] is False
        points = {p['label']: p for p in block['checkpoints']}
        for field in ('worker_footprint', 'controller_footprint'):
            delta = points['32'][field] - points['16'][field]
            assert delta == block['growth'][field]['delta_bytes']
            assert delta / (16 * 4096 / 64) == block['growth'][field]['bytes_per_64_position_block']
    assert data['memory']['C-256']['memory_gate_passed'] is True
    assert data['memory']['C-64']['memory_gate_passed'] is True
    assert data['memory']['A-256']['memory_gate_passed'] is False
    assert data['memory']['C-256']['growth']['worker_footprint']['delta_bytes'] == 16384
    assert data['memory']['C-256']['growth']['controller_footprint']['delta_bytes'] == 114688
    for n in (1024, 4096):
        c, g, c64 = (medians[a, n] for a in ('C-256', 'G-0', 'C-64'))
        print(f'{n} positions: C256 {n*1000/c:.2f}/s, GPU {n*1000/g:.2f}/s; '
              f'GPU/C256 speed ratio {c/g:.4f}; C256 latency reduction {1-c/c64:.6%}')


if __name__ == '__main__':
    verify(json.loads((Path(__file__).resolve().parents[1] / 'native-mlp-followup.json').read_text()))
    print('Native follow-up timings and short-memory observations verified; long residency untested.')

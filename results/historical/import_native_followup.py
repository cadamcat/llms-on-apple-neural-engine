"""Import the later native MLP comparison, without changing the Python-era record.

Only scalar timings, memory checkpoints and source identities travel with this
record. No model, weight, activation or device library is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

BASE = 'results/g1-tile-20260910'


def build(root):
    sources = {}

    def read(rel):
        path = BASE + '/' + rel
        raw = (root / path).read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    review = read('review/verification.json')
    protocol = read('BENCH-PROTOCOL.json')
    prepared = read('prepared.json')
    resources = read('resource-closure.json')
    arms = {}
    for name in ('C-64', 'C-256', 'G-0'):
        raw = read('throughput/' + name + '/calls.json')
        arms[name] = {
            'independent_processes': 1,
            'calls': [{'positions': c['tokens'], 'index': c['index'], 'phase': c['phase'],
                       'input_index': c['input_index'], 'output_sha256': c['output_hash'],
                       'numerical_passed': c['numerical_passed'],
                       'outer_ns': c['meta']['outer_ns'], 'worker_ns': c['meta']['worker_ns']}
                      for c in raw],
            'reported_rows': [{k: v for k, v in row.items() if k in (
                'tokens', 'stage_calls_per_request_from_selected_tile', 'warmup',
                'measurements', 'p50_outer_ms', 'p95_outer_ms', 'p50_worker_ms')}
                for row in review['timings'][name]],
        }
    memory = {}
    for name in ('C-64', 'C-256', 'A-256'):
        item = read('memory/' + name + '/verification.json')
        memory[name] = {
            'request_positions': 4096, 'requests': 32,
            'numerical_checks_passed': item['numerical_checks_passed'],
            'memory_gate_passed': item['memory_gate_passed'],
            'long_term_residency_proven': item['long_term_residency_proven'],
            'growth': item['growth'],
            'checkpoints': [{k: row[k] for k in (
                'label', 'worker_footprint', 'controller_footprint')} for row in item['memory']],
        }
    return {
        'schema_version': 1, 'historical_import': True,
        'scope': 'Later G1-TILE native Swift four-stage first-layer MLP with a Python '
                 'controller and fixed receive buffers. C is W4A16; G is MLX W4A16. '
                 'Positions are component workload, not full-model tokens.',
        'relationship_to_prefill': 'A separate later configuration and process set. '
                 'Does not complete the older 3-of-9 Python-era protocol or replace its numbers.',
        'toolchain': {k: prepared['observations'][k]['stdout'].strip()
                      for k in ('xcode', 'swift', 'system')},
        'sdk': 'macOS 27.0',
        'host_identity': {k: prepared[k] for k in (
            'binary_sha256', 'worker_sha256', 'wire_sha256', 'GPU_source_sha256')},
        'timing_arms': arms, 'memory': memory,
        'resources': {k: resources[k] for k in (
            'owned_rss_peak_bytes', 'free_min_percent_during_experiments',
            'swap_min_MiB', 'swap_max_MiB', 'rss_sample_interval_seconds',
            'no_owned_processes_remaining')},
        'evidence_checks_passed': review['evidence_checks_passed'],
        'formal_three_process_acceptance': review['formal_three_process_acceptance'],
        'long_term_residency_proven': False,
        'power_thermal_coexistence_measured': False,
        'sources': sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    text = json.dumps(build(args.source_root), indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(text)
    print('wrote', args.output)


if __name__ == '__main__':
    main()

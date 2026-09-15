"""Import the ANE-versus-GPU prefill comparison from the closed research workspace.

Three paths ran the same first-layer MLP from the same source Q4_0 weights: the
Neural Engine with A8W4, the Neural Engine with W4A16, and the GPU through MLX.
Only derived scalars are copied — latency percentiles, ratios, process
footprints and the per-call IOSurface growth. No weight tensor, activation
array or model file is published, and this imports no array, model or device
library.

The run stopped early for memory safety, so the recorded incompleteness travels
with the numbers rather than being smoothed away.
"""

import argparse
import json
from pathlib import Path

PREFILL = 'results/g1-prefill-20260910'
ROW_FIELDS = ('p50_outer_ms', 'p95_outer_ms', 'p50_worker_ms',
              'effective_positions_per_second', 'dense_equivalent_Top_s_worker')
ENGINES = {
    'A': 'Neural Engine, group-32 A8W4',
    'C': 'Neural Engine, W4A16',
    'G': 'GPU through MLX 0.32.2, Q4 weights with FP16 activations',
}


def load(root, rel):
    return json.loads((root / rel).read_text())


def build(root):
    review = load(root, PREFILL + '/review/verification.json')
    external = load(root, PREFILL + '/external-sources.json')

    engines = {}
    for name, description in ENGINES.items():
        block = review['engines'][name]
        engines[name] = {
            'description': description,
            'rows': {positions: {k: row[k] for k in ROW_FIELDS}
                     for positions, row in block['rows'].items()},
            'footprint_bytes': block['footprint_bytes'],
            'numeric_mismatches_by_stage': block['numeric'],
        }

    # One diagnostic variant per attempted mitigation; all leak the same amount.
    leak = {name: variant['IOSurface_growth_bytes_per_call']
            for name, variant in review['memory_diagnostic'].items()
            if variant.get('IOSurface_growth_bytes_per_call') is not None}

    return {
        'historical_import': True,
        'scope': 'One first-layer MLP (gate/up, GELU_tanh, product, down) from the same '
                 'source Q4_0 weights, with an FP16 input and output interface. The 1024 '
                 'and 4096 position cases are formed by looping saved real activations: '
                 'this is NOT a full-model context or an end-to-end prefill.',
        'implementation_caveat': 'These numbers describe the current Python, 64-position '
                                 'chunked implementation on this host. They are not a '
                                 'measurement of what the accelerator can do in principle, '
                                 'and the planned native Swift comparison did not run.',
        'engines': engines,
        'ANE_over_GPU_ratios': review['A_G_ratios'],
        'completeness': {
            'timed_processes_completed': review['checks']['timed_processes_completed'],
            'timed_processes_required': review['checks']['required_timed_processes'],
            'samples_recomputed': review['checks']['recomputed_all_360_samples'],
            'assets_loaded_once': review['checks']['assets_loaded_once'],
            'output_repeat_hashes_stable': review['checks']['output_repeat_hashes_stable'],
            'stop_reason': 'IOSurface growth per call; remaining repeats and the long '
                           'windows were not run',
        },
        'gates': review['gates'],
        'iosurface_growth_bytes_per_call': leak,
        'gate_output_bytes': 15360 * 64 * 2,
        'gate_growth_equals_output_bytes': review['checks'][
            'gate_IOSurface_growth_equals_output_bytes'],
        'external_sources': external['sources'],
        'sources': sorted((
            PREFILL + '/review/verification.json',
            PREFILL + '/external-sources.json',
            PREFILL + '/PERF-PROTOCOL.json',
            PREFILL + '/placement-ANE.json',
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'refusing to overwrite existing output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(Path(args.source_root).resolve()), indent=2) + '\n')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()

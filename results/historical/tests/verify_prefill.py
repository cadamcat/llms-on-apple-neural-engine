#!/usr/bin/env python3
"""Recheck the ANE-versus-GPU prefill comparison, with no device access.

Recomputes every ratio from the recorded percentiles, checks the per-call
IOSurface growth against the gate output size, and asserts that the recorded
incompleteness is still there — a run that stopped early must keep saying so.

Standard library only.
"""

import argparse
import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / 'ane-vs-gpu-prefill.json'
# Positions that loop saved activations; 1 is a shape diagnostic only.
REAL_POSITIONS = ('64', '1024', '4096')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path, nargs='?', default=DEFAULT)
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text())
    assert data['historical_import'] is True

    ane, w4a16, gpu = (data['engines'][k] for k in ('A', 'C', 'G'))
    print('Same first-layer MLP, same source Q4_0 weights, one process per path.')
    print('Speed, in positions per second (higher is faster):')
    print(f'  {"positions":>9}  {"ANE A8W4":>12}  {"ANE W4A16":>12}  {"MLX GPU":>12}   GPU faster by')
    speed = lambda block, key: block['rows'][key]['effective_positions_per_second']
    for positions in REAL_POSITIONS:
        a = ane['rows'][positions]['p50_outer_ms']
        g = gpu['rows'][positions]['p50_outer_ms']
        recorded = data['ANE_over_GPU_ratios'][positions]['p50_outer_ms']
        assert abs(a / g - recorded) < 1e-9, positions
        assert a > g, f'{positions}: the GPU was not faster; the claim would change'
        # Speed and latency must tell the same story.
        assert abs(speed(gpu, positions) / speed(ane, positions) - a / g) < 1e-6, positions
        print(f'  {positions:>9}  {speed(ane, positions):>10,.0f}/s  '
              f'{speed(w4a16, positions):>10,.0f}/s  {speed(gpu, positions):>10,.0f}/s   '
              f'{a / g:>6.2f}×')

    # The ANE path does not speed up with batch; the GPU does.
    ane_speeds = [speed(ane, p) for p in REAL_POSITIONS]
    assert max(ane_speeds) / min(ane_speeds) < 1.1, 'the ANE path is no longer flat'
    assert speed(gpu, '1024') > 1.5 * speed(gpu, '64'), 'the GPU no longer scales with batch'
    print(f'\n  the ANE holds {min(ane_speeds):,.0f}-{max(ane_speeds):,.0f} positions per '
          f'second at every size; the GPU climbs from {speed(gpu, "64"):,.0f} to '
          f'{speed(gpu, "1024"):,.0f}')

    # The gap is not mainly inter-process overhead: it survives on worker time.
    for positions in REAL_POSITIONS:
        inner = (ane['rows'][positions]['p50_worker_ms']
                 / gpu['rows'][positions]['p50_worker_ms'])
        assert abs(inner - data['ANE_over_GPU_ratios'][positions]['p50_worker_ms']) < 1e-9
        assert inner > 1

    # A8 activations bought nothing measurable at the largest size.
    a4k = ane['rows']['4096']['p50_outer_ms']
    c4k = w4a16['rows']['4096']['p50_outer_ms']
    assert abs(c4k - a4k) / a4k < 0.02, 'W4A16 and A8W4 are no longer within 2%'
    print(f'\n  W4A16 is within {100 * abs(c4k - a4k) / a4k:.1f}% of A8W4 at 4096 positions: '
          f'8-bit activations bought no measurable speed here')

    # Every attempted mitigation leaked exactly one gate output per call.
    expected = data['gate_output_bytes']
    assert expected == 15360 * 64 * 2
    for variant, growth in data['iosurface_growth_bytes_per_call'].items():
        assert growth == expected, f'{variant}: {growth} != {expected}'
    assert data['gate_growth_equals_output_bytes'] is True
    print(f'  IOSurface grows {expected:,} bytes per call ({expected / 2 ** 20:.3f} MiB) in '
          f'all {len(data["iosurface_growth_bytes_per_call"])} diagnostic variants,')
    print('  which is exactly one gate output')

    footprint = {k: data['engines'][k]['footprint_bytes']['after-footprint']
                 for k in ('A', 'C', 'G')}
    assert footprint['A'] > 8 * footprint['G'] and footprint['C'] > 8 * footprint['G']
    print(f'  process footprint after measurement: ANE {footprint["A"] / 2 ** 30:.2f} GiB, '
          f'GPU {footprint["G"] / 2 ** 30:.2f} GiB')

    # The run stopped early. That must never quietly disappear.
    done = data['completeness']['timed_processes_completed']
    need = data['completeness']['timed_processes_required']
    assert done < need, 'completeness record no longer shows the early stop'
    assert data['completeness']['samples_recomputed'] is True
    for gate in ('energy', 'coexistence'):
        assert data['gates'][gate].startswith('undetermined')
    assert data['gates']['resident_runtime'].startswith('not qualified')
    print(f'\n  {done} of {need} planned timed processes completed; energy, coexistence and '
          f'resident runtime remain undetermined')
    print('\nANE-versus-GPU prefill verification passed')


if __name__ == '__main__':
    main()

"""Recompute G4 A speed, component energy and disk observations from the portable bundle."""
import gzip
import json
import math
import statistics
from pathlib import Path

from g3.evidence import BASE as G3_BASE, derive as derive_g3, model_work, read_rows
from g3.power import decode_power, endpoint, energy_window, response
from g3.protocol import require
from .power import integrate
from .protocol import validate_request

ROOT = Path(__file__).resolve().parents[2]
BASE = 'results/historical/g4a-qwen3-4b'
LAGS = ('0', '1', '2', '5')
GiB = 2 ** 30


def load(bundle, name):
    return json.loads((bundle / name).read_text())


def close(a, b, name, rel=1e-9):
    require(math.isclose(a, b, rel_tol=rel, abs_tol=1e-9), name)


def power_rows(bundle, prefix):
    rows, previous = [], None
    for index, row in enumerate(read_rows(bundle / 'power.jsonl.gz')):
        require(row['index'] == index, prefix + '_power_sample_inventory')
        receipt = row['receipt']
        require(receipt['monotonic_before_ns'] <= receipt['monotonic_after_ns'] and
                (previous is None or 0 < receipt['monotonic_after_ns'] - previous <= 3e9), prefix + '_power_receipt')
        previous = receipt['monotonic_after_ns']
        decoded = decode_power(row['plist_fields'].encode() + b'\0', receipt)
        require(decoded == row['recorded_decoded'] and not decoded['issues'], prefix + '_power_decode')
        rows.append({'receipt': receipt, 'decoded': decoded})
    return rows


def cpu_median_W(power, windows):
    """Median CPU power of samples whose whole interval lies inside a measured window (no lag)."""
    values = []
    for row in power:
        lo, hi, dt = endpoint(row, 0)
        if any(a <= hi - dt and hi <= b for a, b in windows):
            values.append(row['decoded']['domains']['cpu']['power_mw'] / 1000)
    require(len(values) >= 5, 'g4a_cpu_samples')
    return statistics.median(values)


def g3_comparison(repo):
    """G3 prefill blocks and the first 256 steps of each G3 decode block, with energy over the same steps."""
    g3 = derive_g3(repo / G3_BASE)
    bundle = repo / G3_BASE
    requests = {(r['run'], r['result']['id']): r['result'] for r in read_rows(bundle / 'requests.jsonl.gz')}
    power = [{'receipt': r['receipt'], 'decoded': r['recorded_decoded']} for r in read_rows(bundle / 'power.jsonl.gz')]
    rows = {}
    for pair in g3['pairs']:
        n = pair['context']
        for arm in ('ane', 'gpu'):
            block = pair[arm]
            entry = rows.setdefault((n, arm), {'context': n, 'arm': arm})
            if pair['mode'] == 'prefill':
                entry['prefill_rate'] = block['rate']
                entry['prefill_J_per_token'] = block['J_per_token']['components']
                entry['prefill_graphs'] = block['graphs']
            else:
                row = requests[(block['run'], block['request_ids'][0])]
                start, stop = row['request_start_ns'], row['token_times'][255]['end_ns']
                window = energy_window(power, start, stop)
                require(window['integrity_passed'], 'g4a_g3_segment_energy')
                require(math.isclose(block['kv'][0]['rate'], 256 / ((stop - start) / 1e9)), 'g4a_g3_segment_rate')
                entry['decode_rate_256'] = block['kv'][0]['rate']
                entry['decode_J_per_token_256'] = window['domains']['components']['estimate_J'] / 256
                entry['decode_block_steps'] = block['tokens']
                entry['decode_graphs'] = block['graphs']
    return g3, [rows[k] for k in sorted(rows)]


def derive(bundle=None, repo=None):
    repo = Path(repo) if repo is not None else ROOT
    bundle = Path(bundle) if bundle is not None else repo / BASE
    protocol = load(bundle, 'protocol.json')
    assets = load(bundle, 'asset-identity.json')
    g3_bundle = repo / G3_BASE
    g3_sources = load(g3_bundle, 'provenance.json')['sources']
    source = protocol['inputs_source']
    require(source['path'] in g3_sources, 'g4a_inputs_identity')
    structure = load(g3_bundle, 'model-structure.json')
    require(assets['model_source']['config'] == structure['config'] and
            assets['model_source']['source_headers'] == structure['source_headers'] and
            protocol['revision'] == structure['revision'], 'g4a_model_source')
    inputs = load(g3_bundle, 'inputs.json')
    lengths, capacities = protocol['inputs_N'], protocol['ane_capacities']
    capacity = dict(zip(lengths, capacities, strict=True))
    for tier in assets['ane_tiers']:
        c = tier['capacity']
        require(capacity.get(tier['input_N']) == c and tier['input_N'] + protocol['decode_steps'] <= c, 'g4a_tier_capacity')
        kept = tier['tier']['kept_graphs_unchanged_before_serialization']
        require(kept and all(int(g.split('_')[-2]) == c for g in kept), 'g4a_tier_graphs:' + str(c))

    requests = {}
    for item in read_rows(bundle / 'requests.jsonl.gz'):
        row, command, n, arm = item['result'], item['command'], item['input_N'], item['arm']
        key = (n, arm, row['id'])
        require(key not in requests, 'g4a_duplicate_request')
        require(len(inputs[f'reading_{n}']) == n and command['input'] == f'reading_{n}' and
                row['arm'] == arm and command['graph_capacity'] == capacity[n] == item['capacity'], 'g4a_request_identity')
        require(command.get('decode_query') == protocol['decode_query'], 'g4a_decode_query')
        phase = item['phase']
        if phase == 'boundary':
            require(row['mode'] == 'quality' and command['max_new_tokens'] == protocol['full_output_tokens'], 'g4a_boundary_work')
        elif phase == 'full':
            require(row['mode'] == 'full' and command['max_new_tokens'] == protocol['full_output_tokens'] and
                    command['forced_ids'] == protocol['continuation'], 'g4a_forced_continuation')
        else:
            require(row['mode'] == 'prefill' and command['max_new_tokens'] == 1, 'g4a_prefill_work')
        stats = validate_request(row, command, n, require_graph=phase == 'boundary')
        requests[key] = dict(item, stats=stats)

    arms = load(bundle, 'arms.json')
    require({(a['input_N'], a['arm']) for a in arms} == {(n, arm) for n in lengths for arm in ('ane', 'gpu')}
            and len(arms) == 2 * len(lengths), 'g4a_arm_inventory')
    require(all(a['warmup']['last_logits_finite'] for a in arms) and
            all(a['reference_1k']['passed'] for a in arms if a['input_N'] == 1024), 'g4a_quality_controls')
    # Short pre-run checks: the host process logged direct ANE requests on every tier, and none on the GPU path.
    rows = lambda check: [v['target_PID_ANE_direct_request_log_rows'] for v in check['placement'].values()]
    admissions = list(assets['flow_admissions'].values())
    for c in capacities:
        require(any(a['arm'] == 'ane' and a['capacity'] == c and a['passed'] and all(rows(a)) and
                    all(q['passed'] for q in a['quality'].values()) for a in admissions), f'g4a_ane_admission:{c}')
    require(any(a['arm'] == 'gpu' and a['passed'] and not any(rows(a)) for a in admissions), 'g4a_gpu_admission')
    power = power_rows(bundle, 'g4a')
    audit = load(bundle, 'capture-status.json')
    require(audit['integrity_passed'] and audit['capture_completed'] and audit['cleanup_passed'] and
            not audit['clock_issues'] and audit['streams']['power']['samples'] == len(power), 'g4a_capture_integrity')
    last_receipt = power[-1]['receipt']['monotonic_after_ns']
    require(audit['last_power_receipt_ns'] == last_receipt, 'g4a_last_power_receipt')
    recorded = {(e['input_N'], e['arm']): e for e in load(bundle, 'summary-recorded.json')['arms']}
    margin = protocol['energy_interior_margin_seconds'] * 10 ** 9

    blocks, graphs = [], {}
    for arm_record in arms:
        n, arm = arm_record['input_N'], arm_record['arm']
        ids = arm_record['requests']
        require(len(ids['boundary']) == 1 and len(ids['full']) == protocol['full_repetitions'] and
                len(ids['prefill']) == protocol['prefill_counts'][str(n)], 'g4a_block_work_count')
        graphs[(n, arm)] = requests[(n, arm, 'boundary')]['stats']['graphs']
        if arm == 'ane':
            require(graphs[(n, arm)] and all(int(g.split('_')[-2]) == capacity[n] for g in graphs[(n, arm)]),
                    'g4a_boundary_graph_capacity')
        for phase, name in (('full', 'decode'), ('prefill', 'prefill')):
            rows = [requests[(n, arm, i)]['result'] for i in ids[phase]]
            if phase == 'full':
                windows = [[r['first_token_ns'], r['last_token_ns']] for r in rows]
                tokens = protocol['decode_steps'] * len(rows)
                seconds = sum((b - a) / 1e9 for a, b in windows)
            else:
                windows = [[rows[0]['request_start_ns'], rows[-1]['request_end_ns']]]
                tokens = n * len(rows)
                seconds = (windows[0][1] - windows[0][0]) / 1e9
            rate = tokens / seconds
            estimates = {lag: integrate(power, windows, int(lag) * 10 ** 9) for lag in LAGS}
            observed = response(power, rows[0]['request_start_ns'], rows[-1]['request_end_ns'], arm)
            reasons = []
            if max(b for a, b in windows) > last_receipt - margin:
                reasons.append('block_not_interior_to_power_stream')
            if not observed['passed']:
                reasons.append('power_response')
            if not all(e['timing_coverage_passed'] for e in estimates.values()):
                reasons.append('timing_coverage')
            primary = estimates['2']['domains']
            block = {'input_N': n, 'capacity': capacity[n], 'arm': arm, 'mode': name, 'tokens': tokens,
                     'seconds': seconds, 'rate': rate, 'windows': windows, 'energy': primary,
                     'J_per_token': {k: v['estimate_J'] / tokens for k, v in primary.items()},
                     'bounds_J_per_token': {k: (v['lower_J'] / tokens, v['upper_J'] / tokens) for k, v in primary.items()},
                     'sensitivity_J_per_token': {lag: e['domains']['components']['estimate_J'] / tokens for lag, e in estimates.items()},
                     'response': observed, 'cpu_median_W': cpu_median_W(power, windows),
                     'admitted': not reasons, 'reasons': reasons}
            require(block['admitted'], f'g4a_block_admission:{n}:{arm}:{name}')
            require((primary['ane']['estimate_J'] > 0) == (arm == 'ane'), f'g4a_ane_counter:{n}:{arm}:{name}')
            phase_record = recorded[(n, arm)]['phases']['decode' if phase == 'full' else 'prefill_block']
            speed_key = 'decode_tokens_per_second' if phase == 'full' else 'prefill_block_tokens_per_second'
            close(rate, phase_record['speed'][speed_key], f'g4a_recorded_speed:{n}:{arm}:{name}')
            recorded_requests = {r['id']: r['stats'] for r in phase_record['requests']}
            require(len(recorded_requests) == len(phase_record['requests']) and
                    set(recorded_requests) == set(ids[phase]), 'g4a_recorded_request_inventory')
            for request_id in ids[phase]:
                stats = {k: v for k, v in requests[(n, arm, request_id)]['stats'].items() if k != 'graphs'}
                require(recorded_requests[request_id] == stats,
                        f'g4a_recorded_request_stats:{n}:{arm}:{request_id}')
            energy_record = phase_record['energy']
            close(block['J_per_token']['components'], energy_record['component_J_per_token'],
                  f'g4a_recorded_normalization:{n}:{arm}:{name}')
            require(energy_record['response'] == observed and energy_record['reasons'] == reasons,
                    f'g4a_recorded_response:{n}:{arm}:{name}')
            require(energy_record['energy_admitted'] and energy_record['work_tokens'] == tokens,
                    f'g4a_recorded_admission:{n}:{arm}:{name}')
            for lag in LAGS:
                for key, value in estimates[lag].items():
                    if key != 'domains':
                        require(value == energy_record['counter_intervals'][lag][key],
                                f'g4a_recorded_window:{n}:{arm}:{name}:{lag}:{key}')
                for domain, values in estimates[lag]['domains'].items():
                    for bound, value in values.items():
                        close(value, energy_record['counter_intervals'][lag]['domains'][domain][bound],
                              f'g4a_recorded_energy:{n}:{arm}:{name}:{lag}')
            blocks.append(block)

    work = model_work(structure)
    pairs, implied = [], []
    for n in lengths:
        for mode in ('prefill', 'decode'):
            a, g = (next(b for b in blocks if b['input_N'] == n and b['arm'] == arm and b['mode'] == mode)
                    for arm in ('ane', 'gpu'))
            require(a['tokens'] == g['tokens'], 'g4a_same_work')
            without_cpu = lambda b: b['J_per_token']['gpu'] + b['J_per_token']['ane']
            pairs.append({'context': n, 'capacity': capacity[n], 'mode': mode, 'ane': a, 'gpu': g,
                          'gpu_speedup': g['rate'] / a['rate'],
                          'energy_ratio': a['J_per_token']['components'] / g['J_per_token']['components'],
                          'energy_ratio_without_cpu': without_cpu(a) / without_cpu(g)})
            for b in (a, g):
                if mode == 'prefill':
                    implied.append({'mode': mode, 'context': n, 'arm': b['arm'],
                        'projection_TFLOP_per_second': b['rate'] * work['flops_per_token'] / 1e12,
                        'attention_TFLOP_per_second': b['rate'] * n / 2 * work['attention_flops_per_token_per_context'] / 1e12})
                else:
                    steps = protocol['decode_steps']
                    prior = n + (steps - 1) / 2
                    implied.append({'mode': mode, 'context': n, 'arm': b['arm'], 'mean_prior_KV': prior,
                        'weight_GB_per_second': b['rate'] * work['weight_bytes'] / 1e9,
                        'kv_GB_per_second': b['rate'] * prior * work['kv_bytes_per_token'] / 1e9})

    g3, g3_rows = g3_comparison(repo)
    comparison = []
    for n in lengths:
        ane_now = {p['mode']: p['ane'] for p in pairs if p['context'] == n}
        before = next(r for r in g3_rows if r['context'] == n and r['arm'] == 'ane')
        gpu_before = next(r for r in g3_rows if r['context'] == n and r['arm'] == 'gpu')
        gpu_now = {p['mode']: p['gpu'] for p in pairs if p['context'] == n}
        comparison.append({'context': n, 'capacity': capacity[n], 'g3_ane': before, 'g3_gpu': gpu_before,
            'ane_decode_speed': ane_now['decode']['rate'] / before['decode_rate_256'],
            'ane_decode_energy': ane_now['decode']['J_per_token']['components'] / before['decode_J_per_token_256'],
            'ane_prefill_speed': ane_now['prefill']['rate'] / before['prefill_rate'],
            'ane_prefill_energy': ane_now['prefill']['J_per_token']['components'] / before['prefill_J_per_token'],
            'gpu_decode_speed': gpu_now['decode']['rate'] / gpu_before['decode_rate_256'],
            'gpu_prefill_speed': gpu_now['prefill']['rate'] / gpu_before['prefill_rate']})

    disk = load(bundle, 'disk.json')
    order = sorted(disk['host_windows'], key=lambda w: w['host_started_monotonic_s'])
    gate = disk['load_gate']
    steps, previous_end = [], 0
    for window in order:
        start = window['host_started_monotonic_s'] * 1e9
        rows = [r for r in gate if previous_end <= r['monotonic_ns'] < start]
        require(rows and rows[-1]['usable_bytes'] >= rows[-1]['required_bytes'] and
                all(r['usable_bytes'] < r['required_bytes'] for r in rows[:-1]), 'g4a_load_gate_order')
        steps.append({'input_N': window['input_N'], 'arm': window['arm'], 'gate': rows[-1], 'waits': len(rows) - 1,
                      'waiting_usable_bytes': [r['usable_bytes'] for r in rows[:-1]],
                      'host_seconds': window['host_ended_monotonic_s'] - window['host_started_monotonic_s']})
        previous_end = window['host_ended_monotonic_s'] * 1e9
    require(sum(s['waits'] for s in steps) + len(steps) == len(gate), 'g4a_load_gate_inventory')
    for step, following in zip(steps, steps[1:]):
        step['statfs_change_GiB'] = (following['gate']['statfs_free_bytes'] - step['gate']['statfs_free_bytes']) / GiB
        step['important_change_GiB'] = (following['gate']['important_usage_bytes'] - step['gate']['important_usage_bytes']) / GiB

    return {'protocol': protocol, 'blocks': blocks, 'pairs': pairs, 'implied': implied, 'work': work,
            'graphs': {f'{n}:{arm}': v for (n, arm), v in graphs.items()}, 'comparison': comparison,
            'g3_rows': g3_rows, 'g3': g3, 'disk': {'arms': steps, 'observer': disk['observer'],
            'host_windows': order}, 'request_count': len(requests), 'power_samples': len(power)}


def measurements(data):
    lines = ['## G4 A: Qwen3-4B FP16 with matched fixed ANE graphs', '',
        'Six inputs, each with an ANE graph of capacity N + 256 rounded up; the GPU uses its 32K asset. '
        'Decode is three teacher-forced 256-step requests timed from the first to the last token; prefill is a block of one-token requests timed over the block. '
        'Energy is CPU + GPU + ANE software component energy at the 2-second lag, admitted per block, without idle subtraction. '
        '[Scope](SCOPE.md#g4-a-matched-graph-observations) · [Source bundle](../' + BASE + '/).', '',
        '| Stage | Input N | ANE capacity | GPU token/s | ANE token/s | GPU / ANE speed | GPU J/token | ANE J/token | ANE / GPU J/token | Without CPU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in data['pairs']:
        a, g = p['ane'], p['gpu']
        lines.append(f"| {p['mode']} | {p['context']:,} | {p['capacity']:,} | {g['rate']:.3f} | {a['rate']:.3f} | "
                     f"{p['gpu_speedup']:.2f}× | {g['J_per_token']['components']:.6f} | {a['J_per_token']['components']:.6f} | "
                     f"{p['energy_ratio']:.3f} | {p['energy_ratio_without_cpu']:.3f} |")
    lines += ['', '### Blocks', '',
        '| Stage | Input N | Arm | Work tokens | Seconds | CPU J/token | GPU J/token | ANE J/token | Components [timing bounds] | CPU median W |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in data['blocks']:
        e, lo = b['J_per_token'], b['bounds_J_per_token']['components']
        lines.append(f"| {b['mode']} | {b['input_N']:,} | {b['arm']} | {b['tokens']:,} | {b['seconds']:.3f} | {e['cpu']:.6f} | "
                     f"{e['gpu']:.6f} | {e['ane']:.6f} | {e['components']:.6f} [{lo[0]:.6f}, {lo[1]:.6f}] | {b['cpu_median_W']:.3f} |")
    lines += ['', '### Against G3 (256 / 2K / 32K ANE graphs)', '',
        'G3 decode uses the first 256 steps of each 1024-step decode block, with energy over those steps. '
        'G3 prefill uses its warmed prefill blocks.', '',
        '| Input N | G3 ANE decode token/s | G4 A / G3 ANE decode speed | ANE decode J/token G4 A / G3 | G3 ANE prefill token/s | G4 A / G3 ANE prefill speed | ANE prefill J/token G4 A / G3 | G4 A / G3 GPU decode, prefill speed |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in data['comparison']:
        lines.append(f"| {c['context']:,} | {c['g3_ane']['decode_rate_256']:.3f} | {c['ane_decode_speed']:.2f}× | "
                     f"{c['ane_decode_energy']:.3f} | {c['g3_ane']['prefill_rate']:.3f} | {c['ane_prefill_speed']:.2f}× | "
                     f"{c['ane_prefill_energy']:.3f} | {c['gpu_decode_speed']:.3f}, {c['gpu_prefill_speed']:.3f} |")
    work = data['work']
    lines += ['', '### Implied compute and memory reads', '',
        f"Prefill counts two FLOPs per projection weight per token ({work['projection_parameters']:,} weights) and, separately, "
        f"causal attention over an average of N/2 keys. Decode assumes every FP16 weight ({work['weight_bytes']:,} bytes) and the existing KV "
        f"({work['kv_bytes_per_token']:,} bytes per token) are read once per step, over KV N to N + 255. "
        'These are model-based rates for the measured token rates; no device counter or memory traffic was measured.', '',
        '| Input N | Arm | Projection TFLOP/s | Attention TFLOP/s | Weight GB/s | KV GB/s |',
        '|---:|---|---:|---:|---:|---:|']
    for n in data['protocol']['inputs_N']:
        for arm in ('gpu', 'ane'):
            pre = next(r for r in data['implied'] if r['mode'] == 'prefill' and r['context'] == n and r['arm'] == arm)
            dec = next(r for r in data['implied'] if r['mode'] == 'decode' and r['context'] == n and r['arm'] == arm)
            lines.append(f"| {n:,} | {arm} | {pre['projection_TFLOP_per_second']:.2f} | {pre['attention_TFLOP_per_second']:.2f} | "
                         f"{dec['weight_GB_per_second']:.1f} | {dec['kv_GB_per_second']:.1f} |")
    lines += ['', '### Free space at each host load', '',
        'Recorded by the load gate before each host started, in run order. The change is to the next load.', '',
        '| Order | Input N | Arm | statfs free GiB | Important-usage GiB | Waiting rows | Change to next load, statfs GiB |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for i, s in enumerate(data['disk']['arms'], 1):
        change = f"{s['statfs_change_GiB']:+.1f}" if 'statfs_change_GiB' in s else '—'
        lines.append(f"| {i} | {s['input_N']:,} | {s['arm']} | {s['gate']['statfs_free_bytes'] / GiB:.1f} | "
                     f"{s['gate']['important_usage_bytes'] / GiB:.1f} | {s['waits']} | {change} |")
    return '\n'.join(lines) + '\n'

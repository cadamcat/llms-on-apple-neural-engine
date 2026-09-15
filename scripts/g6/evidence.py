"""Recompute G6 decode-query, W4 and G4 A repeat speed and component energy from the portable bundle."""
import json
from pathlib import Path

from g3.evidence import BASE as G3_BASE, read_rows
from g3.power import decode_power, response
from g3.protocol import require
from g4a.evidence import BASE as G4A_BASE, close, cpu_median_W, derive as derive_g4a
from g4a.power import integrate
from g4a.protocol import validate_request

ROOT = Path(__file__).resolve().parents[2]
BASE = 'results/historical/g6-qwen3-4b'
LAGS = ('0', '1', '2', '5')
GiB = 2 ** 30


def load(bundle, name):
    return json.loads((bundle / name).read_text())


def power_by_capture(bundle):
    captures = {}
    for row in read_rows(bundle / 'power.jsonl.gz'):
        rows = captures.setdefault(row['capture'], [])
        require(row['index'] == len(rows), 'g6_power_sample_inventory:' + row['capture'])
        receipt = row['receipt']
        require(receipt['monotonic_before_ns'] <= receipt['monotonic_after_ns'] and
                (not rows or 0 < receipt['monotonic_after_ns'] - rows[-1]['receipt']['monotonic_after_ns'] <= 3e9),
                'g6_power_receipt:' + row['capture'])
        decoded = decode_power(row['plist_fields'].encode() + b'\0', receipt)
        require(decoded == row['recorded_decoded'] and not decoded['issues'], 'g6_power_decode:' + row['capture'])
        rows.append({'receipt': receipt, 'decoded': decoded})
    return captures


def measure(power, last_receipt, margin_s, rows, full, tokens_per_row, arm, recorded):
    """One block as the runner measured it, checked against the recorded summary entry."""
    if full:
        windows = [[r['first_token_ns'], r['last_token_ns']] for r in rows]
        seconds = sum((b - a) / 1e9 for a, b in windows)
    else:
        windows = [[rows[0]['request_start_ns'], rows[-1]['request_end_ns']]]
        seconds = (windows[0][1] - windows[0][0]) / 1e9
    tokens = tokens_per_row * len(rows)
    estimates = {lag: integrate(power, windows, int(lag) * 10 ** 9) for lag in LAGS}
    observed = response(power, rows[0]['request_start_ns'], rows[-1]['request_end_ns'], arm)
    reasons = []
    if max(b for a, b in windows) > last_receipt - margin_s * 10 ** 9:
        reasons.append('block_not_interior_to_power_stream')
    if not observed['passed']:
        reasons.append('power_response')
    if not all(e['timing_coverage_passed'] for e in estimates.values()):
        reasons.append('timing_coverage')
    primary = estimates['2']['domains']
    block = {'tokens': tokens, 'seconds': seconds, 'rate': tokens / seconds, 'windows': windows,
             'J_per_token': {k: v['estimate_J'] / tokens for k, v in primary.items()},
             'bounds_J_per_token': {k: (v['lower_J'] / tokens, v['upper_J'] / tokens) for k, v in primary.items()},
             'response': observed, 'cpu_median_W': cpu_median_W(power, windows), 'admitted': not reasons, 'reasons': reasons}
    return block, estimates, recorded


def compare_recorded(measured):
    """Recomputed block versus what the runner wrote; called after the block's own rules."""
    block, estimates, recorded = measured
    energy = recorded['energy']
    close(block['J_per_token']['components'], energy['component_J_per_token'], 'g6_recorded_normalization')
    require(energy['response'] == block['response'] and energy['reasons'] == block['reasons'] and
            energy['energy_admitted'] == block['admitted'] and energy['work_tokens'] == block['tokens'], 'g6_recorded_admission')
    for lag in LAGS:
        for key, value in estimates[lag].items():
            if key != 'domains':
                require(value == energy['counter_intervals'][lag][key], f'g6_recorded_window:{lag}:{key}')
        for domain, values in estimates[lag]['domains'].items():
            for bound, value in values.items():
                close(value, energy['counter_intervals'][lag]['domains'][domain][bound], f'g6_recorded_energy:{lag}')
    speed = 'decode_tokens_per_second' if 'decode_tokens_per_second' in recorded['speed'] else 'prefill_block_tokens_per_second'
    close(block['rate'], recorded['speed'][speed], 'g6_recorded_speed')
    return block


def derive(bundle=None, repo=None):
    repo = Path(repo) if repo is not None else ROOT
    bundle = Path(bundle) if bundle is not None else repo / BASE
    protocol = load(bundle, 'protocol.json')
    assets = load(bundle, 'asset-identity.json')
    g3_bundle = repo / G3_BASE
    source = protocol['inputs_source']
    require(source['path'] in load(g3_bundle, 'provenance.json')['sources'], 'g6_inputs_identity')
    # The repeat arms ran the G4 A host binaries: the recorded launch hashes of both runs are the evidence.
    require(load(bundle, 'runtime-implementation.json')['g4a_host_sha256'] ==
            load(repo / G4A_BASE, 'runtime-implementation.json')['host_sha256'], 'g6_repeat_host_identity')
    require(protocol['revision'] == load(g3_bundle, 'model-structure.json')['revision'], 'g6_model_source')
    inputs = load(g3_bundle, 'inputs.json')
    g4a = derive_g4a(repo=repo)
    require(g4a['protocol']['continuation'] == protocol['continuation'] and
            g4a['protocol']['prefill_counts'] == protocol['prefill_counts'], 'g6_g4a_protocol')

    # Assets: the new FP16 tiers keep G4 A's Q8/Q64 signatures; W4 codes cover every projection with group 32.
    tiers = {(t['weights'], t['capacity']): t for t in assets['tiers']}
    for (weights, c), t in tiers.items():
        kept = t['tier']['kept_graphs_unchanged_before_serialization']
        require(sorted(kept) == sorted([f'prompt_opt_{c}_64'] + [f'extend_{c}_{q}' for q in (4, 64, 8)]), f'g6_tier_graphs:{weights}:{c}')
        require({f'extend_{c}_8', f'extend_{c}_64', f'prompt_opt_{c}_64'} <= set(t['tier']['signatures_equal_to_G4_A']),
                f'g6_tier_signatures:{weights}:{c}')
    codes = assets['w4_codes']
    require(codes['count'] == len(codes['modules']) == 252 and
            all(m['group_size'] == assets['w4_preset']['group_size'] and m['lut_shape'][-2] == 2 ** assets['w4_preset']['n_bits']
                and m['max_index'] < 16 for m in codes['modules'].values()), 'g6_w4_codes')
    reference = assets['w4_reference']
    require(reference['codes_inventory_sha256'] == codes['inventory_sha256'], 'g6_w4_reference_codes')

    # Placement at admission: ANE direct requests for every FP16 tier, none for W4.
    placement = {}
    for name, a in assets['admissions'].items():
        weights = 'w4' if name.startswith('w4-') else 'fp16'
        logs = a['placement_logs'].values()
        ane = [x['ane_direct_request_success_rows'] for x in logs]
        metal = [x['metal_shader_compile_rows'] for x in logs]
        require(logs and all(r['passed'] for r in a['results'].values()), 'g6_admission_passed:' + name)
        if weights == 'fp16':
            require(all(v > 0 for v in ane), 'g6_fp16_admission_placement:' + name)
        else:
            require(not any(ane) and all(v > 0 for v in metal), 'g6_w4_admission_placement:' + name)
        placement[name] = {'weights': weights, 'capacity': a['capacity'], 'ane_requests': sum(ane),
                           'metal_shader_compiles': sum(metal), 'results': a['results']}

    requests = {}
    for item in read_rows(bundle / 'requests.jsonl.gz'):
        row, command, n = item['result'], item['command'], item['input_N']
        key = (item['segment'], row['id'])
        require(key not in requests, 'g6_duplicate_request')
        require(len(inputs[f'reading_{n}']) == n and command['input'] == f'reading_{n}' and row['arm'] == item['arm'] and
                command['graph_capacity'] == item['capacity'], 'g6_request_identity')
        phase = item['phase']
        if phase == 'boundary':
            require(row['mode'] == 'quality' and command['max_new_tokens'] == protocol['full_output_tokens'], 'g6_boundary_work')
        elif phase in ('full', 'decode_q8', 'decode_q4'):
            require(row['mode'] == 'full' and command['max_new_tokens'] == protocol['full_output_tokens'] and
                    command['forced_ids'] == protocol['continuation'], 'g6_forced_continuation')
            expected = 8 if phase in ('full', 'decode_q8') else 4
            require(command['decode_query'] == expected, 'g6_decode_query')
        else:
            require(row['mode'] == 'prefill' and command['max_new_tokens'] == 1, 'g6_prefill_work')
        requests[key] = dict(item, stats=validate_request(row, command, n, require_graph=phase == 'boundary'))

    power = power_by_capture(bundle)
    status = load(bundle, 'capture-status.json')
    for name, rows in power.items():
        s = status[name]
        require(s['integrity_passed'] and s['capture_completed'] and s['cleanup_passed'] and not s['clock_issues'] and
                s['streams']['power']['samples'] == len(rows) and
                s['last_power_receipt_ns'] == rows[-1]['receipt']['monotonic_after_ns'], 'g6_capture_integrity:' + name)
    last = {name: rows[-1]['receipt']['monotonic_after_ns'] for name, rows in power.items()}
    margin = protocol['energy_interior_margin_seconds']
    sessions = load(bundle, 'sessions.json')
    summary = load(bundle, 'summary-recorded.json')
    recorded = {e['id']: e for e in summary['segments']}
    planned = [s for capture in protocol['captures'] for s in capture]
    require({s['id'] for s in sessions['query_sessions']} | {s['id'] for s in sessions['repeat_arms']} |
            {s['id'] for s in sessions['idle']} == {s['id'] for s in planned}, 'g6_segment_inventory')

    queries = []
    for s in sessions['query_sessions']:
        entry = recorded[s['id']]
        require(entry['monitor'] == s['monitor'] and sorted(s['queries']) == [4, 8], 'g6_session_identity:' + s['id'])
        boundary = {q: requests[(s['id'], f'boundary-q{q}')]['stats']['graphs'] for q in s['queries']}
        for q, graphs in boundary.items():
            require(f"extend_{s['capacity']}_{q}" in graphs and all(int(g.split('_')[-2]) == s['capacity'] for g in graphs),
                    f"g6_boundary_graphs:{s['id']}:{q}")
            if s['input_N'] == 1024:
                require(s['warmup'][str(q)]['reference_1k']['passed'], 'g6_reference_1k:' + s['id'])
        record = {'id': s['id'], 'weights': s['weights'], 'input_N': s['input_N'], 'capacity': s['capacity'],
                  'order': s['queries'], 'monitor': s['monitor'], 'blocks': {}}
        phases = [(f'decode_q{q}', True) for q in (8, 4)] + ([('prefill', False)] if s['prefill'] else [])
        for phase, full in phases:
            ids = s['requests'][phase]
            count = protocol['full_repetitions'] if full else protocol['prefill_counts'][str(s['input_N'])]
            require(len(ids) == count, f"g6_block_work_count:{s['id']}:{phase}")
            rows = [requests[(s['id'], i)]['result'] for i in ids]
            key = phase if full else 'prefill_block'
            measured = measure(power[s['monitor']], last[s['monitor']], margin, rows, full,
                               protocol['decode_steps'] if full else s['input_N'], 'ane', entry['phases'][key])
            block = measured[0]
            if s['weights'] == 'fp16':
                require(block['admitted'] and block['J_per_token']['ane'] > 0, f"g6_fp16_block:{s['id']}:{phase}")
            else:
                # Every W4 block: no ANE counter energy, and the ANE response check is the only failed admission rule.
                require(block['J_per_token']['ane'] == 0 and block['reasons'] == ['power_response'],
                        f"g6_w4_placement:{s['id']}:{phase}")
            record['blocks'][key] = compare_recorded(measured)
        queries.append(record)

    fp16 = {r['input_N']: r for r in queries if r['weights'] == 'fp16'}
    w4 = {r['input_N']: r for r in queries if r['weights'] == 'w4'}
    widths = []
    for n, r in sorted(fp16.items()):
        q8, q4 = r['blocks']['decode_q8'], r['blocks']['decode_q4']
        widths.append({'input_N': n, 'capacity': r['capacity'], 'order': r['order'], 'q8': q8, 'q4': q4,
                       'speed_q4_over_q8': q4['rate'] / q8['rate'],
                       'ane_energy_q4_over_q8': q4['J_per_token']['ane'] / q8['J_per_token']['ane'],
                       'components_energy_q4_over_q8': q4['J_per_token']['components'] / q8['J_per_token']['components']})

    arms, repeat = [], []
    original = {(b['input_N'], b['arm'], b['mode']): b for b in g4a['blocks']}
    recorded_arms = {(e['input_N'], e['arm']): e for e in summary['g4a_repeat_arms']}
    for a in sessions['repeat_arms']:
        n, arm = a['input_N'], a['arm']
        require(len(a['requests']['full']) == protocol['full_repetitions'] and
                len(a['requests']['prefill']) == protocol['prefill_counts'][str(n)], 'g6_repeat_work_count:' + a['id'])
        require(a['warmup']['last_logits_finite'] and (n != 1024 or a['reference_1k']['passed']), 'g6_repeat_quality:' + a['id'])
        entry = recorded_arms[(n, arm)]
        for phase, full, mode, key in (('full', True, 'decode', 'decode'), ('prefill', False, 'prefill', 'prefill_block')):
            rows = [requests[(a['id'], i)]['result'] for i in a['requests'][phase]]
            measured = measure(power[a['monitor']], last[a['monitor']], margin, rows, full,
                               protocol['decode_steps'] if full else n, arm, entry['phases'][key])
            block = measured[0]
            require(block['admitted'] and (block['J_per_token']['ane'] > 0) == (arm == 'ane'), f'g6_repeat_block:{n}:{arm}:{mode}')
            compare_recorded(measured)
            before = original[(n, arm, mode)]
            block.update(input_N=n, arm=arm, mode=mode, capacity=a['capacity'],
                         speed_vs_g4a=block['rate'] / before['rate'],
                         energy_vs_g4a=block['J_per_token']['components'] / before['J_per_token']['components'])
            arms.append(block)
    for n in g4a['protocol']['inputs_N']:
        for mode in ('prefill', 'decode'):
            a = next(b for b in arms if b['input_N'] == n and b['arm'] == 'ane' and b['mode'] == mode)
            g = next(b for b in arms if b['input_N'] == n and b['arm'] == 'gpu' and b['mode'] == mode)
            before = next(p for p in g4a['pairs'] if p['context'] == n and p['mode'] == mode)
            repeat.append({'context': n, 'mode': mode, 'gpu_speedup': g['rate'] / a['rate'], 'g4a_gpu_speedup': before['gpu_speedup'],
                           'energy_ratio': a['J_per_token']['components'] / g['J_per_token']['components'],
                           'g4a_energy_ratio': before['energy_ratio']})

    idle = []
    for s in sessions['idle']:
        rows = power[s['monitor']]
        window = s['window_ns']
        estimates = {lag: integrate(rows, [window], int(lag) * 10 ** 9) for lag in LAGS}
        reasons = []
        if window[1] > last[s['monitor']] - margin * 10 ** 9:
            reasons.append('block_not_interior_to_power_stream')
        if not all(e['timing_coverage_passed'] for e in estimates.values()):
            reasons.append('timing_coverage')
        seconds = (window[1] - window[0]) / 1e9
        mean = {d: estimates['2']['domains'][d]['estimate_J'] / seconds for d in ('cpu', 'gpu', 'ane', 'components')}
        entry = recorded[s['id']]['power']
        require(entry['reasons'] == reasons and entry['energy_admitted'] == (not reasons) and not reasons, 'g6_idle_admission:' + s['id'])
        for d, v in mean.items():
            close(v, entry['mean_W'][d], 'g6_recorded_idle_power:' + s['id'])
        idle.append({'id': s['id'], 'seconds': seconds, 'mean_W': mean})

    w4_rows = []
    for n, r in sorted(w4.items()):
        f = fp16[n]
        row = {'input_N': n, 'capacity': r['capacity'], 'w4': r['blocks'], 'fp16': f['blocks'],
               'fp16_faster_decode_q8': f['blocks']['decode_q8']['rate'] / r['blocks']['decode_q8']['rate'],
               'fp16_faster_decode_q4': f['blocks']['decode_q4']['rate'] / r['blocks']['decode_q4']['rate'],
               'w4_speed_q4_over_q8': r['blocks']['decode_q4']['rate'] / r['blocks']['decode_q8']['rate']}
        ane_prefill = next(b for b in arms if b['input_N'] == n and b['arm'] == 'ane' and b['mode'] == 'prefill')
        row['fp16_ane_prefill_rate'] = ane_prefill['rate']
        row['fp16_faster_prefill'] = ane_prefill['rate'] / r['blocks']['prefill_block']['rate']
        w4_rows.append(row)

    disk = load(bundle, 'disk.json')
    return {'protocol': protocol, 'queries': queries, 'widths': widths, 'w4': w4_rows, 'repeat_blocks': arms,
            'repeat_pairs': repeat, 'idle': idle, 'placement': placement, 'reference': reference, 'g4a': g4a,
            'reclaim': disk['reclaim_breaks'], 'request_count': len(requests),
            'power_samples': sum(len(v) for v in power.values()), 'captures': len(power),
            'blocks': sum(len(r['blocks']) for r in queries) + len(arms)}


def measurements(data):
    lines = ['## G6: decode query width, W4 palettization and a G4 A repeat', '',
        'One night on the G4 A inputs and energy rules. FP16 decode runs with query 8 and query 4 in one ANE host per input; '
        'the upstream iOS 4-bit palettized preset runs at 1K and 4K; every G4 A arm is repeated in a new host with the arm order reversed. '
        '[Scope](SCOPE.md#g6-decode-query-w4-and-repeat) · [Source bundle](../' + BASE + '/).', '',
        '### FP16 decode, query 8 and query 4', '',
        '| Input N | ANE capacity | First query | Q8 token/s | Q4 token/s | Q4 / Q8 speed | Q8 ANE J/token | Q4 ANE J/token | Q8 GPU J/token | Q4 GPU J/token | Q4 / Q8 components J/token |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for w in data['widths']:
        a, b = w['q8']['J_per_token'], w['q4']['J_per_token']
        lines.append(f"| {w['input_N']:,} | {w['capacity']:,} | Q{w['order'][0]} | {w['q8']['rate']:.3f} | {w['q4']['rate']:.3f} | "
                     f"{w['speed_q4_over_q8']:.4f} | {a['ane']:.4f} | {b['ane']:.4f} | {a['gpu']:.4f} | {b['gpu']:.4f} | "
                     f"{w['components_energy_q4_over_q8']:.3f} |")
    lines += ['', '### W4 palettized weights (executed on the GPU)', '',
        'No ANE direct request at admission and no ANE counter energy in any block; energy admission fails its ANE response rule, so the J/token columns are not admitted comparisons.', '',
        '| Input N | Stage | W4 token/s | FP16 ANE token/s | FP16 / W4 speed | W4 CPU J/token | W4 GPU J/token | W4 ANE J/token |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in data['w4']:
        for stage, key, fast in (('decode Q8', 'decode_q8', r['fp16']['decode_q8']['rate']), ('decode Q4', 'decode_q4', r['fp16']['decode_q4']['rate']),
                                 ('prefill', 'prefill_block', r['fp16_ane_prefill_rate'])):
            b = r['w4'][key]
            lines.append(f"| {r['input_N']:,} | {stage} | {b['rate']:.3f} | {fast:.3f} | {fast / b['rate']:.2f}× | "
                         f"{b['J_per_token']['cpu']:.4f} | {b['J_per_token']['gpu']:.4f} | {b['J_per_token']['ane']:.4f} |")
    lines += ['', '### G4 A repeat', '',
        'Same assets, hosts and protocol as G4 A in new host sessions, arm order reversed at every input. Ratios are repeat / G4 A.', '',
        '| Stage | Input N | Arm | Repeat token/s | Speed vs G4 A | Repeat J/token | Energy vs G4 A |',
        '|---|---:|---|---:|---:|---:|---:|']
    for b in sorted(data['repeat_blocks'], key=lambda b: (b['mode'], b['input_N'], b['arm'])):
        lines.append(f"| {b['mode']} | {b['input_N']:,} | {b['arm']} | {b['rate']:.3f} | {b['speed_vs_g4a']:.3f} | "
                     f"{b['J_per_token']['components']:.6f} | {b['energy_vs_g4a']:.3f} |")
    lines += ['', '| Stage | Input N | GPU / ANE speed, G4 A | GPU / ANE speed, repeat | ANE / GPU J/token, G4 A | ANE / GPU J/token, repeat |',
              '|---|---:|---:|---:|---:|---:|']
    for p in data['repeat_pairs']:
        lines.append(f"| {p['mode']} | {p['context']:,} | {p['g4a_gpu_speedup']:.2f}× | {p['gpu_speedup']:.2f}× | "
                     f"{p['g4a_energy_ratio']:.3f} | {p['energy_ratio']:.3f} |")
    lines += ['', '### Idle captures', '', '| Segment | Seconds | CPU W | GPU W | ANE W | Components W |', '|---|---:|---:|---:|---:|---:|']
    for i in data['idle']:
        m = i['mean_W']
        lines.append(f"| {i['id']} | {i['seconds']:.0f} | {m['cpu']:.3f} | {m['gpu']:.3f} | {m['ane']:.3f} | {m['components']:.3f} |")
    return '\n'.join(lines) + '\n'

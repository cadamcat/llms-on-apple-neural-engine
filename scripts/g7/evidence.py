"""Recompute G7 Core ML / Core AI same-code speed, component energy and admission from the portable bundle."""
import hashlib
import json
from pathlib import Path
import statistics

from g3.evidence import read_rows
from g3.power import decode_power, response
from g3.protocol import require
from g4a.evidence import close
from g4a.power import integrate

ROOT = Path(__file__).resolve().parents[2]
BASE = 'results/historical/g7-coreml-coreai'
LAGS = ('0', '1', '2', '5')
RUNTIMES = ('coreml', 'coreai')
FORMATS = ('fp16_from_w4_codes', 'w4a16', 'w8a8_same_codes', 'a8w4_int8_lut')
A8 = ('w8a8_same_codes', 'a8w4_int8_lut')
LABELS = ('original', 'negative', 'benchmark')


def load(bundle, name):
    return json.loads((bundle / name).read_text())


def power_by_capture(bundle):
    captures = {}
    for row in read_rows(bundle / 'power.jsonl.gz'):
        rows = captures.setdefault(row['capture'], [])
        require(row['index'] == len(rows), 'g7_power_sample_inventory:' + row['capture'])
        receipt = row['receipt']
        require(receipt['monotonic_before_ns'] <= receipt['monotonic_after_ns'] and
                (not rows or 0 < receipt['monotonic_after_ns'] - rows[-1]['receipt']['monotonic_after_ns'] <= 3e9),
                'g7_power_receipt:' + row['capture'])
        decoded = decode_power(row['plist_fields'].encode() + b'\0', receipt)
        require(decoded == row['recorded_decoded'] and not decoded['issues'], 'g7_power_decode:' + row['capture'])
        rows.append({'receipt': receipt, 'decoded': decoded})
    return captures


def limit(asset, limits):
    if asset['workload'] == 'synthetic':
        return limits['synthetic_depth2'] if asset['depth'] == 2 else limits['synthetic_depth128']
    return limits['real_a8'] if asset['format'] in A8 else limits['real_fp16_or_w4a16']


def admitted(name, asset, record, limits):
    """Apply the admission rules to a recorded control: numeric screen, repeat identity, zero output, ANE participation."""
    numeric, placement, hashes = record['numeric'], record['placement'], record['control_output_sha256']
    bound = limit(asset, limits)
    require(numeric['limit'] == bound, 'g7_numeric_limit:' + name)
    reasons = [f'numerical:{label}' for label in LABELS
               if any(not v['finite'] or v['relative_l2'] is None or v['relative_l2'] > bound for v in numeric['comparisons'][label].values())]
    zero = hashlib.sha256(bytes(2 * asset['output_shape'][1] * asset['output_shape'][3] *
                                (asset['output_shape'][2] if asset['workload'] == 'synthetic' else 1))).hexdigest()
    if hashes['zero'] != zero:
        reasons.append('zero')
    for a, b in (('original', 'repeat'), ('benchmark', 'benchmark_repeat')):
        if hashes[a] != hashes[b]:
            reasons.append('repeat:' + a)
    require(numeric['reasons'] == reasons and numeric['passed'] == (not reasons), 'g7_numeric_rule:' + name)
    plan = placement['compute_plan']
    # The run's rule: a successful ANE request in every control call, no compile or fallback failure line and,
    # for Core ML, the expected number of projection convolutions, each preferring the Neural Engine.
    ane = all(r >= 1 for r in placement['requests_per_control']) and not placement['failures']
    if asset['runtime'] == 'coreml':
        conv = [p for p in plan if p['operator'].split('.')[-1] == 'conv']
        expected = asset['depth'] if asset['workload'] == 'synthetic' else 1 if asset['workload'] == 'gate' else 3
        ane = ane and len(conv) == expected and all(p['preferred'] == 'MLNeuralEngineComputeDevice' for p in conv)
    else:
        require(plan == {'per_operator_mapping': 'unavailable', 'preferred': 'neuralEngine'}, 'g7_coreai_plan:' + name)
    require(placement['passed'] == ane, 'g7_placement_rule:' + name)
    if asset['runtime'] == 'coreml':
        # Beyond the run's rule: every other non-constant operation also prefers the Neural Engine.
        others = [p for p in plan if not p['operator'].split('.')[-1].startswith('constexpr_')]
        require(all(p['preferred'] == 'MLNeuralEngineComputeDevice' for p in others), 'g7_coreml_operations_not_all_ane:' + name)
    require(record['passed'] == (numeric['passed'] and ane), 'g7_admission_rule:' + name)
    return record['passed']


def measure(block, positions, rows, status, recorded, name):
    """One block as the runner measured it, checked against its recorded measurement."""
    start, end = block['block']['start_ns'], block['block']['end_ns']
    seconds = (end - start) / 1e9
    work = positions * block['block']['calls']
    estimates = {lag: integrate(rows, [[start, end]], int(lag) * 10 ** 9) for lag in LAGS}
    observed = response(rows, start, end, 'ane')
    reasons = []
    if not status['integrity_passed']:
        reasons.append('capture_integrity')
    if rows[-1]['receipt']['monotonic_after_ns'] < end + 15e9:
        reasons.append('not_interior')
    if not all(e['timing_coverage_passed'] for e in estimates.values()):
        reasons.append('timing_coverage')
    if not observed['passed']:
        reasons.append('power_response')
    require(recorded['positions'] == work and recorded['calls'] == block['block']['calls'], 'g7_recorded_work:' + name)
    close(work / seconds, recorded['positions_per_second'], 'g7_recorded_speed:' + name)
    require(block['api_latency_ms']['p50'] == recorded['API_p50_ms'], 'g7_recorded_latency:' + name)
    require(recorded['energy_reasons'] == reasons and recorded['energy_admitted'] == (not reasons) and
            recorded['response'] == observed, 'g7_recorded_admission:' + name)
    for lag in LAGS:
        for domain, values in estimates[lag]['domains'].items():
            for bound, value in values.items():
                close(value, recorded['integrals'][lag]['domains'][domain][bound], f'g7_recorded_energy:{lag}:{name}')
    J = {k: v['estimate_J'] / work for k, v in estimates['2']['domains'].items()}
    for k, v in J.items():
        close(v, recorded['J_per_position'][k], 'g7_recorded_normalization:' + name)
    return {'rate': work / seconds, 'seconds': seconds, 'work': work, 'calls': block['block']['calls'], 'admitted': not reasons,
            'J_per_position': J, 'W': {k: v['estimate_J'] / seconds for k, v in estimates['2']['domains'].items()},
            'api': block['api_latency_ms']}


def median(values):
    return statistics.median(values)


def derive(bundle=None, repo=None):
    repo = Path(repo) if repo is not None else ROOT
    bundle = Path(bundle) if bundle is not None else repo / BASE
    protocol = load(bundle, 'protocol.json')
    configs = load(bundle, 'configs.json')
    blocks = load(bundle, 'blocks.json')
    status = load(bundle, 'capture-status.json')
    limits = protocol['numeric_relative_L2_limits']

    admission = {name: admitted(name, c['asset'], c['admission'], limits) for name, c in configs.items()}
    real = {name: c for name, c in configs.items() if c['asset']['workload'] != 'synthetic' and not c['asset']['clip']}
    # Byte identity of the original control output across runtimes, per workload, positions and format.
    identical = {}
    for name, c in real.items():
        a = c['asset']
        if a['runtime'] == 'coreml':
            other = name.replace('-coreml-', '-coreai-')
            identical[name.replace('-coreml-', '-')] = (c['admission']['control_output_sha256']['original'] ==
                                                         configs[other]['admission']['control_output_sha256']['original'])
    clamp = {name: c for name, c in configs.items() if c['asset']['clip']}
    for name, c in clamp.items():
        other = name.replace('-coreml-', '-coreai-') if '-coreml-' in name else None
        if other:
            identical[name.replace('-coreml-', '-')] = (c['admission']['control_output_sha256']['original'] ==
                                                         configs[other]['admission']['control_output_sha256']['original'])

    power = power_by_capture(bundle)
    for name, rows in power.items():
        s = status[name]
        require(s['integrity_passed'] and s['capture_completed'] and s['cleanup_passed'] and
                s['streams']['power']['samples'] == len(rows) and
                s['last_power_receipt_ns'] == rows[-1]['receipt']['monotonic_after_ns'], 'g7_capture_integrity:' + name)

    measured = {}
    require(len({b['id'] for b in blocks}) == len(blocks), 'g7_duplicate_block')
    for b in blocks:
        c = configs[b['config']]
        a = c['asset']
        require(admission[b['config']], 'g7_block_without_admission:' + b['id'])
        require(admitted(b['id'], a, b['admission'], limits), 'g7_block_control:' + b['id'])
        require(b['block']['end_ns'] - b['block']['start_ns'] >= b['seconds_requested'] * 1e9 and
                b['seed'] == protocol['benchmark_seeds'][b['round']] and status[b['capture']]['group'] == b['group'], 'g7_block_contract:' + b['id'])
        m = measure(b, a['positions'], power[b['capture']], status[b['capture']], b['recorded_measurement'], b['id'])
        require(m['admitted'], 'g7_block_energy_admission:' + b['id'])
        # No GPU work and ANE energy in every block: the placement check is not the only evidence of ANE execution.
        require(m['W']['ane'] > 1 and m['W']['gpu'] < 0.05, 'g7_block_domains:' + b['id'])
        measured[b['id']] = dict(m, round=b['round'], workload=a['workload'], runtime=a['runtime'], format=a['format'],
                                 positions=a['positions'], depth=a.get('depth'))
    rounds = sorted({m['round'] for m in measured.values()})
    require(rounds == list(range(protocol['rounds'])), 'g7_rounds')
    key = {(m['workload'], m['positions'], m['runtime'], m['format'], m['round']): m for m in measured.values()}
    cells = sorted({k[:4] for k in key})
    for cell in cells:
        require(all(cell + (r,) in key for r in rounds), 'g7_cell_rounds:' + '-'.join(map(str, cell)))
    summary = []
    for cell in cells:
        ms = [key[cell + (r,)] for r in rounds]
        summary.append({'workload': cell[0], 'positions': cell[1], 'runtime': cell[2], 'format': cell[3],
                        'rate': median([m['rate'] for m in ms]), 'J_per_position': median([m['J_per_position']['components'] for m in ms]),
                        'W': {d: median([m['W'][d] for m in ms]) for d in ('cpu', 'gpu', 'ane')},
                        'api_p50_ms': median([m['api']['p50'] for m in ms])})

    def ratios(a, b):
        speed = [key[a + (r,)]['rate'] / key[b + (r,)]['rate'] for r in rounds]
        energy = [key[a + (r,)]['J_per_position']['components'] / key[b + (r,)]['J_per_position']['components'] for r in rounds]
        return {'speed': speed, 'energy': energy, 'speed_median': median(speed)}

    within, across = [], []
    for workload, n, runtime, fmt in cells:
        if workload == 'synthetic':
            continue
        for base in ('fp16_from_w4_codes', 'w4a16'):
            # Every format against FP16; the A8 formats also against W4A16.
            if fmt != base and (base == 'fp16_from_w4_codes' or fmt in A8) and (workload, n, runtime, base) in cells:
                within.append({'workload': workload, 'positions': n, 'runtime': runtime, 'format': fmt, 'baseline': base} |
                              ratios((workload, n, runtime, fmt), (workload, n, runtime, base)))
        if runtime == 'coreml':
            across.append({'workload': workload, 'positions': n, 'format': fmt} |
                          ratios((workload, n, 'coreml', fmt), (workload, n, 'coreai', fmt)))
    synthetic = [s for s in summary if s['workload'] == 'synthetic']
    synthetic_ratio = ratios(('synthetic', 4096, 'coreml', 'w8a8_same_codes'), ('synthetic', 4096, 'coreml', 'fp16_from_w4_codes'))

    idle = {}
    for name, entry in load(bundle, 'idle.json').items():
        window = entry['window_ns']
        estimate = integrate(power[name], [window], 2 * 10 ** 9)
        seconds = (window[1] - window[0]) / 1e9
        mean = {d: estimate['domains'][d]['estimate_J'] / seconds for d in ('cpu', 'gpu', 'ane', 'components')}
        require(estimate['timing_coverage_passed'] and entry['recorded']['timing_coverage_passed'], 'g7_idle_coverage:' + name)
        for d, v in mean.items():
            close(v, entry['recorded']['mean_W'][d], 'g7_recorded_idle_power:' + name)
        idle[name] = mean
    closures = load(bundle, 'closures.json')
    require(closures['run']['status'] == 'closed' and not closures['run']['cancelled_assets'] and not closures['run']['unrun'] and
            not closures['processes']['still_present'], 'g7_run_closure')
    return {'protocol': protocol, 'configs': configs, 'admission': admission, 'identical_original_output': identical,
            'blocks': measured, 'summary': summary, 'within': within, 'across': across, 'synthetic': synthetic,
            'synthetic_w8a8_over_fp16': synthetic_ratio, 'idle': idle, 'captures': len(power),
            'power_samples': sum(len(v) for v in power.values()), 'calls': sum(m['calls'] for m in measured.values())}


def pick(rows, **match):
    found = [r for r in rows if all(r[k] == v for k, v in match.items())]
    require(len(found) == 1, 'g7_pick:' + json.dumps(match))
    return found[0]


NAMES = {'fp16_from_w4_codes': 'FP16 from the codes', 'w4a16': 'W4A16', 'w8a8_same_codes': 'W8A8, same codes', 'a8w4_int8_lut': 'A8W4'}


def measurements(data):
    lines = ['## G7: Core ML and Core AI on the same E4B codes', '',
             'The first E4B mobile QAT gate projection and MLP from the same four-bit codes and scales, four representations, '
             'both runtimes on the ANE, 64 and 1024 positions, three independent host rounds of 120 s each. Medians across rounds; '
             'ranges are the three paired rounds. Software component energy (CPU + GPU + ANE counters, no idle subtraction). '
             '[Scope](SCOPE.md#g7-core-ml-and-core-ai-on-the-same-codes) · [Source bundle](../' + BASE + '/).', '',
             '### Speed and component energy', '',
             '| Work | Positions | Representation | Core ML positions/s | Core AI positions/s | Core ML / Core AI speed | Core ML mJ/position | Core AI mJ/position | Core ML ANE W | Core AI ANE W |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    cell = {(s['workload'], s['positions'], s['runtime'], s['format']): s for s in data['summary']}
    for x in data['across']:
        ml, ai = cell[(x['workload'], x['positions'], 'coreml', x['format'])], cell[(x['workload'], x['positions'], 'coreai', x['format'])]
        lines.append(f"| {x['workload']} | {x['positions']:,} | {NAMES[x['format']]} | {ml['rate']:,.0f} | {ai['rate']:,.0f} | "
                     f"{min(x['speed']):.3f}–{max(x['speed']):.3f}× | {ml['J_per_position'] * 1e3:.4f} | {ai['J_per_position'] * 1e3:.4f} | "
                     f"{ml['W']['ane']:.2f} | {ai['W']['ane']:.2f} |")
    lines += ['', '### Within each runtime', '', '| Work | Positions | Runtime | Ratio | Speed | Component energy |', '|---|---:|---|---|---:|---:|']
    for w in data['within']:
        lines.append(f"| {w['workload']} | {w['positions']:,} | {w['runtime']} | {NAMES[w['format']]} / {NAMES[w['baseline']]} | "
                     f"{min(w['speed']):.3f}–{max(w['speed']):.3f}× | {min(w['energy']):.3f}–{max(w['energy']):.3f}× |")
    lines += ['', '### Admission', '', 'Relative L2 on ordinary rows of the original control; placement is one successful ANE request in every control call '
              'and, for Core ML, every projection convolution preferring the Neural Engine. Configurations that fail are not timed.', '',
              '| Configuration | Relative L2 | Limit | Numeric | ANE | Timed |', '|---|---:|---:|---|---|---|']
    timed = {k.split('-', 1)[1] for k in data['blocks']}
    for name, c in data['configs'].items():
        n = c['admission']['numeric']
        lines.append(f"| {name} | {n['comparisons']['original']['ordinary']['relative_l2']:.4g} | {n['limit']} | "
                     f"{'pass' if n['passed'] else 'fail'} | {'pass' if c['admission']['placement']['passed'] else 'fail'} | "
                     f"{'yes' if name in timed else 'no'} |")
    lines += ['', 'Core ML and Core AI original-control outputs are byte-identical for: ' +
              ', '.join(k for k, v in data['identical_original_output'].items() if v) + '.', '',
              '### Core ML synthetic chain', '', '| Representation | Median API p50 (ms) | Median positions/s |', '|---|---:|---:|']
    for s in data['synthetic']:
        lines.append(f"| {NAMES[s['format']]} | {s['api_p50_ms']:.3f} | {s['rate']:,.0f} |")
    r = data['synthetic_w8a8_over_fp16']
    lines += ['', f"W8A8 / FP16 speed by round: {min(r['speed']):.3f}–{max(r['speed']):.3f}×.", '',
              '### Idle captures', '', '| Capture | CPU W | GPU W | ANE W | Components W |', '|---|---:|---:|---:|---:|']
    for name, m in data['idle'].items():
        lines.append(f"| {name} | {m['cpu']:.3f} | {m['gpu']:.3f} | {m['ane']:.3f} | {m['components']:.3f} |")
    return '\n'.join(lines) + '\n'

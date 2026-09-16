"""Recompute G8 four-bit representation speed, component energy, admission and weight identity from the portable bundle."""
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
BASE = 'results/historical/g8-coreml-four-bit'
LAGS = ('0', '1', '2', '5')
LABELS = ('original', 'negative', 'benchmark')
PHASES = ('original', 'follow-up')
COREML = ('fp16', 'int8_lut', 'int4', 'fp16_lut')
# Constant-expression operators each saved representation must contain, per projection.
OPERATORS = {'fp16': {}, 'int8_lut': {'constexpr_blockwise_shift_scale': 1, 'constexpr_lut_to_dense': 1},
             'int4': {'constexpr_blockwise_shift_scale': 1}, 'fp16_lut': {'constexpr_lut_to_dense': 1},
             'coreai': {'coreai.lut_to_dense': 1, 'coreai.blockwise_shift_scale': 1}}
PROJECTIONS = {'gate': ('gate_proj',), 'mlp': ('gate_proj', 'up_proj', 'down_proj')}


def load(bundle, name):
    return json.loads((bundle / name).read_text())


def label(asset):
    return asset['representation'] if asset['runtime'] == 'coreml' else 'coreai'


def power_by_capture(bundle):
    captures = {}
    for row in read_rows(bundle / 'power.jsonl.gz'):
        rows = captures.setdefault(row['capture'], [])
        require(row['index'] == len(rows), 'g8_power_sample_inventory:' + row['capture'])
        receipt = row['receipt']
        require(receipt['monotonic_before_ns'] <= receipt['monotonic_after_ns'] and
                (not rows or 0 < receipt['monotonic_after_ns'] - rows[-1]['receipt']['monotonic_after_ns'] <= 3e9),
                'g8_power_receipt:' + row['capture'])
        decoded = decode_power(row['plist_fields'].encode() + b'\0', receipt)
        require(decoded == row['recorded_decoded'] and not decoded['issues'], 'g8_power_decode:' + row['capture'])
        rows.append({'receipt': receipt, 'decoded': decoded})
    return captures


def weights(configs, protocol):
    """Same codes in every configuration; the four Core ML representations decode to the same FP16 weights."""
    for name, c in configs.items():
        a = c['asset']
        counts = {k: v for k, v in a['audit']['counts'].items() if k.startswith(('constexpr_', 'coreai.l', 'coreai.b'))}
        per = len(PROJECTIONS[a['workload']])
        require(counts == {k: v * per for k, v in OPERATORS[label(a)].items()}, 'g8_saved_representation:' + name)
        if a['runtime'] == 'coreai':
            require(a['audit']['weights'] == list(PROJECTIONS[a['workload']]), 'g8_coreai_projections:' + name)
            continue
        codes = protocol['references'][f"n{a['positions']}"]['source_codes']
        require(sorted(w['name'] for w in a['audit']['weights']) == sorted(PROJECTIONS[a['workload']]) and
                all(w['codes_sha256'] == codes[w['name']] for w in a['audit']['weights']), 'g8_source_codes:' + name)
    decoded = {}
    for name, c in configs.items():
        a = c['asset']
        if a['runtime'] == 'coreml':
            decoded.setdefault((a['workload'], a['positions']), {})[a['representation']] = \
                {w['name']: w['decoded_fp16_sha256'] for w in a['audit']['weights']}
    for key, forms in decoded.items():
        require(sorted(forms) == sorted(COREML) and len({json.dumps(v, sort_keys=True) for v in forms.values()}) == 1,
                'g8_decoded_weights:' + '-'.join(map(str, key)))
    # Decoded weights also repeat across the two position counts.
    require(len({json.dumps(next(iter(f.values())), sort_keys=True) for (w, _), f in decoded.items() if w == 'mlp'}) == 1,
            'g8_decoded_weights_positions')
    return len(decoded)


def admitted(name, asset, record, limit):
    """Apply the admission rules to a recorded control: numeric screen, repeat identity, zero output, ANE participation."""
    numeric, placement, hashes = record['numeric'], record['placement'], record['control_output_sha256']
    require(numeric['limit'] == limit, 'g8_numeric_limit:' + name)
    comparisons = numeric.get('comparisons')
    require(isinstance(comparisons, dict) and set(comparisons) == set(LABELS), 'g8_numeric_inventory:' + name)
    for control in LABELS:
        groups = comparisons[control]
        require(isinstance(groups, dict) and set(groups) == {'all', 'ordinary'} and
                all(isinstance(v, dict) and {'finite', 'relative_l2'} <= v.keys() for v in groups.values()),
                'g8_numeric_inventory:' + name + ':' + control)
    reasons = [f'numerical:{x}' for x in LABELS
               if any(not v['finite'] or v['relative_l2'] is None or v['relative_l2'] > limit for v in numeric['comparisons'][x].values())]
    if hashes['zero'] != hashlib.sha256(bytes(2 * asset['output_shape'][1] * asset['output_shape'][3])).hexdigest():
        reasons.append('zero')
    for a, b in (('original', 'repeat'), ('benchmark', 'benchmark_repeat')):
        if hashes[a] != hashes[b]:
            reasons.append('repeat:' + a)
    require(numeric['reasons'] == reasons and numeric['passed'] == (not reasons), 'g8_numeric_rule:' + name)
    # The run's rule, under the phase's clock mapping: a successful ANE request in every control call, no compile or
    # fallback failure line and, for Core ML, the expected projection convolutions each preferring the Neural Engine.
    requests = placement['requests_per_control']
    require(len(requests) == 6 and all(type(r) is int and r >= 0 for r in requests), 'g8_control_inventory:' + name)
    require(placement['clock_mapping'] in ('wall', 'mach') and
            record['requests_per_control_by_mapping'][placement['clock_mapping']] == requests, 'g8_clock_mapping:' + name)
    ane = all(r >= 1 for r in requests) and not placement['failures']
    plan = placement['compute_plan']
    if asset['runtime'] == 'coreml':
        conv = [p for p in plan if p['operator'].split('.')[-1] == 'conv']
        ane = ane and len(conv) == len(PROJECTIONS[asset['workload']]) and all(p['preferred'] == 'MLNeuralEngineComputeDevice' for p in conv)
    else:
        require(plan == {'per_operator_mapping': 'unavailable', 'preferred': 'neuralEngine'}, 'g8_coreai_plan:' + name)
    require(placement['passed'] == ane and (not placement['reasons']) == ane, 'g8_placement_rule:' + name)
    require(record['passed'] == (numeric['passed'] and ane), 'g8_admission_rule:' + name)
    return record['passed']


def beyond_run_rule(name, asset, record):
    """Checks the run did not make: every non-constant Core ML operation prefers the Neural Engine, and the
    Mach-clock recount finds a successful ANE request in every control call whatever the phase's own mapping."""
    if asset['runtime'] == 'coreml':
        others = [p for p in record['placement']['compute_plan'] if not p['operator'].split('.')[-1].startswith('constexpr_')]
        require(all(p['preferred'] == 'MLNeuralEngineComputeDevice' for p in others), 'g8_coreml_operations_not_all_ane:' + name)
    requests = record['requests_per_control_by_mapping'].get('mach')
    require(isinstance(requests, list) and len(requests) == 6 and all(type(r) is int and r >= 0 for r in requests),
            'g8_mach_inventory:' + name)
    require(all(r >= 1 for r in requests), 'g8_mach_requests:' + name)


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
    require(recorded['positions'] == work and recorded['calls'] == block['block']['calls'], 'g8_recorded_work:' + name)
    close(work / seconds, recorded['positions_per_second'], 'g8_recorded_speed:' + name)
    require(block['api_latency_ms']['p50'] == recorded['API_p50_ms'], 'g8_recorded_latency:' + name)
    require(recorded['energy_reasons'] == reasons and recorded['energy_admitted'] == (not reasons) and
            recorded['response'] == observed, 'g8_recorded_admission:' + name)
    for lag in LAGS:
        for domain, values in estimates[lag]['domains'].items():
            for bound, value in values.items():
                close(value, recorded['integrals'][lag]['domains'][domain][bound], f'g8_recorded_energy:{lag}:{name}')
    J = {k: v['estimate_J'] / work for k, v in estimates['2']['domains'].items()}
    for k, v in J.items():
        close(v, recorded['J_per_position'][k], 'g8_recorded_normalization:' + name)
    return {'rate': work / seconds, 'seconds': seconds, 'work': work, 'calls': block['block']['calls'], 'admitted': not reasons,
            'J_per_position': J, 'W': {k: v['estimate_J'] / seconds for k, v in estimates['2']['domains'].items()},
            'api': block['api_latency_ms']}


def output_classes(configs, blocks):
    """Group runtime/representation labels whose control outputs are byte-identical, per workload, positions and input."""
    hashes = {}
    records = [(c['asset'], c['admission'], None) for c in configs.values()]
    records += [(configs[b['config']]['asset'], b['admission'], b['seed']) for b in blocks]
    for asset, record, seed in records:
        key = (asset['workload'], asset['positions'])
        inputs = {'original': record['control_output_sha256']['original'], 'negative': record['control_output_sha256']['negative']}
        if seed is not None:
            inputs[f'benchmark-{seed}'] = record['control_output_sha256']['benchmark']
        for control, digest in inputs.items():
            hashes.setdefault(key + (control,), {}).setdefault(label(asset), set()).add(digest)
    classes = {}
    for key, by_label in hashes.items():
        require(all(len(v) == 1 for v in by_label.values()), 'g8_output_repeat:' + '-'.join(map(str, key)))
        groups = {}
        for name, digest in by_label.items():
            groups.setdefault(next(iter(digest)), []).append(name)
        classes[key] = sorted(sorted(g) for g in groups.values())
    return classes


def median(values):
    return statistics.median(values)


def derive(bundle=None, repo=None):
    repo = Path(repo) if repo is not None else ROOT
    bundle = Path(bundle) if bundle is not None else repo / BASE
    protocol = load(bundle, 'protocol.json')
    configs = load(bundle, 'configs.json')
    blocks = load(bundle, 'blocks.json')
    rejections = load(bundle, 'rejections.json')
    status = load(bundle, 'capture-status.json')
    limit = protocol['numeric_relative_L2_limit']

    identities = weights(configs, protocol)
    admission = {}
    for name, c in configs.items():
        admission[name] = admitted(name, c['asset'], c['admission'], limit)
        beyond_run_rule(name, c['asset'], c['admission'])

    power = power_by_capture(bundle)
    require(sorted(power) == sorted(status), 'g8_capture_inventory')
    for name, rows in power.items():
        s = status[name]
        require(s['integrity_passed'] and s['capture_completed'] and s['cleanup_passed'] and
                s['streams']['power']['samples'] == len(rows) and
                s['last_power_receipt_ns'] == rows[-1]['receipt']['monotonic_after_ns'], 'g8_capture_integrity:' + name)

    measured = {}
    require(len({(b['id'], b['phase']) for b in blocks}) == len(blocks) and len({b['id'] for b in blocks}) == len(blocks), 'g8_duplicate_block')
    for b in blocks:
        a = configs[b['config']]['asset']
        require(admission[b['config']], 'g8_block_without_admission:' + b['id'])
        require(admitted(b['id'], a, b['admission'], limit), 'g8_block_control:' + b['id'])
        beyond_run_rule(b['id'], a, b['admission'])
        require(b['admission']['placement']['clock_mapping'] == ('wall' if b['phase'] == 'original' else 'mach'), 'g8_phase_mapping:' + b['id'])
        require(b['block']['end_ns'] - b['block']['start_ns'] >= b['seconds_requested'] * 1e9 and
                b['seed'] == protocol['benchmark_seeds'][b['round']] and status[b['capture']]['group'] == b['group'] and
                b['capture'].startswith(b['phase'] + '/') and b['id'] == f"r{b['round']}-{b['config']}", 'g8_block_contract:' + b['id'])
        m = measure(b, a['positions'], power[b['capture']], status[b['capture']], b['recorded_measurement'], b['id'])
        require(m['admitted'], 'g8_block_energy_admission:' + b['id'])
        # ANE energy and no GPU work in every block: the placement check is not the only evidence of ANE execution.
        require(m['W']['ane'] > 1 and m['W']['gpu'] < 0.05, 'g8_block_domains:' + b['id'])
        measured[b['id']] = dict(m, phase=b['phase'], round=b['round'], workload=a['workload'], positions=a['positions'],
                                 runtime=a['runtime'], representation=label(a))
    rounds = list(range(protocol['rounds']))
    key = {(m['workload'], m['positions'], m['representation'], m['round']): m for m in measured.values()}
    cells = sorted({k[:3] for k in key})
    require(len(cells) == len(configs) and len(key) == len(measured), 'g8_cell_inventory')
    for cell in cells:
        require(all(cell + (r,) in key for r in rounds) and len({key[cell + (r,)]['phase'] for r in rounds}) == 1,
                'g8_cell_rounds:' + '-'.join(map(str, cell)))

    # The follow-up measured exactly the rejected configurations' rounds, after numeric controls had passed.
    rejected = rejections['rejected_first_round']
    follow = sorted(b['id'] for b in blocks if b['phase'] == 'follow-up')
    require(sorted([r['id'] for r in rejected] + rejections['cancelled_later_rounds']) == follow, 'g8_follow_up_inventory')
    for r in rejected:
        record, asset = r['admission'], configs[r['config']]['asset']
        require(not record['passed'] and record['numeric']['passed'] and record['placement']['clock_mapping'] == 'wall' and
                record['placement']['reasons'] == ['missing_ane_control_request'] and not record['placement']['failures'] and
                not all(x >= 1 for x in record['placement']['requests_per_control']), 'g8_rejection_record:' + r['id'])
        admitted(r['id'], asset, record, limit)
        beyond_run_rule(r['id'], asset, record)

    summary = []
    for cell in cells:
        ms = [key[cell + (r,)] for r in rounds]
        summary.append({'workload': cell[0], 'positions': cell[1], 'representation': cell[2], 'phase': ms[0]['phase'],
                        'rate': median([m['rate'] for m in ms]), 'J_per_position': median([m['J_per_position']['components'] for m in ms]),
                        'W': {d: median([m['W'][d] for m in ms]) for d in ('cpu', 'gpu', 'ane')},
                        'api_p50_ms': median([m['api']['p50'] for m in ms])})

    def ratios(a, b):
        speed = [key[a + (r,)]['rate'] / key[b + (r,)]['rate'] for r in rounds]
        energy = [key[a + (r,)]['J_per_position']['components'] / key[b + (r,)]['J_per_position']['components'] for r in rounds]
        return {'speed': speed, 'energy': energy, 'phases': [key[a + (0,)]['phase'], key[b + (0,)]['phase']],
                'cross_phase': key[a + (0,)]['phase'] != key[b + (0,)]['phase']}

    # Each Core ML representation against Core ML FP16 and against Core AI's palette, and the direct and palette forms against each other.
    comparisons = [(rep, 'fp16') for rep in COREML[1:]] + [(rep, 'coreai') for rep in COREML] + \
                  [('int4', 'int8_lut'), ('int4', 'fp16_lut'), ('fp16_lut', 'int8_lut')]
    pairs = [{'workload': workload, 'positions': n, 'representation': rep, 'baseline': base} |
             ratios((workload, n, rep), (workload, n, base))
             for workload, n in sorted({c[:2] for c in cells}) for rep, base in comparisons]

    idle = {}
    for name, entry in load(bundle, 'idle.json').items():
        window = entry['window_ns']
        estimate = integrate(power[name], [window], 2 * 10 ** 9)
        seconds = (window[1] - window[0]) / 1e9
        require(estimate['timing_coverage_passed'] and seconds >= 300, 'g8_idle_coverage:' + name)
        mean = {d: estimate['domains'][d]['estimate_J'] / seconds for d in ('cpu', 'gpu', 'ane', 'components')}
        for d, v in mean.items():
            close(v, entry['recorded_mean_W'][d], 'g8_recorded_idle_power:' + name)
        idle[name] = mean
    closures = load(bundle, 'closures.json')
    for phase in PHASES:
        run = closures[phase]['run']
        require(run['status'] == 'closed' and not run['unrun'] and not closures[phase]['processes']['still_present'], 'g8_run_closure:' + phase)
    require(sorted(closures['original']['run']['cancelled_assets']) == sorted(r['config'] for r in rejected) and
            not closures['follow-up']['run']['cancelled_assets'], 'g8_run_cancellations')
    controls = [c['admission'] for c in configs.values()] + [b['admission'] for b in blocks] + [r['admission'] for r in rejected]
    max_relative_l2 = max(v['relative_l2'] for c in controls for groups in c['numeric']['comparisons'].values() for v in groups.values())
    return {'protocol': protocol, 'configs': configs, 'admission': admission, 'weight_sets': identities, 'blocks': measured,
            'max_relative_l2': max_relative_l2,
            'summary': summary, 'pairs': pairs, 'classes': output_classes(configs, blocks), 'rejections': rejections,
            'idle': idle, 'captures': len(power), 'power_samples': sum(len(v) for v in power.values()),
            'calls': sum(m['calls'] for m in measured.values())}


def pick(rows, **match):
    found = [r for r in rows if all(r[k] == v for k, v in match.items())]
    require(len(found) == 1, 'g8_pick:' + json.dumps(match))
    return found[0]


# G7 stored the same codes as FP16 and as the INT8-palette W4A16 in both runtimes.
G7_FORMS = {('coreml', 'fp16_from_w4_codes'): 'fp16', ('coreml', 'w4a16'): 'int8_lut', ('coreai', 'w4a16'): 'coreai'}


def g7_agreement(data, g7):
    """Median-speed ratio G8 / G7 for every configuration both runs measured."""
    rows = []
    for s in g7['summary']:
        rep = G7_FORMS.get((s['runtime'], s['format']))
        if rep and s['workload'] in PROJECTIONS:
            mine = pick(data['summary'], workload=s['workload'], positions=s['positions'], representation=rep)
            rows.append({'workload': s['workload'], 'positions': s['positions'], 'representation': rep, 'phase': mine['phase'],
                         'g7_rate': s['rate'], 'g8_rate': mine['rate'], 'ratio': mine['rate'] / s['rate']})
    require(len(rows) == 12, 'g8_g7_overlap')
    return rows


NAMES = {'fp16': 'Core ML FP16', 'int8_lut': 'Core ML INT8 palette', 'int4': 'Core ML direct INT4', 'fp16_lut': 'Core ML FP16 palette',
         'coreai': 'Core AI INT8 palette'}


def measurements(data, g7):
    lines = ['## G8: four-bit representations of the same E4B codes', '',
             'The G7 gate projection and MLP from the same four-bit codes and scales, stored four ways in Core ML with Core AI\'s palette as a '
             'control; 64 and 1024 positions, FP16 activations, three independent host rounds of 120 s each. Medians across rounds; ranges are '
             'the three paired rounds. Phase O is the original run; phase F is the follow-up that timed the nine blocks the original rejected '
             'before timing. Software component energy (CPU + GPU + ANE counters, no idle subtraction). '
             '[Scope](SCOPE.md#g8-four-bit-representations-of-the-same-codes) · [Source bundle](../' + BASE + '/).', '',
             '### Speed and component energy', '',
             '| Work | Positions | Representation | Phase | Positions/s | mJ/position | CPU W | ANE W | API p50 ms |',
             '|---|---:|---|---|---:|---:|---:|---:|---:|']
    order = {r: i for i, r in enumerate(('fp16', 'int8_lut', 'fp16_lut', 'int4', 'coreai'))}
    for s in sorted(data['summary'], key=lambda s: (s['workload'], s['positions'], order[s['representation']])):
        lines.append(f"| {s['workload']} | {s['positions']:,} | {NAMES[s['representation']]} | {'O' if s['phase'] == 'original' else 'F'} | "
                     f"{s['rate']:,.0f} | {s['J_per_position'] * 1e3:.4f} | {s['W']['cpu']:.3f} | {s['W']['ane']:.3f} | {s['api_p50_ms']:.3f} |")
    lines += ['', '### Paired ratios', '', 'Rows marked across phases compare blocks from different capture periods.', '',
              '| Work | Positions | Representation / baseline | Speed | Component energy | Phases |', '|---|---:|---|---:|---:|---|']
    for p in data['pairs']:
        lines.append(f"| {p['workload']} | {p['positions']:,} | {NAMES[p['representation']]} / {NAMES[p['baseline']]} | "
                     f"{min(p['speed']):.3f}–{max(p['speed']):.3f}× | {min(p['energy']):.3f}–{max(p['energy']):.3f}× | "
                     f"{'across phases' if p['cross_phase'] else 'same phase'} |")
    lines += ['', '### Byte-identical outputs', '', 'Labels whose outputs are byte-identical for every control input of a workload and size.', '',
              '| Work | Positions | Output classes |', '|---|---:|---|']
    by_size = {}
    for (w, n, control), groups in data['classes'].items():
        by_size.setdefault((w, n), set()).add(json.dumps(groups))
    for (w, n), variants in sorted(by_size.items()):
        require(len(variants) == 1, f'g8_output_classes_vary:{w}-{n}')
        groups = json.loads(next(iter(variants)))
        lines.append(f"| {w} | {n:,} | " + ' · '.join('{' + ', '.join(NAMES[x] for x in g) + '}' for g in groups) + ' |')
    lines += ['', '### Agreement with G7', '', 'Median speed of the configurations G7 also measured.', '',
              '| Work | Positions | Representation | G8 phase | G7 positions/s | G8 positions/s | G8 / G7 |', '|---|---:|---|---|---:|---:|---:|']
    for r in g7_agreement(data, g7):
        lines.append(f"| {r['workload']} | {r['positions']:,} | {NAMES[r['representation']]} | {'O' if r['phase'] == 'original' else 'F'} | "
                     f"{r['g7_rate']:,.0f} | {r['g8_rate']:,.0f} | {r['ratio']:.3f} |")
    lines += ['', '### Idle captures', '', '| Capture | CPU W | GPU W | ANE W | Components W |', '|---|---:|---:|---:|---:|']
    for name, m in data['idle'].items():
        lines.append(f"| {name} | {m['cpu']:.3f} | {m['gpu']:.3f} | {m['ane']:.3f} | {m['components']:.3f} |")
    return '\n'.join(lines) + '\n'

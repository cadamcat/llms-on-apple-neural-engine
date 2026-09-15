"""Named document quantities derived from the imported G7 records and the Core ML QDQ probe."""
import importlib.util

from .evidence import BASE, ROOT, pick

PROBE = 'findings/coreai-qdq-multiply-scale/repro/coreml'


def span(values, places, unit=''):
    low, high = f'{min(values):.{places}f}', f'{max(values):.{places}f}'
    return (low if low == high else f'{low}–{high}') + unit


def quantities(data):
    from doc_claims import Quantity
    out = {}

    def add(key, value, en='', zh=None, source='blocks.json'):
        out['g7.' + key] = Quantity(str(value), en, en if zh is None else zh,
                                    source if source.startswith(('findings/', 'results/')) else BASE + '/' + source)

    across = data['across']
    for fmt, key in (('fp16_from_w4_codes', 'fp16'), ('w8a8_same_codes', 'w8a8'), ('w4a16', 'w4a16'), ('a8w4_int8_lut', 'a8w4')):
        add(f'ml-over-ai.{key}.speed', span([v for x in across if x['format'] == fmt for v in x['speed']], 3, '×'))
    four = [v for x in across if x['format'] in ('w4a16', 'a8w4_int8_lut') for v in x['speed']]
    add('ml-over-ai.four-bit.speed', span(four, 3, '×'))
    add('ai-over-ml.four-bit.speed', span([1 / v for v in four], 1, '×'))
    within = data['within']
    w4_fp16 = lambda runtime: [v for w in within if w['runtime'] == runtime and w['format'] == 'w4a16' for v in w['speed']]
    add('coreml.w4a16-over-fp16', span(w4_fp16('coreml'), 3, '×'))
    add('coreai.w4a16-over-fp16', span(w4_fp16('coreai'), 3, '×'))
    for n in (64, 1024):
        for runtime in ('coreml', 'coreai'):
            for fmt, base, key in (('a8w4_int8_lut', 'w4a16', 'a8w4-over-w4a16'), ('a8w4_int8_lut', 'fp16_from_w4_codes', 'a8w4-over-fp16'),
                                   ('w8a8_same_codes', 'w4a16', 'w8a8-over-w4a16'), ('w8a8_same_codes', 'fp16_from_w4_codes', 'w8a8-over-fp16'),
                                   ('w4a16', 'fp16_from_w4_codes', 'w4a16-over-fp16')):
                row = pick(within, workload='gate', positions=n, runtime=runtime, format=fmt, baseline=base)
                add(f'{runtime}.gate.{n}.{key}.speed', span(row['speed'], 3, '×'))
                add(f'{runtime}.gate.{n}.{key}.energy', span(row['energy'], 3, '×'))
    for s in data['summary']:
        label = f"{s['workload']}.{s['positions']}.{s['runtime']}.{s['format'].split('_')[0]}"
        add(f'rate.{label}', f"{s['rate']:,.0f}")
        add(f'ane-power.{label}', f"{s['W']['ane']:.1f}", 'W')
    add('gate.positions-64', '64', source='protocol.json')
    add('gate.positions-1024', '1024', source='protocol.json')

    configs = data['configs']
    mlp_a8 = [c['admission']['numeric']['comparisons']['original']['ordinary']['relative_l2'] for name, c in configs.items()
              if c['asset']['workload'] == 'mlp' and c['asset']['format'] in ('w8a8_same_codes', 'a8w4_int8_lut') and not c['asset']['clip']]
    add('mlp.a8.l2', span([100 * v for v in mlp_a8], 0, '%'), source='configs.json')
    clamp = [c['admission']['numeric']['comparisons']['original']['ordinary']['relative_l2'] for c in configs.values() if c['asset']['clip']]
    add('mlp.clamp.l2', span([100 * v for v in clamp], 1, '%'), source='configs.json')
    add('mlp.a8.configs', len(mlp_a8), source='configs.json')
    add('mlp.a8.admitted', sum(1 for name, c in configs.items() if c['asset']['workload'] == 'mlp' and c['asset']['format'] in
                                ('w8a8_same_codes', 'a8w4_int8_lut') and data['admission'][name]), source='configs.json')
    synthetic = configs['synthetic-d128-coreml-a8w4_int8_lut']['admission']['numeric']['comparisons']['original']['ordinary']['relative_l2']
    add('synthetic.a8w4.l2', f'{synthetic:.3f}', source='configs.json')
    add('synthetic.w8a8-over-fp16', span(data['synthetic_w8a8_over_fp16']['speed'], 2, '×'))
    add('blocks', len(data['blocks']))
    add('calls', f"{data['calls']:,}")
    add('power-samples', f"{data['power_samples']:,}", source='power.jsonl.gz')
    add('captures', data['captures'], source='capture-status.json')
    add('configs', len(configs), source='configs.json')
    add('rounds', data['protocol']['rounds'], source='protocol.json')

    spec = importlib.util.spec_from_file_location('qdq_coreml_verify', ROOT / PROBE / 'verify.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    record, rows = verifier.outputs()
    source = PROBE + '/recorded/results.json'
    for row in rows:
        arm = row['arm'].replace('_', '-')
        units = 'ane' if row['compute_units'] == 'cpuAndNeuralEngine' else 'cpu'
        for phase, value in row['observed'].items():
            add(f'qdq.{arm}.{units}.{phase}', f'{value:g}', source=source)
    add('qdq.channels', f"{record['shape'][1]:,}", source=source)
    add('qdq.positions', f"{record['shape'][3]:,}", source=source)
    add('qdq.small-runs', len(record['small_graph_placement']), source=source)
    return out

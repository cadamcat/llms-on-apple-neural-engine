"""Named document quantities derived from the imported G8 records, with the G7 medians they are compared against."""
from .evidence import BASE, g7_agreement, pick


def span(values, places, unit=''):
    low, high = f'{min(values):.{places}f}', f'{max(values):.{places}f}'
    return (low if low == high else f'{low}–{high}') + unit


def quantities(data, g7):
    from doc_claims import Quantity
    out = {}

    def add(key, value, en='', zh=None, source='blocks.json'):
        out['g8.' + key] = Quantity(str(value), en, en if zh is None else zh,
                                    source if source.startswith('results/') else BASE + '/' + source)

    pairs, summary = data['pairs'], data['summary']
    cells = [('gate', 64), ('gate', 1024), ('mlp', 64), ('mlp', 1024)]
    for s in summary:
        add(f"rate.{s['workload']}.{s['positions']}.{s['representation'].replace('_', '-')}", f"{s['rate']:,.0f}")
        add(f"ane-power.{s['workload']}.{s['positions']}.{s['representation'].replace('_', '-')}", f"{s['W']['ane']:.1f}", 'W')
    for w, n in cells:
        for rep, base, key in (('int4', 'coreai', 'int4-over-coreai'), ('int4', 'fp16', 'int4-over-fp16'),
                               ('int4', 'int8_lut', 'int4-over-int8-lut'), ('int8_lut', 'fp16', 'int8-lut-over-fp16')):
            row = pick(pairs, workload=w, positions=n, representation=rep, baseline=base)
            add(f'{key}.{w}.{n}.speed', span(row['speed'], 3, '×'))
            add(f'{key}.{w}.{n}.energy', span(row['energy'], 3, '×'))
    same = [v for w, n in cells for r in [pick(pairs, workload=w, positions=n, representation='int4', baseline='coreai')]
            if not r['cross_phase'] for v in r['speed']]
    add('int4-over-coreai.same-phase.speed', span(same, 3, '×'))
    add('int4-over-fp16.other.speed', span([v for w, n in cells[:3] for v in pick(pairs, workload=w, positions=n, representation='int4', baseline='fp16')['speed']], 3, '×'))
    add('fp16-lut-over-int8-lut.speed', span([v for w, n in cells for v in pick(pairs, workload=w, positions=n, representation='fp16_lut', baseline='int8_lut')['speed']], 3, '×'))
    palettes = [v for rep in ('int8_lut', 'fp16_lut') for v in pick(pairs, workload='gate', positions=1024, representation=rep, baseline='fp16')['speed']]
    add('palettes-over-fp16.gate.1024.speed', span(palettes, 3, '×'))
    api = {(s['workload'], s['positions'], s['representation']): s['api_p50_ms'] for s in summary}
    add('api.mlp.1024.fp16', f"{api['mlp', 1024, 'fp16']:.1f}", 'ms')
    add('api.gate.1024.fp16', f"{api['gate', 1024, 'fp16']:.1f}", 'ms')
    add('api.mlp.1024.int4', f"{api['mlp', 1024, 'int4']:.1f}", 'ms')
    add('api.gate.1024.int4', f"{api['gate', 1024, 'int4']:.1f}", 'ms')
    add('api.mlp-over-gate.1024.fp16', f"{api['mlp', 1024, 'fp16'] / api['gate', 1024, 'fp16']:.1f}×")
    add('api.mlp-over-gate.1024.int4', f"{api['mlp', 1024, 'int4'] / api['gate', 1024, 'int4']:.1f}×")

    agreement = g7_agreement(data, g7)
    add('g7-agreement', span([r['ratio'] for r in agreement], 3, '×'))
    add('g7-agreement.count', len(agreement))
    add('g7-agreement.mlp.1024.fp16', f"{pick(agreement, workload='mlp', positions=1024, representation='fp16')['ratio']:.3f}×")
    add('g7.rate.mlp.1024.coreai.fp16', f"{pick(g7['summary'], workload='mlp', positions=1024, runtime='coreai', format='fp16_from_w4_codes')['rate']:,.0f}",
        source='results/historical/g7-coreml-coreai/blocks.json')

    idle = data['idle']
    add('idle.original.start', f"{idle['original/idle-start']['components']:.3f}", 'W', source='idle.json')
    add('idle.follow-up.start', f"{idle['follow-up/idle-start']['components']:.3f}", 'W', source='idle.json')
    blocks = data['blocks'].values()
    add('blocks', len(data['blocks']))
    add('blocks.original', sum(1 for b in blocks if b['phase'] == 'original'))
    add('blocks.follow-up', sum(1 for b in blocks if b['phase'] == 'follow-up'))
    add('rejected', len(data['rejections']['rejected_first_round']), source='rejections.json')
    add('cancelled', len(data['rejections']['cancelled_later_rounds']), source='rejections.json')
    add('calls', f"{data['calls']:,}")
    add('power-samples', f"{data['power_samples']:,}", source='power.jsonl.gz')
    add('captures', data['captures'], source='capture-status.json')
    add('configs', len(data['configs']), source='configs.json')
    add('rounds', data['protocol']['rounds'], source='protocol.json')
    add('max-relative-l2', f"{data['max_relative_l2']:.5f}")
    add('positions-64', '64', source='protocol.json')
    add('positions-1024', '1024', source='protocol.json')
    return out

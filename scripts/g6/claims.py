"""Named document quantities derived from the imported G6 records."""
from .evidence import BASE


def span(values, places, unit=''):
    low, high = f'{min(values):.{places}f}', f'{max(values):.{places}f}'
    return (low if low == high else f'{low}–{high}') + unit


def quantities(data):
    from doc_claims import Quantity
    out = {}

    def add(key, value, en='', zh=None, source='power.jsonl.gz'):
        out['g6.' + key] = Quantity(str(value), en, en if zh is None else zh, BASE + '/' + source)

    for n in sorted({w['input_N'] for w in data['widths']}):
        add(f'n.{n}', str(n) if n < 1000 else f'{n // 1024}K', source='protocol.json')
    lengths = sorted({w['input_N'] for w in data['widths']})
    label = lambda n: str(n) if n < 1000 else f'{n // 1024}K'
    add('short-contexts', label(lengths[0]) + '–' + label(lengths[2]), source='protocol.json')
    add('long-contexts', label(lengths[3]) + '–' + label(lengths[-1]), source='protocol.json')
    widths = data['widths']
    add('q4-q8-speed', span([w['speed_q4_over_q8'] for w in widths], 3, '×'), source='requests.jsonl.gz')
    add('q4-q8-ane-energy', span([w['ane_energy_q4_over_q8'] for w in widths], 2, '×'))
    add('q4-q8-components-energy', span([w['components_energy_q4_over_q8'] for w in widths], 2, '×'))
    add('decode-gpu-energy', span([b['J_per_token']['gpu'] for w in widths for b in (w['q8'], w['q4'])], 3), 'J/token')
    for w in widths:
        add(f"q4-q8-speed.{w['input_N']}", f"{w['speed_q4_over_q8']:.3f}×", source='requests.jsonl.gz')

    repeat = data['repeat_blocks']
    add('repeat.speed', span([b['speed_vs_g4a'] for b in repeat], 2, '×'), source='requests.jsonl.gz')
    add('repeat.decode-energy', span([b['energy_vs_g4a'] for b in repeat if b['mode'] == 'decode'], 2, '×'))
    add('repeat.prefill-energy', span([b['energy_vs_g4a'] for b in repeat if b['mode'] == 'prefill'], 2, '×'))
    pairs = {(p['mode'], p['context']): p for p in data['repeat_pairs']}
    lengths = sorted({p['context'] for p in data['repeat_pairs']})
    both = lambda mode, ns: [v for n in ns for v in (pairs[(mode, n)]['energy_ratio'], pairs[(mode, n)]['g4a_energy_ratio'])]
    add('two-run.short-prefill-energy-x', span(both('prefill', lengths[:3]), 2, '×'))
    add('two-run.long-prefill-energy-x', span(both('prefill', lengths[3:]), 2, '×'))
    add('two-run.decode-energy-x', span(both('decode', lengths), 2, '×'))
    for (mode, n), p in pairs.items():
        add(f'repeat.{mode}.{n}.energy-x', f"{p['energy_ratio']:.2f}×")

    for r in data['w4']:
        n = r['input_N']
        add(f'w4.decode.{n}.rate', f"{r['w4']['decode_q8']['rate']:.2f}", 'token/s', source='requests.jsonl.gz')
        add(f'w4.decode.{n}.gpu-energy', f"{r['w4']['decode_q8']['J_per_token']['gpu']:.1f}", 'J/token')
        add(f'w4.prefill.{n}.rate', f"{r['w4']['prefill_block']['rate']:.0f}", 'token/s', source='requests.jsonl.gz')
    add('w4.fp16-faster-decode', span([r['fp16_faster_decode_q8'] for r in data['w4']], 1, '×'), source='requests.jsonl.gz')
    add('w4.fp16-faster-prefill', span([r['fp16_faster_prefill'] for r in data['w4']], 1, '×'), source='requests.jsonl.gz')
    w4_admissions = [p for p in data['placement'].values() if p['weights'] == 'w4']
    add('w4.admission-ane-requests', sum(p['ane_requests'] for p in w4_admissions), source='asset-identity.json')
    add('w4.metal-compiles', max(p['metal_shader_compiles'] for p in w4_admissions), source='asset-identity.json')
    kl = max(r['first_output_reference']['KL_ref_to_actual'] for p in w4_admissions for r in p['results'].values()
             if 'KL_ref_to_actual' in r['first_output_reference'])
    add('w4.reference-kl', f'{kl:.5f}', source='asset-identity.json')
    add('w4.quantization-kl-1k', f"{data['reference']['cases']['reading_1024']['w4_vs_fp16_reference']['KL_ref_to_actual']:.2f}",
        source='asset-identity.json')
    add('idle-power', span([i['mean_W']['components'] for i in data['idle']], 2), 'W')
    add('requests', data['request_count'], source='requests.jsonl.gz')
    add('power-samples', f"{data['power_samples']:,}")
    add('blocks', data['blocks'], source='summary-recorded.json')
    add('captures', data['captures'], source='capture-status.json')
    return out

"""Named document quantities derived from the imported G4 A records and disk observations."""
import json
import re
from pathlib import Path

from .disk import EVIDENCE, derive_disk
from .evidence import BASE, derive


def label(n):
    return str(n) if n < 1000 else f'{n // 1024}K'


def span(values, places, unit=''):
    return f'{min(values):.{places}f}–{max(values):.{places}f}{unit}'


def quantities(data, observed, root):
    from doc_claims import Quantity
    from g3.evidence import synthetic_reference
    out = {}

    def add(key, value, en='', zh=None, source='requests.jsonl.gz'):
        out['g4a.' + key] = Quantity(str(value), en, en if zh is None else zh,
                                     source if source.startswith(('findings/', 'results/')) else BASE + '/' + source)

    lengths = data['protocol']['inputs_N']
    pairs = {(p['mode'], p['context']): p for p in data['pairs']}
    for (mode, n), p in pairs.items():
        stem = f'{mode}.{n}.'
        places = 1 if mode == 'prefill' else 2
        for arm in ('ane', 'gpu'):
            b = p[arm]
            add(stem + arm + '-rate', f"{b['rate']:,.{places}f}", 'token/s')
            if mode == 'prefill':
                add(stem + arm + '-energy', f"{b['J_per_token']['components'] * 1000:.2f}", 'mJ/token', source='power.jsonl.gz')
            else:
                add(stem + arm + '-energy', f"{b['J_per_token']['components']:.3f}", 'J/token', source='power.jsonl.gz')
            add(stem + arm + '-cpu-median', f"{b['cpu_median_W']:.2f}", 'W', source='power.jsonl.gz')
        add(stem + 'gpu-faster', f"{p['gpu_speedup']:.2f}×")
        add(stem + 'energy-x', f"{p['energy_ratio']:.2f}×", source='power.jsonl.gz')
        add(stem + 'energy-x-without-cpu', f"{p['energy_ratio_without_cpu']:.2f}×", source='power.jsonl.gz')
    prefill = [pairs[('prefill', n)] for n in lengths]
    decode = [pairs[('decode', n)] for n in lengths]
    add('decode-gpu-faster', span([p['gpu_speedup'] for p in decode], 1, '×'))
    add('prefill-gpu-faster', span([p['gpu_speedup'] for p in prefill], 1, '×'))
    add('short-prefill-energy-x', span([p['energy_ratio'] for p in prefill[:3]], 2, '×'), source='power.jsonl.gz')
    add('long-prefill-energy-x', span([p['energy_ratio'] for p in prefill[3:]], 2, '×'), source='power.jsonl.gz')
    add('decode-energy-x', span([p['energy_ratio'] for p in decode], 2, '×'), source='power.jsonl.gz')
    add('decode-energy-x-without-cpu', span([p['energy_ratio_without_cpu'] for p in decode], 2, '×'), source='power.jsonl.gz')
    add('ane-arm-gpu-decode-energy', span([p['ane']['J_per_token']['gpu'] for p in decode], 2), 'J/token', source='power.jsonl.gz')
    add('ane-arm-gpu-decode-share', span([100 * p['ane']['J_per_token']['gpu'] / p['ane']['J_per_token']['components'] for p in decode], 0, '%'),
        source='power.jsonl.gz')
    add('ane-arm-gpu-prefill-share', span([100 * p['ane']['J_per_token']['gpu'] / p['ane']['J_per_token']['components'] for p in prefill], 0, '%'),
        source='power.jsonl.gz')
    add('ane-arm-ane-decode-energy', span([p['ane']['J_per_token']['ane'] for p in decode], 2), 'J/token', source='power.jsonl.gz')

    comparison = {c['context']: c for c in data['comparison']}
    for n, c in comparison.items():
        add(f'g3.decode.{n}.ane-rate', f"{c['g3_ane']['decode_rate_256']:.2f}", 'token/s', source=G3_SOURCE)
        add(f'g3.prefill.{n}.ane-rate', f"{c['g3_ane']['prefill_rate']:,.1f}", 'token/s', source=G3_SOURCE)
        add(f'vs-g3.decode.{n}.speed', f"{c['ane_decode_speed']:.2f}×")
        add(f'vs-g3.prefill.{n}.speed', f"{c['ane_prefill_speed']:.2f}×")
        add(f'vs-g3.decode.{n}.energy', f"{c['ane_decode_energy']:.2f}×", source='power.jsonl.gz')
        add(f'vs-g3.prefill.{n}.energy', f"{c['ane_prefill_energy']:.2f}×", source='power.jsonl.gz')
    long_decode = [comparison[n] for n in lengths if n >= 2048]
    long_prefill = [comparison[n] for n in lengths if n >= 4096]
    add('vs-g3.long-decode-speed', span([c['ane_decode_speed'] for c in long_decode], 2, '×'))
    add('vs-g3.long-prefill-speed', span([c['ane_prefill_speed'] for c in long_prefill], 2, '×'))
    add('vs-g3.long-decode-energy', span([c['ane_decode_energy'] for c in long_decode], 2, '×'), source='power.jsonl.gz')
    add('vs-g3.long-prefill-energy', span([c['ane_prefill_energy'] for c in long_prefill], 2, '×'), source='power.jsonl.gz')
    add('vs-g3.short-decode-speed', span([comparison[n]['ane_decode_speed'] for n in lengths[:2]], 2, '×'))
    add('vs-g3.gpu-speed', span([v for c in comparison.values() for v in (c['gpu_decode_speed'], c['gpu_prefill_speed'])], 2, '×'))

    rows = {(r['mode'], r['context'], r['arm']): r for r in data['implied']}
    for n in lengths:
        for arm in ('ane', 'gpu'):
            pre, dec = rows[('prefill', n, arm)], rows[('decode', n, arm)]
            add(f'prefill.{n}.{arm}-tflops', f"{pre['projection_TFLOP_per_second']:.1f}", 'TFLOP/s', source='requests.jsonl.gz')
            add(f'prefill.{n}.{arm}-total-tflops', f"{pre['projection_TFLOP_per_second'] + pre['attention_TFLOP_per_second']:.1f}",
                'TFLOP/s', source='requests.jsonl.gz')
            add(f'decode.{n}.{arm}-read', f"{dec['weight_GB_per_second'] + dec['kv_GB_per_second']:.0f}", 'GB/s', source='requests.jsonl.gz')
    for arm in ('ane', 'gpu'):
        add(f'{arm}-total-tflops-span', span([rows[('prefill', n, arm)]['projection_TFLOP_per_second'] +
                                               rows[('prefill', n, arm)]['attention_TFLOP_per_second'] for n in lengths], 1),
            'TFLOP/s')
        add(f'{arm}-read-span', span([rows[('decode', n, arm)]['weight_GB_per_second'] + rows[('decode', n, arm)]['kv_GB_per_second']
                                      for n in lengths], 0), 'GB/s')
    work = data['work']
    add('projection-parameters', f"{work['projection_parameters'] / 1e9:.2f}", 'billion', source=G3_SOURCE)
    add('projection-parameters-cn', f"{work['projection_parameters'] / 1e8:.1f}", 'hundred million', '亿', source=G3_SOURCE)
    add('weight-bytes', f"{work['weight_bytes'] / 1e9:.2f}", 'GB', source=G3_SOURCE)
    add('synthetic-fp16', f'{synthetic_reference(root):.1f}', 'T source-equivalent ops/s', 'T 源图等效 ops/s',
        source='results/fresh/throughput.json')

    protocol = data['protocol']
    for n in lengths:
        add(f'n.{n}', label(n), source='protocol.json')
    add('contexts', ' / '.join(f'{n:,}' for n in lengths), source='protocol.json')
    add('capacities', ' / '.join(f'{c:,}' for c in protocol['ane_capacities']), source='protocol.json')
    add('decode-steps', protocol['decode_steps'], source='protocol.json')
    add('repetitions', protocol['full_repetitions'], source='protocol.json')
    add('prefill-counts', ' / '.join(str(protocol['prefill_counts'][str(n)]) for n in lengths), source='protocol.json')
    add('margin', protocol['energy_interior_margin_seconds'], 's', '秒', source='protocol.json')
    add('quiet', protocol['quiet_seconds'], 's', '秒', source='protocol.json')
    add('primary-lag', protocol['primary_extra_lag_ns'] // 10 ** 9, 's', '秒', source='protocol.json')
    add('gpu-capacity', f"{protocol['gpu_capacity']:,}", source='protocol.json')
    add('blocks', len(data['blocks']), source='summary-recorded.json')
    add('requests', data['request_count'])
    add('power-samples', f"{data['power_samples']:,}", source='power.jsonl.gz')
    add('short-contexts', label(lengths[0]) + '–' + label(lengths[2]), source='protocol.json')
    add('long-contexts', label(lengths[3]) + '–' + label(lengths[-1]), source='protocol.json')
    add('decode-long-contexts', label(lengths[2]) + '–' + label(lengths[-1]), source='protocol.json')
    add('all-contexts', label(lengths[0]) + '–' + label(lengths[-1]), source='protocol.json')

    disk = EVIDENCE + '/observations.json'
    add('disk.macos', f"{observed['macos']['version']} ({observed['macos']['build']})", source=disk)
    add('disk.held-files', observed['held_files'], source=disk)
    add('disk.held', f"{observed['held_GiB']:.1f}", 'GiB', source=disk)
    add('disk.file-size', span(observed['large_GiB'], 2), 'GiB', source=disk)
    add('disk.large-files', observed['large_files'], source=disk)
    add('disk.oldest', observed['oldest'][0], source=disk)
    add('disk.released', f"{observed['released_GiB']:.1f}", 'GiB', source=disk)
    add('disk.free-before', f"{observed['free_before_GiB']:.1f}", 'GiB', source=disk)
    add('disk.free-after', f"{observed['free_after_GiB']:.1f}", 'GiB', source=disk)
    add('disk.per-ane-host', span([-v for v in observed['ane_change_before_restart_GiB']], 1), 'GiB', source='disk.json')
    add('disk.after-restart', f"{-sum(observed['ane_change_after_restart_GiB']) / len(observed['ane_change_after_restart_GiB']):.1f}",
        'GiB', source='disk.json')
    add('disk.reclaim', f"{observed['reclaim_GiB']:.1f}", 'GiB', source=disk)
    add('disk.reclaim-files', observed['reclaim_files'], source=disk)
    add('disk.gate', f"{observed['gate_required_GiB']:.0f}", 'GiB', source='disk.json')
    add('disk.gate-usable', f"{observed['gate_waiting_usable_GiB']:.1f}", 'GiB', source='disk.json')
    add('disk.gate-waits', observed['gate_waits'], source='disk.json')
    return out


G3_SOURCE = 'results/historical/g3-qwen3-4b/requests.jsonl.gz'


def check(root, data=None, observed=None):
    from doc_claims import check_document
    root = Path(root)
    data = data if data is not None else derive(root / BASE, root)
    observed = observed if observed is not None else derive_disk(data, root)
    catalog = quantities(data, observed, root)
    required = json.loads((root / 'scripts/g4a/claims.json').read_text())
    errors = []
    for name, locations in required.items():
        language = 'zh' if name.endswith('zh-CN.md') or '/zh/' in name else 'en'
        body = (root / name).read_text()
        # Other claim families in a document are checked by their own registry.
        body = re.sub(r'<!-- claim:(?!g4a\.)[^>]+-->.*?<!-- /claim -->', '', body, flags=re.DOTALL)
        errors.extend(check_document(body, catalog, language, required=locations, name=name))
    return errors

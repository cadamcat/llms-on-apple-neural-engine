"""Named quantities from G5 timing samples and imported precision controls."""
from doc_claims import Quantity
from .evidence import BASE, _span_pct


def quantities(data):
    out = {}

    def add(name, value, source='evidence.json'):
        out['g5.' + name.replace('_', '-')] = Quantity(str(value), '', '', BASE + '/' + source)

    n, speed, ms = data['numerics'], data['speed'], data['absolute_ms']
    seeds = n['random_three_seeds']
    errors = lambda arms: [row['relative_l2'] for arm in arms for row in seeds[arm]]
    add('random.dense-l2', _span_pct(errors(['dense']), 2))
    add('random.block-l2', _span_pct(errors(['kv_unrolled', 'kv_streamed']), 2))
    add('random.post-pv-l2', _span_pct(errors(['post_pv_kv_unrolled', 'post_pv_kv_streamed']), 3))
    add('cpu.dense-l2', f"{100 * n['cpu_fp16_same_expression_seed0']['dense']:.3f}%")
    add('pv.l2', f"{100 * n['isolated_pv']['relative_l2']:.2f}%")
    add('pv.probability-span', '–'.join(f'{v:.6f}' for v in n['isolated_pv']['P_range_nonzero']))
    for scale, digits in (('16', 2), ('64', 3)):
        add('pv.scale-' + scale + '-l2', f"{100 * n['pv_scale_cpu_division'][scale]:.{digits}f}%")
    for kind in ('full', 'half'):
        add('constant.' + kind + '-unnormalized-l2',
            f"{100 * n['constant_v_pv']['weighted_value/' + kind + '-unnormalized']['relative_l2']:.2f}%")
    for arm in ('kv_unrolled', 'post1024', 'dense'):
        add('near-zero.' + arm + '-l2', f"{100 * n['near_zero_half'][arm]['relative_l2']:.2f}%")
    for model, digits in (('rounded_half_products_wide_sum', (3, 2)),
                           ('serial_half_accumulation', (2, 2)),
                           ('products_flushed_below_half_min_normal', (1, 1))):
        for side, places in zip(('vs_truth', 'vs_device'), digits):
            add('rounding.' + model + '.' + side, f"{100 * n['simple_fp16_models_row0'][model][side]:.{places}f}%")
    for name, value in ms.items():
        add('time.' + name, f'{value:.2f}', 'timings.json.gz')
    for name, value in speed.items():
        add('speed.' + name, f'{value:.3f}×', 'timings.json.gz')
    transfer = [ms[a + '.with_transfer'] - ms[a + '.resident'] for a in ('dense', 'kv_unrolled', 'post4096', 'post1024')]
    add('transfer-ms-span', f'{min(transfer):.1f}–{max(transfer):.1f}', 'timings.json.gz')
    add('timed-operations', f"{data['timed_operations']:,}", 'timings.json.gz')
    return out

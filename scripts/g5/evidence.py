"""Recompute G5 attention timing comparisons and expose its numerical fields."""
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 'results/historical/g5-attention'


def require(condition, name):
    if not condition:
        raise ValueError(name)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def load(bundle):
    bundle = Path(bundle or ROOT / BASE)
    provenance = json.loads((bundle / 'provenance.json').read_text())
    for name, digest in provenance['products'].items():
        require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest,
                'g5_product_hash.' + name)
    evidence = json.loads((bundle / 'evidence.json').read_text())
    with gzip.open(bundle / 'timings.json.gz', 'rt') as stream:
        timings = json.load(stream)
    return evidence, timings


def block_means(blocks):
    """Mean ms per operation for each (process, pair, mode, arm) block."""
    out = {}
    for block in blocks:
        require(block['calls_per_operation'] >= 1 and block['operation_ns'], 'g5_block_shape')
        key = block['process'], block['pair'], block['mode'], block['arm']
        require(key not in out, 'g5_duplicate_block')
        out[key] = statistics.fmean(block['operation_ns']) / 1e6
    return out


def paired(means, mode, baseline, candidate):
    """Candidate speed against the baseline for every process pair: baseline ms / candidate ms."""
    pairs = sorted({(p, q) for p, q, m, _ in means if m == mode})
    return [means[p, q, mode, baseline] / means[p, q, mode, candidate] for p, q in pairs]


def derive(bundle=None):
    evidence, timings = load(bundle)
    reported = evidence['reported']
    chunked = block_means(timings['dense_vs_post_pv_b1024'])
    single = block_means(timings['post_pv_b4096_vs_b1024'])
    speed, absolute = {}, {}
    for mode in ('resident', 'with_transfer'):
        for arm in ('kv_unrolled', 'kv_streamed'):
            value = statistics.median(paired(chunked, mode, 'dense', arm))
            require(close(value, reported['dense_vs_post_pv_b1024'][mode][arm]),
                    f'g5_dense_speed.{mode}.{arm}')
            speed[f'{arm}_vs_dense.{mode}'] = value
        value = statistics.median(paired(single, mode, 'post4096', 'post1024'))
        require(close(value, reported['post_pv_b4096_vs_b1024'][mode]), f'g5_block_speed.{mode}')
        speed[f'b1024_vs_b4096.{mode}'] = value
        for arm in ('post4096', 'post1024'):
            value = statistics.median(v for (_, _, m, a), v in single.items() if (m, a) == (mode, arm))
            require(close(value, reported['absolute_ms'][mode][arm]), f'g5_absolute_ms.{mode}.{arm}')
            absolute[f'{arm}.{mode}'] = value
        for arm in ('dense', 'kv_unrolled', 'kv_streamed'):
            absolute[f'{arm}.{mode}'] = statistics.median(
                v for (_, _, m, a), v in chunked.items() if (m, a) == (mode, arm))
    operations = sum(len(b['operation_ns']) for blocks in timings.values() for b in blocks)
    return {'numerics': evidence['numerics'], 'shape': evidence['shape'],
            'screen': evidence['screen'], 'packages': evidence['packages'],
            'speed': speed, 'absolute_ms': absolute, 'timed_operations': operations}


def _span_pct(values, digits):
    return f'{100 * min(values):.{digits}f}–{100 * max(values):.{digits}f}%'


def quoted_values(data):
    """Numbers the documents quote, formatted from the recomputed and imported records."""
    n, speed, ms = data['numerics'], data['speed'], data['absolute_ms']
    seeds = n['random_three_seeds']
    l2 = lambda rows: [r['relative_l2'] for r in rows]
    dense = _span_pct(l2(seeds['dense']), 2)
    blocks = _span_pct(l2(seeds['kv_unrolled']) + l2(seeds['kv_streamed']), 2)
    post = _span_pct(l2(seeds['post_pv_kv_unrolled']) + l2(seeds['post_pv_kv_streamed']), 3)
    scale = n['pv_scale_cpu_division']
    constant = n['constant_v_pv']
    require(constant['weighted_value/full-normalized']['output_max'] == 0
            and constant['weighted_value/half-normalized']['output_max'] == 0, 'g5_normalized_zero_output')
    require(math.isclose(scale['64'], n['pv_scale_64_device_division']), 'g5_device_division')
    models = n['simple_fp16_models_row0']
    transfer = [ms[f'{a}.with_transfer'] - ms[f'{a}.resident'] for a in ('dense', 'kv_unrolled', 'post4096', 'post1024')]
    page = [
        dense, blocks, post, f"{100 * n['cpu_fp16_same_expression_seed0']['dense']:.3f}%",
        f"{100 * n['isolated_pv']['relative_l2']:.2f}%",
        '–'.join(f'{v:.6f}' for v in n['isolated_pv']['P_range_nonzero']),
        f"{100 * scale['16']:.2f}%", f"{100 * scale['64']:.3f}%",
        f"{100 * constant['weighted_value/full-unnormalized']['relative_l2']:.2f}%",
        f"{100 * constant['weighted_value/half-unnormalized']['relative_l2']:.2f}%",
        f"{100 * n['near_zero_half']['kv_unrolled']['relative_l2']:.2f}%",
        f"{100 * n['near_zero_half']['post1024']['relative_l2']:.2f}%",
        f"{100 * n['near_zero_half']['dense']['relative_l2']:.2f}%",
        f"{100 * models['rounded_half_products_wide_sum']['vs_truth']:.3f}%",
        f"{100 * models['rounded_half_products_wide_sum']['vs_device']:.2f}%",
        f"{100 * models['serial_half_accumulation']['vs_truth']:.2f}%",
        f"{100 * models['serial_half_accumulation']['vs_device']:.2f}%",
        f"{100 * models['products_flushed_below_half_min_normal']['vs_truth']:.1f}%",
        f"{100 * models['products_flushed_below_half_min_normal']['vs_device']:.1f}%",
        *[f"{ms[f'{a}.{m}']:.2f}" for a in ('dense', 'kv_unrolled', 'kv_streamed')
          for m in ('resident', 'with_transfer')],
        f"{speed['kv_unrolled_vs_dense.resident']:.3f}×",
        f"{ms['post4096.resident']:.2f}", f"{ms['post1024.resident']:.2f}",
        f"{speed['b1024_vs_b4096.resident']:.3f}×", f"{speed['b1024_vs_b4096.with_transfer']:.3f}×",
        f'{min(transfer):.1f}–{max(transfer):.1f}', f"{data['timed_operations']:,}",
    ]
    index = [dense, f"{100 * scale['64']:.3f}%", f"{speed['kv_unrolled_vs_dense.resident']:.3f}×",
             f"{speed['b1024_vs_b4096.resident']:.3f}×"]
    return {'findings/attention-product-precision/README.md': page,
            'findings/README.md': index,
            'docs/VALIDATION.md': [f"{data['timed_operations']:,}"]}


def measurements(data):
    ms, speed, n = data['absolute_ms'], data['speed'], data['numerics']
    lines = ['## G5 weight-free attention', '',
             'Q64 over 4,096 keys, 32 query and 8 key-value heads of size 128. Times are the median of '
             'block-mean ms per operation over three processes; speeds are medians of paired block ratios. '
             'Relative L2 values are imported. [Finding](../findings/attention-product-precision/).', '',
             '| Path | Resident (ms) | With layout conversion and readback (ms) | Speed vs reference path, resident / with transfer |',
             '|---|---:|---:|---:|']
    rows = [('dense', 'dense softmax → P @ V', '—'),
            ('kv_unrolled', 'post-PV, 1,024-key blocks, one graph',
             f"{speed['kv_unrolled_vs_dense.resident']:.4f}× / {speed['kv_unrolled_vs_dense.with_transfer']:.4f}× vs dense"),
            ('kv_streamed', 'post-PV, 1,024-key blocks, four calls',
             f"{speed['kv_streamed_vs_dense.resident']:.4f}× / {speed['kv_streamed_vs_dense.with_transfer']:.4f}× vs dense"),
            ('post4096', 'post-PV, one 4,096-key block (later round)', '—'),
            ('post1024', 'post-PV, 1,024-key blocks (later round)',
             f"{speed['b1024_vs_b4096.resident']:.4f}× / {speed['b1024_vs_b4096.with_transfer']:.4f}× vs one block")]
    for arm, label, ratio in rows:
        lines.append(f"| {label} | {ms[arm + '.resident']:.4f} | {ms[arm + '.with_transfer']:.4f} | {ratio} |")
    lines += ['', '| Numerical case | Relative L2 |', '|---|---:|']
    for arm, rows_ in n['random_three_seeds'].items():
        values = ', '.join(f'{100 * r["relative_l2"]:.4f}%' for r in rows_)
        lines.append(f'| {arm}, seeds 0–2 | {values} |')
    lines.append(f"| isolated P @ V, uniform P | {100 * n['isolated_pv']['relative_l2']:.4f}% |")
    for factor, value in n['pv_scale_cpu_division'].items():
        lines.append(f"| P × {factor}, divided on the CPU | {100 * value:.4f}% |")
    for name, block in n['constant_v_pv'].items():
        lines.append(f"| constant V, {name} | {100 * block['relative_l2']:.4f}% |")
    labels = {'dense': 'dense, P × 512', 'kv_unrolled': 'post-PV, one 4,096-key block',
              'post1024': 'post-PV, 1,024-key blocks'}
    for arm, block in n['near_zero_half'].items():
        lines.append(f"| V = 0.00025, {labels[arm]} | {100 * block['relative_l2']:.4f}% |")
    lines.append('')
    return '\n'.join(lines)

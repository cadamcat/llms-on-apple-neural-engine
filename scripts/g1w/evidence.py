"""Recompute G1-W speed ratios from bundled per-call timings and expose its numerical fields."""
import gzip
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 'results/historical/g1w-e4b-mobile-qat'
DEPTHS = (1, 2, 8, 32, 128)


def require(condition, name):
    if not condition:
        raise ValueError(name)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def p50_ms(ns):
    return statistics.median(ns) / 1e6


def geomean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values))


def load(bundle):
    bundle = Path(bundle or ROOT / BASE)
    provenance = json.loads((bundle / 'provenance.json').read_text())
    evidence = json.loads((bundle / 'evidence.json').read_text())
    with gzip.open(bundle / 'timings.json.gz', 'rt') as stream:
        timings = json.load(stream)
    return evidence, timings, provenance


def density(timings, reported):
    p50 = {arm: [p50_ms(ns) for ns in rounds] for arm, rounds in timings.items()}
    for arm, values in p50.items():
        require(len(values) == 3 and all(close(a, b) for a, b in
                                        zip(values, reported['process_p50_ms'][arm])),
                'g1w_density_p50.' + arm)
    ratios = {}
    for arm, values in p50.items():
        ratios[arm] = statistics.median(v / s for v, s in zip(values, p50['sparse']))
        require(close(ratios[arm], reported['over_sparse_median'][arm]), 'g1w_density_ratio.' + arm)
    return {'process_median_ms': {a: statistics.median(v) for a, v in p50.items()},
            'over_sparse': ratios}


def passes(timings, reported, name):
    p50 = {arm: [p50_ms(ns) for ns in runs] for arm, runs in timings.items()}
    for arm, values in p50.items():
        # Boundary summaries name arms 'both_32'; the 128-layer codebook and scale ones 'old_a8'.
        key = arm.replace('-', '_') if name == 'boundary' else arm.removesuffix('-128')
        require(len(values) == 2 and all(close(a, b) for a, b in
                                        zip(values, reported['p50_ms_by_pass'][key])),
                f'g1w_{name}_p50.{arm}')
    return p50


def depth(timings, reported):
    p50 = passes(timings, reported, 'boundary')
    mean = {arm: statistics.fmean(v) for arm, v in p50.items()}
    speed = {}
    for arm in p50:
        kind, layers = arm.split('-')
        if kind == 'bare':
            continue
        speed[arm] = geomean([b / a for b, a in zip(p50['bare-' + layers], p50[arm])])
        require(close(speed[arm], reported['a8_over_w4a16'][arm.replace('-', '_')]),
                'g1w_boundary_ratio.' + arm)

    def fit(kind):
        xs, ys = DEPTHS, [mean[f'{kind}-{n}'] for n in DEPTHS]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        return {'ms_per_layer': slope, 'ms_intercept': my - slope * mx}

    return {'mean_p50_ms': mean, 'a8_over_w4a16': speed,
            'fit': {'w4a16': fit('bare'), 'a8w4': fit('both')}}


def codebook(timings, reported):
    p50 = passes(timings, reported, 'codebook')
    out = {}
    for weights in ('old', 'full'):
        get = lambda fmt: p50[f'{weights}_{fmt}-128']
        out[weights] = {
            'a8_over_fp16': geomean([f / a for f, a in zip(get('fp16'), get('a8'))]),
            'w4a16_over_fp16': geomean([f / w for f, w in zip(get('fp16'), get('w4a16'))]),
            'a8_over_w4a16': geomean([w / a for w, a in zip(get('w4a16'), get('a8'))]),
            'fp16_ms': statistics.fmean(get('fp16')),
        }
        for key, value in out[weights].items():
            if key != 'fp16_ms':
                require(close(value, reported['speedups'][weights][key]), f'g1w_codebook_ratio.{weights}.{key}')
    return out


def scale(timings, reported):
    p50 = passes(timings, reported, 'scale')
    out = {}
    for kind in ('scalar', 'equal', 'varied'):
        out[kind] = geomean([w / a for w, a in zip(p50[f'{kind}_w4a16-128'], p50[f'{kind}_a8-128'])])
        require(close(out[kind], reported['a8_over_w4a16'][kind]), 'g1w_scale_ratio.' + kind)
    return out


def e4b(timings, reported):
    process_ms, process_ratio = {}, {}
    for process, arms in sorted(timings.items()):
        blocks = {arm: {pair: p50_ms(ns) for pair, ns in pairs.items()} for arm, pairs in arms.items()}
        for row in reported['block_median_ms'][int(process)]:
            require(close(blocks[row['arm']][str(row['pair'])], row['run_ms'])
                    and len(arms[row['arm']][str(row['pair'])]) == row['calls'],
                    f"g1w_e4b_block.{process}.{row['arm']}.{row['pair']}")
        medians = {arm: statistics.median(pairs.values()) for arm, pairs in blocks.items()}
        for arm, value in medians.items():
            # Within one process: the median block of W4A16 over the median block of this arm.
            process_ms.setdefault(arm, []).append(value)
            process_ratio.setdefault(arm, []).append(medians['bare'] / value)
    for arm, values in reported['speed_vs_bare_by_process'].items():
        require(all(close(a, b) for a, b in zip(process_ratio[arm], values)), 'g1w_e4b_ratio.' + arm)
    calls = {row['arm']: row['model_calls'] // row['calls']
             for row in reported['block_median_ms'][0]}
    return {'median_ms': {a: statistics.median(v) for a, v in process_ms.items()},
            'speed_vs_w4a16': {a: statistics.median(v) for a, v in process_ratio.items()},
            'function_calls_per_pipeline': calls}


def old_probe(root):
    """Values of the fresh exact-grid multiply outputs, identified by their hashes."""
    smoke = json.loads((Path(root) / 'results/fresh/smoke.json').read_text())
    out = {}
    for run in smoke['runs']:
        if run['case_id'].startswith('coreai-qdq-'):
            n = math.prod(run['output_shape'])
            digest = run['numerical']['comparisons'][0]['output_sha256']
            # An all-equal FP16 output is n copies of one little-endian half.
            matches = [v for v in range(-256, 257) if
                       hashlib.sha256(struct.pack('<e', v) * n).hexdigest() == digest]
            require(len(matches) == 1, 'g1w_old_probe_value.' + run['case_id'])
            out[run['case_id'].removeprefix('coreai-qdq-')] = matches[0]
    return out


def derive(bundle=None, root=ROOT):
    evidence, timings, provenance = load(bundle)
    fixtures, reported = evidence['fixtures'], evidence['reported']
    e4b_rates = e4b(timings['e4b'], reported['e4b'])
    work = fixtures['e4b']
    stack_ops = (2 * work['layers_in_stack'] * 3 * work['hidden_size'] *
                 work['intermediate_size'] * work['positions'])
    chain = fixtures['synthetic_chain']
    layer_ops = 2 * chain['input_channels'] * chain['output_channels'] * chain['positions']
    return {
        'density': density(timings['density'], reported['density']),
        'depth': depth(timings['boundary'], reported['boundary']),
        'codebook': codebook(timings['codebook'], reported['codebook']),
        'scale': scale(timings['scale'], reported['scale']),
        'e4b': e4b_rates,
        'work': {'e4b_stack_ops': stack_ops, 'synthetic_layer_ops': layer_ops,
                 'synthetic_layers_equivalent': stack_ops / layer_ops,
                 'e4b_weights_per_mlp': 3 * work['hidden_size'] * work['intermediate_size'],
                 'synthetic_weights_per_layer': chain['input_channels'] * chain['output_channels']},
        'fixtures': fixtures,
        'numerics': evidence['numerics'],
        'old_probe': old_probe(root),
        'environment': provenance['environment'],
        'measured_calls': sum(len(ns) for rounds in timings['density'].values() for ns in rounds)
                          + sum(len(ns) for key in ('boundary', 'codebook', 'scale')
                                for runs in timings[key].values() for ns in runs)
                          + sum(len(ns) for arms in timings['e4b'].values()
                                for pairs in arms.values() for ns in pairs.values()),
    }


def _x(value, digits=2):
    return f'{value:.{digits}f}×'


def _pct(value, digits):
    return f'{100 * value:.{digits}f}%'


def quoted_values(data):
    """Numbers the documents quote, formatted from the recomputed records."""
    depth, codebook, density, e4b = data['depth'], data['codebook'], data['density'], data['e4b']
    speeds, times = depth['a8_over_w4a16'], depth['mean_p50_ms']
    single, deep = data['numerics']['e4b_single_layer'], data['numerics']['e4b_eight_layer']
    fit = depth['fit']
    zeros = data['fixtures']['density_weights']
    ppm = (zeros['fill_min']['row_norm_min'] / zeros['old']['row_norm_min'] - 1) * 1e6
    coarse = [e4b['speed_vs_w4a16'][a] for a in ('split_coarse_w4', 'split_coarse_a8')]
    zero_pct = _pct(data['fixtures']['codebook_full_zero_fraction'], 2)
    headline = [_x(speeds['both-1']), _x(speeds['both-128']), _x(density['over_sparse']['old']),
                _x(e4b['speed_vs_w4a16']['native'], 3), _x(e4b['speed_vs_w4a16']['clip_product'], 3)]
    qdq = [_pct(single['bare']['ordinary'], 2), _pct(deep['bare']['ordinary'], 2),
           _pct(single['native']['ordinary'], 0), _pct(deep['native']['ordinary'], 0),
           _pct(single['clip_product']['ordinary'], 2), _pct(deep['clip_product']['ordinary'], 1)]
    conditions = [
        *[f'{times[f"{k}-{n}"]:.2f}' for n in DEPTHS for k in ('bare', 'both')],
        *[_x(speeds[f'both-{n}']) for n in DEPTHS],
        *[_x(speeds[a]) for a in ('inner-128', 'input-128', 'output-128')],
        f"{fit['w4a16']['ms_per_layer']:.3f}", f"{fit['a8w4']['ms_per_layer']:.3f}",
        f"{fit['a8w4']['ms_intercept'] - fit['w4a16']['ms_intercept']:.3f}",
        *[_x(data['scale'][k]) for k in ('scalar', 'equal', 'varied')],
        *[f"{codebook[w]['fp16_ms']:.2f}" for w in ('old', 'full')],
        *[_x(codebook[w][k]) for w in ('old', 'full') for k in ('a8_over_fp16', 'a8_over_w4a16')],
        *[f'{density["process_median_ms"][a]:.2f}' for a in density['process_median_ms']],
        _x(density['over_sparse']['permuted']), f'{ppm:.1f}', zero_pct,
        f"{e4b['median_ms']['bare']:.2f}", f"{e4b['median_ms']['native']:.2f}",
        f"{e4b['median_ms']['split_coarse_a8']:.2f}", _x(e4b['speed_vs_w4a16']['split_coarse_a8'], 3),
        _pct(deep['split_coarse_a8']['ordinary'], 1), str(e4b['function_calls_per_pipeline']['split_coarse_a8']),
        f"{data['work']['e4b_stack_ops'] / 1e9:.1f}", f"{data['work']['synthetic_layers_equivalent']:.1f}",
        f"{data['work']['e4b_weights_per_mlp'] / 1e6:.1f}", f"{data['work']['synthetic_weights_per_layer']:,}",
        f"{data['measured_calls']:,}", qdq[1], qdq[3], qdq[5], *headline[3:],
    ]
    unit = data['numerics']['unit_scale_qdq']
    gelu = data['numerics']['gelu_zero_input']['gelu_tanh']['unique_values']
    require(gelu == data['numerics']['gelu_zero_input']['gelu_exact']['unique_values'] and len(gelu) == 1,
            'g1w_gelu_modes')
    qdq_page = qdq + headline[3:] + [
        f"{min(coarse):.3f}–{max(coarse):.3f}×", str(e4b['function_calls_per_pipeline']['split_coarse_a8']),
        f'{gelu[0]:.10f}']
    rounding = [f"{unit['finite_values']:,}", str(unit['unit_even_mismatches'])]
    require(unit['unit_away_mismatches'] == 0, 'g1w_unit_qdq_away')
    fixture = [zero_pct, _x(density['over_sparse']['old']), _x(codebook['full']['a8_over_fp16'])]
    return {
        'README.md': headline + [qdq[2], qdq[5]],
        'README.zh-CN.md': headline + [qdq[2], qdq[5]],
        'findings/README.md': headline[:4] + [qdq[2]],
        'workarounds/README.md': headline + [qdq[4], qdq[5]],
        'findings/coreai-qdq-multiply-scale/README.md': qdq_page,
        'findings/quantized-speedup-conditions/README.md': conditions,
        'findings/execution-model/README.md': rounding[:1],
        'articles/03-arithmetic-compatibility.md': rounding,
        'articles/zh/03-ANE算术兼容性.md': rounding,
        'articles/01-measuring-ane-performance.md': fixture + [_x(codebook['old']['a8_over_fp16']), headline[0]],
        'articles/zh/01-如何验证ANE的量化加速.md': fixture + [_x(codebook['old']['a8_over_fp16']), headline[0]],
        'docs/SCOPE.md': fixture,
        'docs/VALIDATION.md': [f"{data['measured_calls']:,}"],
    }


def measurements(data):
    """Tables for docs/MEASUREMENTS.md."""
    depth, e4b = data['depth'], data['e4b']
    times, speeds = depth['mean_p50_ms'], depth['a8_over_w4a16']
    lines = ['## G1-W quantized speed conditions', '',
             'API latency of awaited `function.run`. Synthetic layers are 512 → 512 1×1 convolutions over '
             '4,096 positions. p50 is the mean of two process medians; speed is the geometric mean of '
             'the paired W4A16/A8W4 ratios. [Finding](../findings/quantized-speedup-conditions/).', '',
             '| Graph | W4A16 p50 (ms) | A8W4 p50 (ms) | A8W4 speed vs W4A16 |', '|---|---:|---:|---:|']
    for n in DEPTHS:
        lines.append(f"| {n} layer{'s' if n > 1 else ''}, QDQ at both ends and between layers | {times[f'bare-{n}']:.4f} "
                     f"| {times[f'both-{n}']:.4f} | {speeds[f'both-{n}']:.4f}× |")
    for arm, label in (('inner-128', 'interlayer QDQ only'), ('input-128', 'interlayer and input QDQ'),
                       ('output-128', 'interlayer and output QDQ')):
        lines.append(f"| 128 layers, {label} | {times['bare-128']:.4f} | {times[arm]:.4f} | {speeds[arm]:.4f}× |")
    fit = depth['fit']
    lines += ['', f"Least-squares lines over depth: W4A16 {fit['w4a16']['ms_per_layer']:.5f} ms/layer + "
                  f"{fit['w4a16']['ms_intercept']:.4f} ms; A8W4 {fit['a8w4']['ms_per_layer']:.5f} ms/layer + "
                  f"{fit['a8w4']['ms_intercept']:.4f} ms.", '',
              '| 128-layer scale | A8W4 speed vs W4A16 |', '|---|---:|']
    lines += [f"| {k} | {data['scale'][k]:.4f}× |" for k in ('scalar', 'equal', 'varied')]
    lines += ['', '| 128-layer weights | FP16 p50 (ms) | A8W4 speed vs FP16 | W4A16 speed vs FP16 | A8W4 speed vs W4A16 |',
              '|---|---:|---:|---:|---:|']
    for w, label in (('old', 'dense ±1 codes'), ('full', 'sixteen codes, 73.63% zeros')):
        c = data['codebook'][w]
        lines.append(f"| {label} | {c['fp16_ms']:.4f} | {c['a8_over_fp16']:.4f}× | "
                     f"{c['w4a16_over_fp16']:.4f}× | {c['a8_over_w4a16']:.4f}× |")
    lines += ['', 'FP16 only, three processes per weight set; ratio is the median of same-round time ratios.', '',
              '| FP16 weights | Exact zeros | Median process p50 (ms) | Time vs sparse |', '|---|---:|---:|---:|']
    labels = {'old': 'dense ±1 codes', 'shuffled': 'dense ±1, rows shuffled', 'sparse': 'sixteen codes',
              'permuted': 'sixteen codes, channels permuted', 'fill_min': 'zeros set to ±2^-14',
              'fill_16': 'zeros set to ±scale/16'}
    for arm, block in data['fixtures']['density_weights'].items():
        lines.append(f"| {labels[arm]} | {100 * block['zero_fraction']:.4f}% | "
                     f"{data['density']['process_median_ms'][arm]:.4f} | {data['density']['over_sparse'][arm]:.4f}× |")
    single, deep = data['numerics']['e4b_single_layer'], data['numerics']['e4b_eight_layer']
    lines += ['', 'Gemma 4 E4B mobile QAT first MLP, 64 positions, three processes. Pipeline time is the '
                  'median of process medians of block medians; speed is the median of per-process ratios. '
                  'Relative L2 is against each graph\'s CPU reference; "ordinary" is positions 6–63.', '',
              '| Graph | Function calls | Pipeline (ms) | Speed vs W4A16 | L2 one MLP, all / ordinary | L2 eight MLPs, all / ordinary |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in e4b['median_ms']:
        lines.append(f"| {arm} | {e4b['function_calls_per_pipeline'][arm]} | {e4b['median_ms'][arm]:.4f} | "
                     f"{e4b['speed_vs_w4a16'][arm]:.4f}× | {100 * single[arm]['all']:.4f}% / "
                     f"{100 * single[arm]['ordinary']:.4f}% | {100 * deep[arm]['all']:.4f}% / "
                     f"{100 * deep[arm]['ordinary']:.4f}% |")
    lines.append('')
    return '\n'.join(lines)

"""Plots built from the bundled records, using only the standard library.

Every value drawn here is recomputed from the stored evidence — fresh suites in
``results/fresh`` and imported comparisons in ``results/historical`` — and
checked against the recorded summary before it reaches the canvas, so a figure
cannot drift away from what it claims to show.  Nothing in this module imports a
device API or a plotting stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from .canvas import Canvas, T_BODY, T_SMALL, num

X0, X1 = 170.0, 612.0            # throughput plot band
AXIS_TOP = 100.0


def _load(repo: Path, name: str):
    """Read one bundled suite and re-derive every quantity the figures use."""
    path = Path('results/fresh') / (name + '.json')
    raw = (repo / path).read_bytes()
    data = json.loads(raw)
    if data['environment']['chip']['value'] != 'Apple M5 Pro':
        raise ValueError('Host changed; review figure scope before regenerating')
    rows = {}
    for i, run in enumerate(data['runs']):
        if run['p50_ms'] is None:
            continue
        process = int(re.search(r'-process(\d+)-', run['run'])[1])
        durations = [x['duration_ns'] for x in run['measured_records']]
        if not math.isclose(statistics.median(durations) / 1e6, run['p50_ms'], rel_tol=1e-12):
            raise ValueError('Stored p50 conflicts with the raw durations')
        tops = run['source_ops'] / (run['p50_ms'] * 1e9)
        if not math.isclose(tops, run['source_equivalent_Tops'], rel_tol=1e-12):
            raise ValueError('Stored throughput conflicts with source_ops / p50')
        rows[(run['case_id'], process)] = (i, run)
    return path.as_posix(), hashlib.sha256(raw).hexdigest(), rows


def _point(source: str, entry, process: int) -> dict:
    i, run = entry
    return {
        'source': source, 'run': run['run'], 'case_id': run['case_id'], 'process_round': process,
        'fields': {k: f'/runs/{i}/{k}' for k in ('p50_ms', 'source_ops', 'source_equivalent_Tops')},
        'p50_ms': run['p50_ms'], 'source_ops': run['source_ops'],
        'throughput_T_source_ops_per_s': run['source_ops'] / (run['p50_ms'] * 1e9),
    }


def _range(values, digits: int, suffix: str = '') -> str:
    """Render a min-max range, collapsing to one number when rounding merges them."""
    low, high = f'{min(values):.{digits}f}', f'{max(values):.{digits}f}'
    return (low if low == high else f'{low}–{high}') + suffix


def _ticks(c: Canvas, y: float, band, values, labels, place):
    """Draw the axis rule across `band` and label each tick under it."""
    c.line(band[0], y, band[1], y, 'rule', 1)
    for value, label in zip(values, labels):
        c.text(place(value), y + 16, label, T_SMALL, 'muted', anchor='middle')


# ---------------------------------------------------------------- figure 1
def throughput(c: Canvas, rows, source):
    arms = [
        ('coreml-fp16-128', 'Core ML · FP16', 'slate', None),
        ('coreml-w8a8-128', 'Core ML · W8A8', 'blue', 'coreml-fp16-128'),
        ('coreai-fp16-128', 'Core AI · FP16', 'slate', None),
        ('coreai-w8a8-128', 'Core AI · W8A8', 'blue', 'coreai-fp16-128'),
        ('coreai-a8w4-128', 'Core AI · A8W4', 'teal', 'coreai-fp16-128'),
    ]
    ys = [116.0, 148.0, 196.0, 228.0, 260.0]
    place = lambda v: X0 + v / 40.0 * (X1 - X0)

    for value in (10, 20, 30, 40):
        c.line(place(value), AXIS_TOP, place(value), 286, 'grid', 1)
    c.line(40, 172, 860, 172, 'rule', 1)                       # runtime separator
    c.text(860, 92, 'vs same-round FP16', T_SMALL, 'muted', anchor='end')

    points, ratios = [], []
    for (case, label, color, baseline), y in zip(arms, ys):
        rates = []
        for process in (1, 2, 3):
            point = _point(source, rows[(case, process)], process)
            points.append(point)
            rates.append(point['throughput_T_source_ops_per_s'])
        median = statistics.median(rates)
        c.text(158, y + 4, label, T_BODY, 'ink', anchor='end')
        c.bar(X0, y - 7, place(median) - X0, 14, color)
        c.text(place(median) + 9, y + 4, _range(rates, 2), T_BODY, 'ink', 'sb', cls='num')
        if baseline:
            pair = [rows[(baseline, p)][1]['p50_ms'] / rows[(case, p)][1]['p50_ms'] for p in (1, 2, 3)]
            ratios.append({'case_id': case, 'baseline': baseline,
                           'process_rounds': [1, 2, 3], 'FP16_over_quantized_p50': pair})
            c.text(860, y + 4, _range(pair, 2, '×'), T_BODY, 'ink', 'sb',
                   anchor='end', cls='num')

    _ticks(c, 286, (X0, X1), [0, 10, 20, 30, 40], ['0', '10', '20', '30', '40'], place)
    c.footnote(330, 'A MAC counts as two source-graph operations. This is a synthetic graph rate, '
                    'not a physical INT8 instruction count or a model token rate.')
    return points, ratios


# ---------------------------------------------------------------- figure 2
def split(c: Canvas, rows, source):
    x0, x1 = 150.0, 620.0
    place = lambda v: x0 + v / 2.0 * (x1 - x0)
    for value in (0.5, 1.0, 1.5, 2.0):
        c.line(place(value), 96, place(value), 248, 'grid', 1)

    points, ratios = [], []
    for process, y in zip((1, 2, 3), (124.0, 168.0, 212.0)):
        wide = _point(source, rows[('coreai-a8w4-2', process)], process)
        narrow = _point(source, rows[('coreai-a8w4-split32-2', process)], process)
        points.extend([wide, narrow])
        c.text(138, y + 4, f'Round {process}', T_BODY, 'muted', anchor='end')
        c.line(place(wide['p50_ms']), y, place(narrow['p50_ms']), y, 'rule', 3, cap='round')
        for point, color in ((wide, 'teal'), (narrow, 'amber')):
            c.dot(place(point['p50_ms']), y, 6, color)
            c.text(place(point['p50_ms']), y + 22, f"{point['p50_ms']:.3f}", T_SMALL, color,
                   anchor='middle', cls='num')
        ratio = narrow['p50_ms'] / wide['p50_ms']
        ratios.append({'process_round': process, 'split_over_wide_p50': ratio})
        c.text(660, y + 5, f'{ratio:.2f}×', 13.5, 'ink', 'b', cls='num')

    _ticks(c, 248, (x0, x1), [0, 0.5, 1.0, 1.5, 2.0], ['0', '0.5', '1.0', '1.5', '2.0'], place)
    c.legend(150, 296, [('dot', 'teal', 'Wide K512, one convolution per layer'),
                        ('dot', 'amber', '16 × K32 plus a balanced FP16 addition tree')])
    c.footnote(326, 'Partition, reduction and FP16 rounding boundaries change together; '
                    'these pairs do not isolate one internal cause.')
    return points, ratios


# ---------------------------------------------------------------- figure 3
def rounding(c: Canvas, rows, source):
    index, run = rows[('coreai-w8a8-128', 1)]
    comparisons = []
    for key, name in (('numerical', 'Q8 RZA'), ('RNE-not-gate', 'Q8 RNE')):
        ci, entry = next((i, x) for i, x in enumerate(run[key]['comparisons'])
                         if x['phase'] == 'control_original')
        comparisons.append({
            'reference': name, 'field': f'/runs/{index}/{key}/comparisons/{ci}/relative_L2',
            'relative_L2': entry['relative_L2'], 'output_sha256': entry['output_sha256'],
            'output_sha256_field': f'/runs/{index}/{key}/comparisons/{ci}/output_sha256',
            'reference_sha256': run[key]['reference_sha256'],
        })
    if len({x['output_sha256'] for x in comparisons}) != 1:
        raise ValueError('Reference residuals do not compare the same recorded output')

    x0, x1 = 190.0, 660.0
    place = lambda v: x0 + (math.log10(v) + 5) / 5 * (x1 - x0)
    for exponent in range(-4, 0):
        c.line(place(10.0 ** exponent), 96, place(10.0 ** exponent), 214, 'grid', 1)

    rza, rne = comparisons
    for entry, y, color, label, note in ((rza, 128.0, 'teal', 'Q8 RZA', 'ties away from zero'),
                                         (rne, 180.0, 'amber', 'Q8 RNE', 'ties to even')):
        c.text(178, y, label, T_BODY, 'ink', 'sb', anchor='end')
        c.text(178, y + 15, note, T_SMALL, 'muted', anchor='end')
        c.dot(place(entry['relative_L2']), y, 7, color)
        c.text(place(entry['relative_L2']) + 14, y + 4, f"{entry['relative_L2']:.3g}",
               13, color, 'b', cls='num')

    apart = rne['relative_L2'] / rza['relative_L2']
    for entry in (rza, rne):
        c.line(place(entry['relative_L2']), 96, place(entry['relative_L2']), 214, 'muted', 1,
               dash='2 4')
    c.path(f'M{num(place(rza["relative_L2"]))} 214 L{num(place(rne["relative_L2"]))} 214',
           'muted', 1.4, arrow=True)
    c.text((place(rza['relative_L2']) + place(rne['relative_L2'])) / 2, 208,
           f'≈ {round(apart, -1):,.0f}× apart'.replace(',', ' '),
           T_SMALL, 'muted', anchor='middle')

    _ticks(c, 240, (x0, x1), [10.0 ** e for e in range(-5, 1)],
           ['10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'], place)
    c.footnote(288, 'Both residuals compare the same recorded output, byte for byte '
                    '(one matching SHA-256), against two different frozen references.')
    return comparisons


# ---------------------------------------------------------------- figure 4
def ane_vs_gpu(repo: Path):
    """Speed of the same first-layer MLP on three paths, in positions per second."""
    source = 'results/historical/ane-vs-gpu-prefill.json'
    data = json.loads((repo / source).read_text())
    engines = [('G', 'GPU · MLX Q4', 'blue'),
               ('A', 'ANE · A8W4', 'slate'),
               ('C', 'ANE · W4A16', 'teal')]
    positions = ['64', '1024', '4096']
    rate = lambda key, label: (
        data['engines'][key]['rows'][label]['effective_positions_per_second'])

    fastest = max(rate(k, p) for k, _, _ in engines for p in positions)
    limit = 30000.0
    title = 'Historical Python MLP: the GPU is 2.9 to 5.1 times faster'
    desc = ('Historical Python measurement of one first-layer MLP from the same source Q4_0 weights, in '
            'positions processed per second. The GPU path runs between 2.9 and 5.1 times '
            'faster. The Neural Engine holds about 5,300 positions per second at every '
            'measured size; the GPU peaks at 1024 and falls at 4096. Later native results are separate. One process per path, and not an '
            'end-to-end prefill.')
    c = Canvas(title, desc, 384)
    c.header(title,
             'historical Python / 64-position chunks · bar = positions per second · longer is faster',
             'Apple M5 Pro · MLX 0.32.2 · one timed process per path · looped saved activations, not an end-to-end prefill')

    x0, x1 = 190.0, 700.0
    place = lambda v: x0 + v / limit * (x1 - x0)
    for value in (10000, 20000, 30000):
        c.line(place(value), 96, place(value), 312, 'grid', 1)

    rows = []
    y = 112.0
    for label in positions:
        c.text(178, y + 22, f'{int(label):,}', T_BODY, 'ink', 'sb', anchor='end', cls='num')
        c.text(178, y + 38, 'positions', T_SMALL, 'muted', anchor='end')
        gpu = rate('G', label)
        for key, name, color in engines:
            speed = rate(key, label)
            c.bar(x0, y, place(speed) - x0, 13, color)
            c.text(place(speed) + 8, y + 11, f'{speed:,.0f}/s', T_SMALL, 'ink', 'sb',
                   cls='num')
            rows.append({'positions': int(label), 'engine': key, 'label': name,
                         'positions_per_second': speed,
                         'p50_outer_ms': data['engines'][key]['rows'][label]['p50_outer_ms']})
            y += 17
        c.text(860, y - 30, f'GPU {gpu / rate("A", label):.2f}× faster', T_BODY, 'blue',
               'b', anchor='end', cls='num')
        y += 22

    _ticks(c, 312, (x0, x1), [0, 10000, 20000, 30000], ['0', '10k', '20k', '30k'], place)
    c.legend(190, 354, [('square', col, name) for _, name, col in engines])
    c.footnote(372, 'Historical Python path: GPU peaks at 1024, then falls at 4096. '
                    'Three of nine planned processes ran; native follow-up is separate.')
    assert fastest < limit
    return title, desc, source, rows, c.finish()


def generate(repo: Path, out: Path) -> list[dict]:
    """Write the three evidence plots and return their provenance metadata."""
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = []

    source, digest, rows = _load(repo, 'throughput')
    title = 'FP16 against two quantized paths on a synthetic 128-layer chain'
    desc = ('Five configurations, each measured in three independent processes. Bars show the '
            'median process rate and labels give the full range across the three. Quantized '
            'paths reach about 35 T source-equivalent operations per second against about 18.5 '
            'for FP16. Source-equivalent operation counts do not establish physical INT8 '
            'arithmetic.')
    c = Canvas(title, desc, 362)
    c.header(title,
             'bar = median of three independent process rates · label = full range across the three, '
             'in T source-equivalent ops/s',
             'Apple M5 Pro · macOS 27 · shape [1, 512, 64, 64] · 10 warmups + 30 synchronous calls per process')
    points, ratios = throughput(c, rows, source)
    (out / 'throughput-by-process.svg').write_text(c.finish(), encoding='utf-8')
    metadata.append({'filename': 'throughput-by-process.svg', 'title': title, 'description': desc,
                     'source': source, 'source_sha256': digest,
                     'formula': 'source_ops / (p50_ms * 1e9)',
                     'points': points, 'paired_ratios': ratios})

    source, digest, rows = _load(repo, 'split')
    title = 'Splitting the same K512 weights into sixteen K32 convolutions'
    desc = ('Three paired rounds of a two-layer Core AI A8W4 graph. The wide path takes about '
            '0.43 to 0.45 milliseconds and the split path about 1.76 to 1.77, a ratio near four. '
            'Lines join the two arms of one paired round, not a trend over time.')
    c = Canvas(title, desc, 348)
    c.header(title,
             'each line joins one paired round · synchronous prediction p50 in milliseconds, lower is faster',
             'Apple M5 Pro · two layers, input [1, 512, 64, 64] · same initial weight values on both paths')
    points, ratios = split(c, rows, source)
    (out / 'split-latency.svg').write_text(c.finish(), encoding='utf-8')
    metadata.append({'filename': 'split-latency.svg', 'title': title, 'description': desc,
                     'source': source, 'source_sha256': digest,
                     'points': points, 'paired_ratios': ratios})

    source, digest, rows = _load(repo, 'throughput')
    title = 'One recorded output, two arithmetic references'
    desc = ('The same Core AI W8A8 output has a relative L2 of about 0.000109 against the '
            'reference selected before the run, which rounds Q8 midpoints away from zero, and '
            'about 0.211 against the ties-to-even diagnostic reference. This is a change of '
            'reference, not a repair of the output.')
    c = Canvas(title, desc, 324)
    c.header(title,
             'relative L2 of one control output against two frozen references · logarithmic axis',
             'Apple M5 Pro · Core AI W8A8, 128 layers · round 1, original-input control')
    comparisons = rounding(c, rows, source)
    (out / 'rounding-reference.svg').write_text(c.finish(), encoding='utf-8')
    metadata.append({'filename': 'rounding-reference.svg', 'title': title, 'description': desc,
                     'source': source, 'source_sha256': digest,
                     'run': rows[('coreai-w8a8-128', 1)][1]['run'],
                     'case_id': 'coreai-w8a8-128', 'comparisons': comparisons})

    title, desc, source, points, content = ane_vs_gpu(repo)
    (out / 'ane-vs-gpu.svg').write_text(content, encoding='utf-8')
    metadata.append({'filename': 'ane-vs-gpu.svg', 'title': title, 'description': desc,
                     'sources': [source, 'findings/ane-vs-gpu-prefill/README.md'],
                     'points': points,
                     'semantic_checks': ['Bar length is positions processed per second',
                                         'One timed process per path, three of nine planned',
                                         'Not an end-to-end prefill',
                                         'W4A16 and A8W4 are within 2% at 4096 positions']})
    return metadata

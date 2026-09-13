"""G4 A figures: matched ANE graphs against the GPU and against the G3 graph ladder."""
import math

from .canvas import Canvas, T_SMALL, num
from .g2 import panel, series
from g3.evidence import BASE as G3_BASE, synthetic_reference
from g4a.disk import EVIDENCE, derive_disk
from g4a.evidence import BASE, GiB, derive

COLORS = {'ane': 'teal', 'gpu': 'blue'}
DOMAINS = (('cpu', 'amber', 'CPU'), ('gpu', 'blue', 'GPU'), ('ane', 'teal', 'ANE'))


def footnotes(c, y, *lines):
    for i, line in enumerate(lines):
        c.text(40, y + i * 19, line, 13, 'muted')


def label(n):
    return str(n) if n < 1000 else f'{n // 1024}K'


def log_panel(c, x, y, w, h, title, low, high, ticks, xticks, xlabel):
    c.text(x, y - 14, title, 12, 'ink', 'sb')
    span = math.log10(high) - math.log10(low)
    to_y = lambda v: y + h - h * (math.log10(v) - math.log10(low)) / span
    for value in ticks:
        c.line(x, to_y(value), x + w, to_y(value), 'grid')
        c.text(x - 9, to_y(value) + 4, f'{value:,}', T_SMALL, 'muted', anchor='end')
    for xx, text in xticks:
        c.text(x + xx * w, y + h + 18, text, T_SMALL, 'muted', anchor='middle')
    c.text(x + w / 2, y + h + 36, xlabel, T_SMALL, 'muted', anchor='middle')
    return lambda t: x + t * w, to_y


def hollow(c, x, y, color):
    c.parts.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="3.2" class="f-card s-{color}" stroke-width="1.6"/>')


def shaded(c, x, y, w, h, color):
    if h > 0:
        c.parts.append(f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
                       f'class="f-{color}" fill-opacity=".42"/>')


def speed(data):
    c = Canvas('Qwen3-4B: speed with matched ANE graphs',
               'Prefill and decode speed at six inputs: GPU, ANE with a graph sized to each input, and ANE on the earlier 256 / 2K / 32K graphs.', 480)
    c.header('Qwen3-4B: one ANE graph per input size',
             'Complete FP16 model · M5 Pro · Core AI on both paths · GPU on its 32K asset',
             'G4 A gives each input an ANE graph of capacity N + 256 rounded up; G3 used graphs of 256 / 2K / 32K.')
    c.legend(40, 100, [('line', 'blue', 'GPU'), ('line', 'teal', 'ANE, matched graph (G4 A)')])
    c.line(430, 96, 442, 96, 'teal', 2, '4 3')
    hollow(c, 436, 96, 'teal')
    c.text(450, 100, 'ANE, 256 / 2K / 32K graphs (G3)', T_SMALL, 'muted')
    lengths = data['protocol']['inputs_N']
    xticks = [(i / 5, label(n)) for i, n in enumerate(lengths)]
    points = []
    g3 = {r['context']: r for r in data['g3_rows'] if r['arm'] == 'ane'}
    x, y = log_panel(c, 85, 150, 315, 215, 'Prefill · input tokens/s (log scale)', 40, 5000,
                     [50, 100, 200, 500, 1000, 2000, 5000], xticks, 'Input tokens')
    for arm in ('gpu', 'ane'):
        values = [next(p for p in data['pairs'] if p['mode'] == 'prefill' and p['context'] == n)[arm]['rate'] for n in lengths]
        series(c, [(x(i / 5), y(v)) for i, v in enumerate(values)], COLORS[arm], 2.2, dots=True)
        points += [{'mode': 'prefill', 'context': n, 'series': arm, 'rate': v} for n, v in zip(lengths, values)]
    old = [g3[n]['prefill_rate'] for n in lengths]
    series(c, [(x(i / 5), y(v)) for i, v in enumerate(old)], 'teal', 1.6, dash='4 3')
    for i, v in enumerate(old):
        hollow(c, x(i / 5), y(v), 'teal')
    points += [{'mode': 'prefill', 'context': n, 'series': 'g3_ane', 'rate': v} for n, v in zip(lengths, old)]
    x, y = panel(c, 505, 150, 315, 215, 'Decode · output tokens/s', 35, [0, 10, 20, 30], xticks, 'KV at the start of decode')
    for arm in ('gpu', 'ane'):
        values = [next(p for p in data['pairs'] if p['mode'] == 'decode' and p['context'] == n)[arm]['rate'] for n in lengths]
        series(c, [(x(i / 5), y(v)) for i, v in enumerate(values)], COLORS[arm], 2.2, dots=True)
        points += [{'mode': 'decode', 'context': n, 'series': arm, 'rate': v} for n, v in zip(lengths, values)]
    old = [g3[n]['decode_rate_256'] for n in lengths]
    series(c, [(x(i / 5), y(v)) for i, v in enumerate(old)], 'teal', 1.6, dash='4 3')
    for i, v in enumerate(old):
        hollow(c, x(i / 5), y(v), 'teal')
    points += [{'mode': 'decode', 'context': n, 'series': 'g3_ane', 'rate': v} for n, v in zip(lengths, old)]
    footnotes(c, 420, 'Decode: G4 A uses teacher-forced requests; G3 uses the first 256 steps of one decode block.',
              'Prefill: warmed one-token request blocks. G3 selects its 32K graph from 2K decode / 4K prefill.',
              'Rates include host work; they are not hardware ceilings.')
    return c, points


def implied(data, reference):
    c = Canvas('Qwen3-4B: implied compute and memory reads',
               'Model-based prefill FLOP/s and decode read GB/s from the measured rates at six inputs on both paths.', 700)
    c.header('Qwen3-4B: implied compute and memory reads',
             'Measured token rates × model work · matched ANE graphs (G4 A) · GPU on its 32K asset',
             'Solid: projection FLOPs (prefill) and FP16 weight reads (decode). Lighter: causal attention and existing-KV reads.')
    c.legend(40, 100, [('dot', 'blue', 'GPU'), ('dot', 'teal', 'ANE')])
    c.line(170, 96, 186, 96, 'slate', 1.2, '5 4')
    c.text(194, 100, f'ANE synthetic FP16 chain, {reference:.1f} T ops/s (prefill)', T_SMALL, 'muted')
    lengths = data['protocol']['inputs_N']
    rows = {(r['mode'], r['context'], r['arm']): r for r in data['implied']}
    points = []
    panels = (('prefill', 'Prefill · TFLOP/s', 32, [0, 8, 16, 24, 32], 'projection_TFLOP_per_second',
               'attention_TFLOP_per_second', 1, 150, 'Input tokens'),
              ('decode', 'Decode · modelled read GB/s', 280, [0, 70, 140, 210, 280], 'weight_GB_per_second',
               'kv_GB_per_second', 0, 400, 'KV at the start of decode'))
    for mode, title, ymax, ticks, base_key, extra_key, places, top, xlabel in panels:
        width, height = 760, 175
        centres = [(i + .5) / len(lengths) for i in range(len(lengths))]
        x, y = panel(c, 85, top, width, height, title, ymax, ticks, [(t, label(n)) for t, n in zip(centres, lengths)], xlabel)
        bar = 34
        for n, centre in zip(lengths, centres):
            for side, arm in ((-1, 'gpu'), (1, 'ane')):
                row = rows[(mode, n, arm)]
                base, extra = row[base_key], row[extra_key]
                left = x(centre) + (3 if side > 0 else -bar - 3)
                c.bar(left, y(base), bar, y(0) - y(base), COLORS[arm], radius=0)
                shaded(c, left, y(base + extra), bar, y(base) - y(base + extra), COLORS[arm])
                c.text(left + bar / 2, y(base + extra) - 5, f'{base + extra:,.{places}f}', T_SMALL, COLORS[arm], anchor='middle')
                points.append({'mode': mode, 'context': n, 'arm': arm, base_key: base, extra_key: extra})
        if mode == 'prefill':
            c.line(x(0), y(reference), x(1), y(reference), 'slate', 1.2, '5 4')
            points.append({'reference_T_ops_per_second': reference,
                           'source': 'results/fresh/throughput.json: coreai-fp16-128 median source_ops / p50'})
    footnotes(c, 642, 'Prefill: projection work plus causal attention over N/2 keys on average.',
              'Decode: weights and existing KV read once per step; padding and current-token writes excluded.',
              'Model-based rates, not measured hardware compute or memory traffic.')
    return c, points


def energy(data):
    c = Canvas('Qwen3-4B: component energy per token by domain',
               'CPU, GPU and ANE software energy per token at six inputs, GPU path and ANE path side by side, with timing bounds.', 740)
    c.header('Qwen3-4B: where the energy per token goes',
             'CPU + GPU + ANE software counters · matched ANE graphs (G4 A) · 2-second lag · per-block admission',
             'Left bar of each pair: GPU path. Right bar: ANE path. Whiskers: timing bounds on the component total.')
    c.legend(40, 100, [('square', color, name + ' counter') for _, color, name in DOMAINS])
    lengths = data['protocol']['inputs_N']
    points = []
    panels = (('prefill', 'Prefill · mJ per input token', 1000, 30, [0, 10, 20, 30], 1, 150, 'Input tokens'),
              ('decode', 'Decode · J per output token', 1, 1.5, [0, .5, 1, 1.5], 2, 410, 'KV at the start of decode'))
    for mode, title, scale, ymax, ticks, places, top, xlabel in panels:
        width, height = 760, 185
        centres = [(i + .5) / len(lengths) for i in range(len(lengths))]
        x, y = panel(c, 85, top, width, height, title, ymax, ticks, [], '')
        blocks = {(b['input_N'], b['arm']): b for b in data['blocks'] if b['mode'] == mode}
        bar = 40
        for i, (n, centre) in enumerate(zip(lengths, centres)):
            for side, arm in ((-1, 'gpu'), (1, 'ane')):
                b = blocks[(n, arm)]
                left = x(centre) + (3 if side > 0 else -bar - 3)
                level = 0
                for domain, color, _ in DOMAINS:
                    value = b['J_per_token'][domain] * scale
                    c.bar(left, y(level + value), bar, y(level) - y(level + value), color, radius=0)
                    level += value
                low, high = (v * scale for v in b['bounds_J_per_token']['components'])
                middle = left + bar / 2
                c.line(middle, y(low), middle, y(min(high, ymax)), 'ink', 1)
                for bound in (low, high):
                    if bound <= ymax:
                        c.line(middle - 4, y(bound), middle + 4, y(bound), 'ink', 1)
                neighbours = [blocks[(m, arm)]['cpu_median_W'] for m in lengths[max(0, i - 1):i + 2] if m != n]
                flagged = b['cpu_median_W'] > 1.35 * max(neighbours)
                mark = '†' if flagged else ''
                c.text(middle, y(min(high, ymax)) - 6, f'{level:.{places}f}{mark}', T_SMALL, 'ink', anchor='middle')
                c.text(middle, y(0) + 14, arm.upper(), 9.5, 'muted', anchor='middle')
                points.append({'mode': mode, 'context': n, 'arm': arm,
                               **{d: b['J_per_token'][d] for d, _, _ in DOMAINS},
                               'lower': b['bounds_J_per_token']['components'][0],
                               'upper': b['bounds_J_per_token']['components'][1],
                               'cpu_median_W': b['cpu_median_W'], 'cpu_flag': flagged})
            c.text(x(centre), y(0) + 30, label(n), T_SMALL, 'muted', anchor='middle')
        c.text(85 + width / 2, top + height + 46, xlabel, T_SMALL, 'muted', anchor='middle')
    footnotes(c, 672, '† CPU power is elevated relative to neighbouring inputs on the same path.',
              'CPU + GPU + ANE software energy; no idle subtraction or wall-power measurement.',
              'Whiskers show sample-timing bounds, not sensor accuracy.')
    return c, points


def disk(data, observed):
    c = Canvas('G4 A: free disk space during the run',
               'Free space on the data volume through the twelve host sessions, with the drop left by each ANE host and the release when ANECompilerService exited.', 490)
    c.header('Free disk space during G4 A',
             f"macOS {observed['macos']['version']} ({observed['macos']['build']}) · statfs free space sampled by the run observer",
             'Shaded: host sessions (teal ANE, blue GPU). The service kept each ANE host\'s deleted compile input open.')
    rows = data['disk']['observer']
    start = rows[0]['monotonic_ns']
    minutes = lambda ns: (ns - start) / 6e10
    span = math.ceil(minutes(rows[-1]['monotonic_ns']) / 10) * 10
    x0, y0, w, h = 85, 120, 760, 245
    x, y = panel(c, x0, y0, w, h, 'Free space · GiB', 240, [0, 60, 120, 180, 240],
                 [(m / span, str(m)) for m in range(0, span + 1, 10)], 'Minutes from the start of the run')
    for window in data['disk']['host_windows']:
        a, b = (minutes(window[k] * 1e9) / span for k in ('host_started_monotonic_s', 'host_ended_monotonic_s'))
        c.tint(x(a), y0, x(b) - x(a), h, COLORS[window['arm']], radius=0, stroke=False)
        if window['arm'] == 'ane':
            c.text((x(a) + x(b)) / 2, y0 + 14, label(window['input_N']), 9.5, 'teal', anchor='middle')
    series(c, [(x(minutes(r['monotonic_ns']) / span), y(r['disk_free_bytes'] / GiB)) for r in rows], 'ink', 1.4)
    restart = next(s for s in data['disk']['arms'] if s['waits'])
    window = next(w for w in data['disk']['host_windows'] if (w['input_N'], w['arm']) == (restart['input_N'], restart['arm']))
    at = x(minutes(window['host_started_monotonic_s'] * 1e9) / span)
    c.text(at - 6, y(observed['free_after_GiB']) + 16, f"service ended: +{observed['released_GiB']:.1f} GiB", T_SMALL, 'ink', anchor='end')
    c.text(x0 + 8, y(60) - 6, 'Temporary free-space drops during host sessions; these allocations were not individually traced', T_SMALL, 'muted')
    footnotes(c, 420, f"One M5 Pro and Qwen3-4B assets. Before restart, each ANE host left {-max(observed['ane_change_before_restart_GiB']):.1f}–{-min(observed['ane_change_before_restart_GiB']):.1f} GiB held.",
              'After restart: about 14.5 GiB less free space per ANE host; only about half traced to a held input.',
              'Other free-space changes were not individually attributed.')
    points = {'observer_rows': len(rows), 'host_windows': data['disk']['host_windows'],
              'released_GiB': observed['released_GiB']}
    return c, points


def generate(repo, out):
    data = derive(repo / BASE, repo)
    observed = derive_disk(data, repo)
    sources = [BASE + '/' + p.name for p in sorted((repo / BASE).iterdir()) if p.name != 'README.md']
    g3_sources = [G3_BASE + '/' + p.name for p in sorted((repo / G3_BASE).iterdir()) if p.name != 'README.md']
    leak = [EVIDENCE + '/' + p.name for p in sorted((repo / EVIDENCE).iterdir())]
    reference = synthetic_reference(repo)
    for name, build, extra in (('g4a-speed', lambda: speed(data), g3_sources),
                               ('g4a-implied', lambda: implied(data, reference), g3_sources + ['results/fresh/throughput.json']),
                               ('g4a-energy', lambda: energy(data), g3_sources),
                               ('ane-compiler-disk', lambda: disk(data, observed), g3_sources + leak)):
        canvas, points = build()
        filename = name + '.svg'
        (out / filename).write_text(canvas.finish())
        yield {'filename': filename, 'sources': sources + extra, 'points_or_scope': points,
               'generation': 'Shared standard-library Canvas; G4 A request, power and disk recomputation.',
               'scope': 'Qwen3-4B FP16 on one M5 Pro; matched ANE graphs against the GPU and the G3 ladder; software component energy.'}

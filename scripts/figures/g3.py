"""Complete-model curves using the shared documentation SVG canvas."""
from .canvas import Canvas, T_SMALL
from .g2 import panel, series
from g3.evidence import BASE, derive, synthetic_reference

COLORS = {'ane': 'teal', 'gpu': 'blue'}


def paired_plot(data, kind):
    settings = {
        'speed': ('Qwen3-4B: prefill and decode speed', ['Input tokens/s', 'Output tokens/s'],
                  [4000, 35], [[0, 1000, 2000, 3000, 4000], [0, 10, 20, 30]]),
        'coverage-speed': ('Qwen3-4B: single-request coverage', ['Input tokens/s', 'Output tokens/s'],
                           [4000, 35], [[0, 1000, 2000, 3000, 4000], [0, 10, 20, 30]]),
        'power': ('Qwen3-4B: power during each stage', ['Mean component power (W)', 'Mean component power (W)'],
                  [55, 18], [[0, 10, 20, 30, 40, 50], [0, 5, 10, 15]]),
    }
    title, labels, limits, ticks = settings[kind]
    c = Canvas(title, 'Six measured context sizes on one M5 Pro; prefill and decode shown separately.', 450)
    c.header(title, 'Complete FP16 model · M5 Pro · Core AI on both paths · 32K KV capacity',
             'Single coverage request per point; prefill + 256 decode steps.' if kind == 'coverage-speed' else
             'Warmed stage blocks; speed, power and energy use the same measured work.')
    c.legend(320, 98, [('line', 'teal', 'ANE path'), ('line', 'blue', 'GPU path')])
    points = []
    lengths = data['protocol']['lengths']
    xticks = [(i/(len(lengths)-1), str(n) if n < 1000 else f'{n//1024}K') for i,n in enumerate(lengths)]
    for i, mode in enumerate(('prefill', 'decode')):
        x,y = panel(c, 85+i*420, 145, 315, 195, mode.title()+' · '+labels[i], limits[i], ticks[i], xticks,
                    'Input context (tokens)' if mode == 'prefill' else 'Initial KV (tokens)')
        selected = [p for p in data['pairs'] if p['mode'] == mode]
        for arm in ('gpu', 'ane'):
            values=[]
            for p in selected:
                if kind == 'coverage-speed':
                    row=next(r for r in data['coverage'] if r['arm']==arm and r['context']==p['context'])
                    value=row[mode+'_rate']
                else:
                    value=p[arm]['rate'] if kind=='speed' else p[arm]['mean_W']['components']
                values.append(value)
                points.append({'mode':mode,'context':p['context'],'arm':arm,'value':value})
            series(c, [(x(j/5),y(v)) for j,v in enumerate(values)], COLORS[arm], 2.2, dots=True)
    if kind in ('speed', 'coverage-speed'):
        c.footnote(405, 'ANE graphs: 256 / 2K / 32K contexts. Speed drops where the selected graph changes.',
                   'Measured implementations, including host work. No direct hardware bandwidth measurement.')
    else:
        c.footnote(403, 'CPU + GPU + ANE software power; no idle subtraction. Model load, warmup and recovery excluded.',
                   'Observed thermal conditions; sensor accuracy uncalibrated. Lower power need not mean lower energy.')
    return c, points


def kv_plot(data):
    c=Canvas('Qwen3-4B: decode as KV grows', 'Four successive 256-step blocks for each starting context.', 580)
    c.header('Qwen3-4B: decode as the KV cache grows', 'Each point: 256 sequential forward steps from the same decode block',
             'Horizontal axes show the midpoint KV length of each block; vertical axes show output tokens/s.')
    c.legend(320,100,[('line','teal','ANE path'),('line','blue','GPU path')])
    points=[]
    for i,p in enumerate(v for v in data['pairs'] if v['mode']=='decode'):
        n=p['context']
        x,y=panel(c,80+(i%3)*280,150+(i//3)*195,205,120,f'Initial KV: {n:,} tokens',35,[0,10,20,30],
                  [(j/3,str(n+128+256*j)) for j in range(4)])
        for arm in ('gpu','ane'):
            values=p[arm]['kv']
            series(c,[(x(j/3),y(v['rate'])) for j,v in enumerate(values)],COLORS[arm],2,dots=True)
            points.append({'context':n,'arm':arm,'blocks':values})
    c.footnote(525,'One decode request per path and starting context; points are consecutive segments, not independent runs.',
               'Free continuation, fixed forward count, including steps after EOS. Graph selection differs between the paths.')
    return c,points


def grouped_bars(c, x0, y0, w, h, title, ymax, ticks, groups, xlabel, places, paired_labels=False):
    """Two bars per context: GPU left, ANE right, with optional bound whiskers."""
    labels = [(i + 0.5) / len(groups) for i in range(len(groups))]
    x, y = panel(c, x0, y0, w, h, title, ymax, ticks, list(zip(labels, [g['label'] for g in groups])), xlabel)
    width = min(30, 0.34 * w / len(groups))
    for group, centre in zip(groups, labels):
        label_y = {arm: y(group[arm][2]) - 8 for arm in ('gpu', 'ane')}
        if paired_labels and abs(label_y['gpu'] - label_y['ane']) < 14:
            label_y['ane'] = min(label_y.values())
            label_y['gpu'] = label_y['ane'] - 14
        for side, arm in ((-1, 'gpu'), (1, 'ane')):
            value, low, high = group[arm]
            left = x(centre) + (width + 3) * (0 if side < 0 else 1) - width - 1.5
            c.bar(left, y(value), width, y(0) - y(value), COLORS[arm])
            c.text(left + width / 2, label_y[arm] if paired_labels else y(value) - 6,
                   f'{value:,.{places}f}', T_SMALL,
                   COLORS[arm] if paired_labels else 'muted', anchor='middle')
            if low != high:
                middle = left + width / 2
                c.line(middle, y(low), middle, y(high), 'ink', 1)
                for bound in (low, high):
                    c.line(middle - 4, y(bound), middle + 4, y(bound), 'ink', 1)
    return x, y


def energy_bars(data):
    c = Canvas('Qwen3-4B: energy per token',
               'Component energy per token on both paths, at six context sizes, with timing bounds.', 450)
    c.header('Qwen3-4B: energy per token',
             'Complete FP16 model · M5 Pro · Core AI on both paths · warmed stage blocks',
             'CPU + GPU + ANE software energy; the same block supplies rate and energy.')
    c.legend(320, 98, [('dot', 'blue', 'GPU path'), ('dot', 'teal', 'ANE path')])
    points = []
    label = lambda n: str(n) if n < 1000 else f'{n // 1024}K'
    for i, (mode, axis, ymax, ticks, places, scale) in enumerate(
            (('prefill', 'mJ / input token', 55, [0, 10, 20, 30, 40, 50], 1, 1000),
             ('decode', 'Joules / output token', 1.5, [0, .5, 1, 1.5], 2, 1))):
        groups = []
        for pair in [p for p in data['pairs'] if p['mode'] == mode]:
            entry = {'label': label(pair['context'])}
            for arm in ('gpu', 'ane'):
                block = pair[arm]
                energy = block['energy']['components']
                entry[arm] = (block['J_per_token']['components'],
                              energy['lower_J'] / block['tokens'], energy['upper_J'] / block['tokens'])
                points.append({'mode': mode, 'context': pair['context'], 'arm': arm,
                               'J_per_token': entry[arm][0], 'lower': entry[arm][1], 'upper': entry[arm][2]})
                entry[arm] = tuple(value * scale for value in entry[arm])
            groups.append(entry)
        grouped_bars(c, 85 + i * 420, 145, 315, 195, mode.title() + ' · ' + axis, ymax, ticks, groups,
                     'Input context (tokens)' if mode == 'prefill' else 'Initial KV (tokens)', places,
                     paired_labels=True)
    c.footnote(405, 'CPU + GPU + ANE software energy; no idle subtraction. Model load, warmup and recovery excluded.',
               'Whiskers are timing bounds on the energy window, not confidence intervals. Sensor accuracy uncalibrated.')
    return c, points


def implied_bars(data, reference):
    c = Canvas('Qwen3-4B: implied compute and read bandwidth',
               'Prefill FLOP/s over projection weights and decode bytes read per step, before the ANE graph changes.', 470)
    c.header('Qwen3-4B: implied compute and read bandwidth',
             'Complete FP16 model · M5 Pro · contexts below the ANE path\'s graph change',
             'Derived from the measured rates, 3.63 billion projection weights and 8.04 GB of FP16 weights.')
    c.legend(320, 98, [('dot', 'blue', 'GPU path'), ('dot', 'teal', 'ANE path')])
    points = []
    label = lambda n: str(n) if n < 1000 else f'{n // 1024}K'
    rows = {(r['mode'], r['context'], r['arm']): r for r in data['implied']}
    lengths = data['protocol']['lengths']
    for i, (mode, axis, keep, ymax, ticks, places, field) in enumerate(
            (('prefill', 'TFLOP/s over projection weights', lengths[:3], 28, [0, 7, 14, 21, 28], 1,
              'tera_flops_per_second'),
             ('decode', 'Modelled read GB/s', lengths[:2], 280, [0, 70, 140, 210, 280], 0,
              'giga_bytes_per_second'))):
        groups = []
        for n in keep:
            entry = {'label': label(n)}
            for arm in ('gpu', 'ane'):
                value = rows[(mode, n, arm)][field]
                entry[arm] = (value, value, value)
                points.append({'mode': mode, 'context': n, 'arm': arm, field: value})
            groups.append(entry)
        x, y = grouped_bars(c, 85 + i * 420, 145, 315, 195, mode.title() + ' · ' + axis, ymax, ticks, groups,
                            'Input context (tokens)' if mode == 'prefill' else 'Initial KV (tokens)', places)
        if mode == 'prefill':
            c.line(x(0), y(reference), x(1), y(reference), 'slate', 1.2, '5 4')
            points.append({'reference_T_ops_per_second': reference,
                           'source': 'results/fresh/throughput.json: coreai-fp16-128 median source_ops / p50'})
    c.footnote(398, f'Dashed: the same machine\'s ANE synthetic FP16 chain at {reference:.1f} T ops/s, on one fixed shape.',
               'Prefill counts two FLOPs per projection weight per token and excludes attention.',
               'Decode assumes one weight/KV read per step with KV growth; current-token writes and padding excluded.',
               'Model work divided by measured time; device counters and DRAM traffic were not measured.')
    return c, points


def generate(repo, out):
    data=derive(repo/BASE)
    sources=[BASE+'/'+p.name for p in sorted((repo/BASE).iterdir()) if p.name!='README.md']
    reference=synthetic_reference(repo)
    for name in ('speed','implied','power','energy','coverage-speed','decode-kv'):
        canvas,points=(kv_plot(data) if name=='decode-kv' else energy_bars(data) if name=='energy'
                       else implied_bars(data,reference) if name=='implied' else paired_plot(data,name))
        filename='g3-'+name+'.svg'
        (out/filename).write_text(canvas.finish())
        yield {'filename':filename,
               'sources':sources + (['results/fresh/throughput.json'] if name == 'implied' else []),
               'points_or_scope':points,
               'generation':'Shared standard-library Canvas; raw G3 request and power-field recomputation.',
               'scope':'Qwen3-4B FP16 on one M5 Pro. Two Core AI implementations; software component energy.'}

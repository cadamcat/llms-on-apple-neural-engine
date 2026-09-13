"""Recompute G2 service, timing and thermal scalars from the imported rows.

The portable checks do not replay weights, prove device placement, or rerun the
original workspace audit. They preserve its evidence and independently recompute
the published scalar accounting. Only the standard library is imported.
"""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from .thermal import FIELDS, classify

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'results/historical/g2-w4a16-night'


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] * (high - position) + values[high] * (position - low)


def describe(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'p50': quantile(values, .5), 'p95': quantile(values, .95), 'p99': quantile(values, .99),
            'min': min(values) if values else None, 'max': max(values) if values else None}


def equal(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label)
        for key, value in expected.items():
            require(key in actual, label + '/' + key)
            equal(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f'{label}/{i}')
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label)
    else:
        require(actual == expected, label)


def account(events, offsets, start, end, drain, deadline):
    """All terminal arrivals form the denominator; tails use completed rows only."""
    require(len(events) == len(offsets), 'arrival count differs from frozen sequence')
    events = sorted(events, key=lambda r: r['index'])
    counts = Counter(); response = []; service = []; violations = 0
    bins = [0] * math.ceil((end - start) / 60e9)
    for index, (event, offset) in enumerate(zip(events, offsets)):
        require(event['index'] == index and event['id'] == str(index), 'missing or duplicate arrival ID')
        require(event['arrival_ns'] == start + offset and 0 <= offset < end - start, 'arrival clock')
        require(event['deadline_ns'] == deadline, 'deadline changed')
        state = event['status']
        require(state in ('completed', 'rejected', 'failed', 'cancelled'), 'unknown terminal status')
        counts[state] += 1
        violation = True
        if state == 'completed':
            finish = event['completed_ns']
            require(event['arrival_ns'] <= event['service_start_ns'] <= finish <= drain, 'service clock or drain')
            response.append((finish - event['arrival_ns']) / 1e9)
            service.append((finish - event['service_start_ns']) / 1e9)
            violation = finish - event['arrival_ns'] > deadline
            if start <= finish < end:
                bins[int((finish - start) // 60e9)] += 1
        require(event['deadline_violation'] == violation, 'incorrect violation label')
        violations += violation
    n = len(events)
    return {'arrived': n, 'completed': counts['completed'], 'rejected': counts['rejected'],
            'failed': counts['failed'], 'cancelled': counts['cancelled'],
            'all_completed': counts['completed'] == n, 'deadline_violations': violations,
            'deadline_violation_fraction': violations / n if n else None,
            'success_fraction': counts['completed'] / n if n else None,
            'observed_completed': sum(bins), 'observation_throughput_per_second': sum(bins) / ((end - start) / 1e9),
            'completed_per_minute': bins, 'response_seconds_completed_only': describe(response),
            'service_seconds_completed_only': describe(service)}


def saturated(events, start, end, drain, cap):
    bins = [0] * math.ceil((end - start) / 60e9)
    require(events and len(events) <= cap, 'saturated count/cap')
    for i, event in enumerate(events):
        require(event['index'] == i and event['id'] == str(i) and event['status'] == 'completed', 'saturated IDs/status')
        finish = event['completed_ns']
        require(start <= finish <= drain, 'saturated completion clock')
        if finish < end:
            bins[int((finish - start) // 60e9)] += 1
    censored = len(events) >= cap and events[-1]['completed_ns'] < end
    require(censored or events[-1]['completed_ns'] >= end, 'saturated stream ended early')
    return {'completed': len(events), 'observed_completed': sum(bins), 'censored': censored,
            'observation_throughput_per_second': sum(bins) / ((end - start) / 1e9),
            'completed_per_minute': bins}


def slope(points):
    require(len(points) >= 2, 'slope needs two samples')
    x = [(t - points[0][0]) / 60e9 for t, _ in points]; y = [v for _, v in points]
    mx, my = statistics.mean(x), statistics.mean(y)
    den = sum((v - mx) ** 2 for v in x)
    require(den > 0, 'slope needs distinct timestamps')
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


def native_growth(p0, positions=4096, tile=256, channels=15360):
    """Footprint change of each P0 host, and what retaining one gate output per
    gate call -- the historical Python behaviour -- would have added to a native
    ANE host. Only gate calls are projected, so the projection is a floor."""
    tiles = positions // tile
    gate_bytes = channels * tile * 2
    result = []
    for block in p0:
        host, points = block['host'], block['checkpoints']
        record = {'engine': block['engine'], 'pid': host['pid'], 'requests': host['requests'],
                  'stage_calls': host['stage_calls'],
                  'first_footprint_bytes': points[0]['footprint_bytes'],
                  'last_footprint_bytes': points[-1]['footprint_bytes'],
                  'footprint_change_bytes': points[-1]['footprint_bytes'] - points[0]['footprint_bytes']}
        if block['engine'] == 'C':
            require(host['stage_calls'] == host['requests'] * tiles * 4,
                    'native P0 stage calls are not four stages per tile per request')
            record['gate_calls'] = host['requests'] * tiles
            record['projected_retained_bytes'] = record['gate_calls'] * gate_bytes
        result.append(record)
    return result


def platform(slot, sensors, thermal):
    if slot['config']['phase'] == 'p1_gpu_coexistence':
        return 'not_applicable'
    end = slot['observe_end_ns']; start = end - int(300e9)
    if start < slot['observe_start_ns']:
        return 'undetermined'
    slopes = []
    for field in FIELDS[:2]:
        selected = {r['sample_monotonic_ns']: r[field] for r in sensors
                    if start <= r['sample_monotonic_ns'] <= end and math.isfinite(r[field])}
        ticks = sorted(selected); edges = [start, *ticks, end]
        if len(ticks) < 2 or max(b - a for a, b in zip(edges, edges[1:])) > 10e9:
            return 'undetermined'
        slopes.append(slope([(t, selected[t]) for t in ticks]))
    states = [r for r in thermal if start <= r['sample_monotonic_ns'] <= end]
    ticks = [start, *[r['sample_monotonic_ns'] for r in states], end]
    levels = {'nominal': 0, 'fair': 1, 'serious': 2, 'critical': 3}
    if not states or any(r['thermal_state'] not in levels for r in states) or max(b-a for a,b in zip(ticks,ticks[1:])) > 2e9:
        return 'undetermined'
    bins = slot['service'].get('inference', {}).get('completed_per_minute', [])[-5:]
    if len(bins) != 5 or statistics.mean(bins) <= 0 or slot['service']['inference'].get('censored'):
        return 'undetermined'
    upgrade = any(levels[b['thermal_state']] > levels[a['thermal_state']] for a,b in zip(states,states[1:]))
    stable = (max(abs(v) for v in slopes) <= .2 and
              (max(bins) - min(bins)) / statistics.mean(bins) <= .05 and not upgrade)
    return 'approximate_platform' if stable else 'not_platform'


def derive(bundle=BUNDLE, verify=True):
    protocol = load(bundle / 'protocol.json'); expected = load(bundle / 'expected-summary.json')
    if verify:
        provenance = load(bundle / 'provenance.json')
        for name, record in provenance['products'].items():
            raw = (bundle / name).read_bytes()
            require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], 'bundle identity: ' + name)
    p2 = list(rows(bundle / 'p2.jsonl.gz'))
    require(len(p2) == 45 and len({r['host_id'] for r in p2}) == 6, 'P2 cells or hosts missing')
    for cell in p2:
        records = cell['records']; measured = records[15:]
        require([r['category'] for r in records] == ['control']*5 + ['warmup']*10 + ['measured']*30, 'P2 categories')
        require([r['input_index'] for r in records[:5]] == [0,1,2,None,0], 'P2 controls')
        require([r['index'] for r in measured] == list(range(30)), 'P2 sample IDs')
        require(all(r['pid'] == cell['host']['pid'] and r['positions'] == cell['positions'] and
                    r['tile'] == cell['tile'] and r['engine'] == cell['engine'] and r['numerical_passed']
                    and r['outer_ns'] > 0 and r['worker_ns'] > 0 for r in records), 'P2 identity/numerical status')
        control = {r['input_index']: r['output_hash'] for r in records[:4]}
        require(all(r['output_hash'] == control[r['input_index']] for r in records), 'P2 repeated output hash')
        outer = [r['outer_ns']/1e9 for r in measured]
        cell['rate'] = cell['positions'] * len(measured) / sum(outer)
        equal(cell['rate'], cell['positions_per_second'], 'P2 throughput')
        equal(describe(outer), cell['outer_seconds'], 'P2 durations')
    medians = []
    for n in (64,128,256,512,1024,2048,4096):
        point = {'positions': n}
        for engine in ('C','G'):
            cells = [r for r in p2 if r['positions']==n and r['engine']==engine and not r['diagnostic']]
            require({r['round'] for r in cells} == {0,1,2} and len(cells)==3, 'P2 independent rounds')
            point[engine] = {'positions_per_second': statistics.median(r['rate'] for r in cells),
                'outer_mean_ms': statistics.median(r['outer_seconds']['mean']*1000 for r in cells),
                'process_range': [min(r['rate'] for r in cells), max(r['rate'] for r in cells)]}
        point['C_over_G'] = point['C']['positions_per_second'] / point['G']['positions_per_second']
        medians.append(point)
    equal(medians, expected['P2_process_medians'], 'P2 medians')
    slots = load(bundle/'slots.json'); require(len(slots)==34, 'slot count')
    offsets = {(r['slot_id'],r['stream']): r['offsets_ns'] for r in rows(bundle/'arrivals.jsonl.gz')}
    events_count = 0
    for name, slot in slots.items():
        events = list(rows(bundle/slot['events_file'])); events_count += len(events)
        config = slot['config']; begin,end,drain = (slot[k] for k in ('observe_start_ns','observe_end_ns','drain_end_ns'))
        require(end-begin == config['observe_seconds']*10**9, 'observation duration')
        streams = {r['stream'] for r in events}
        require(streams == set(slot['service']), 'event streams')
        for stream in streams:
            selected = [r for r in events if r['stream']==stream]
            if config['mode']=='saturated':
                result = saturated(selected,begin,end,drain,config['saturated_request_cap'])
                equal(result, {k:slot['service'][stream][k] for k in result}, name)
            else:
                result = account(selected,offsets[name,stream],begin,end,drain,selected[0]['deadline_ns'])
                equal(result,slot['service'][stream],name+'/'+stream)
            slot['service'][stream].update(result)
        if config['engine']:
            selected = sorted((r for r in events if r['stream']=='inference' and r['status']=='completed'),key=lambda r:r['index'])
            inputs = []
            for event in selected:
                meta = event['metadata']
                require(meta['input_index']==event['index']%3 and meta['numerical_passed'] and meta['positions']==1024,
                        'inference identity/numerical status')
                inputs.append([event['id'],event['index'],meta['input_index'],meta['input_file_sha256']])
            identity = hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest()
            require(identity==slot['completed_input_identity'], 'completed input identity')
            if config['mode']!='saturated':
                arrival_hash = hashlib.sha256(json.dumps(offsets[name,'inference'],separators=(',',':')).encode()).hexdigest()
                require(arrival_hash==slot['inference_arrival_identity'], 'arrival identity')
    sensors = list(rows(bundle/'sensors.jsonl.gz')); thermal = list(rows(bundle/'thermal.jsonl.gz'))
    resources = list(rows(bundle/'resources.jsonl.gz'))
    for series, clock in ((sensors,'sample_monotonic_ns'),(thermal,'sample_monotonic_ns'),(resources,'monotonic_ns')):
        require(all(a[clock] < b[clock] for a,b in zip(series,series[1:])), 'telemetry chronology')
    starts = []
    for group in protocol['groups']:
        blocks = {name:dict(window_start_ns=slots[name]['window_start_ns'],window_end_ns=slots[name]['window_end_ns'],
                   samples=sensors,thermal_samples=thermal) for name in group['members']}
        starts.append(classify(group,blocks,protocol['criteria']['thermal_start']))
    equal(starts,load(bundle/'thermal-starts.json'),'thermal starts')
    windows = load(bundle/'thermal-windows.json')
    for name, slot in slots.items():
        selected = [r for r in sensors if slot['observe_start_ns'] <= r['sample_monotonic_ns'] < slot['observe_end_ns']]
        slot['sensor_means'] = {field: statistics.mean(r[field] for r in selected) for field in FIELDS}
        for field in FIELDS:
            equal(describe([r[field] for r in selected]),windows[name]['sensors'][field],name+'/'+field)
        slot['platform'] = platform(slot,sensors,thermal)
        equal(slot['platform'],windows[name]['platform']['status'],name+'/platform')
        tail = [r for r in selected if r['sample_monotonic_ns'] >= slot['observe_end_ns']-300e9]
        slot['tail_medians'] = {field:statistics.median(r[field] for r in tail) for field in FIELDS}
    for record in expected['saturated']:
        slot = slots[record['id']]; service = slot['service']['inference']
        equal(slot['tail_medians'],record['tail_medians'],'saturated tail')
        equal(service['observation_throughput_per_second'],record['requests_per_second'],'saturated throughput')
    sat_ratio = statistics.mean(slots[n]['service']['inference']['observation_throughput_per_second'] for n in ('SAT-0-C','SAT-3-C')) / statistics.mean(slots[n]['service']['inference']['observation_throughput_per_second'] for n in ('SAT-1-G','SAT-2-G'))
    equal(sat_ratio,expected['saturated_C_over_G'],'saturated ratio')
    p3_fans = []
    for pair in expected['P3_pairs']:
        group = next(g for g in protocol['groups'] if g['id']==pair['group'])
        by = {slots[n]['config']['engine']:slots[n] for n in group['members']}
        for field, key in zip(FIELDS[:2],('cpu_mean_G_minus_C','gpu_mean_G_minus_C')):
            equal(by['G']['sensor_means'][field]-by['C']['sensor_means'][field],pair[key],'matched rate temperature')
        require(by['C']['completed_input_identity']==by['G']['completed_input_identity'] and
                by['C']['inference_arrival_identity']==by['G']['inference_arrival_identity'], 'P3 unequal work')
        p3_fans.append({'group': pair['group'], **{f'{e}_{field}': by[e]['sensor_means'][field]
                                                   for e in ('C', 'G') for field in FIELDS[2:]}})
    metrics = ('component_rss_bytes','component_footprint_bytes','foreground_rss_bytes','foreground_footprint_bytes','all_owned_rss_bytes')
    for field in metrics:
        equal(max(r[field] for r in resources),expected['resources']['peaks'][field],'resource peak/'+field)
    equal(len(resources),expected['resources']['samples'],'resource samples')
    equal(max(r['swap_growth_mib'] for r in resources),0,'swap growth')
    equal(min(r['free_percent'] for r in resources),expected['resources']['minimum_free_percent'],'free memory')
    equal(max(b['monotonic_ns']-a['monotonic_ns'] for a,b in zip(resources,resources[1:]))/1e9,
          expected['resources']['maximum_sample_gap_seconds'],'resource gap')
    p0 = load(bundle/'p0.json')
    require([r['engine'] for r in p0]==['C','G','G','C'],'P0 block order')
    for block in p0:
        require(block['passed'] and block['host']['closed'] and block['host']['requests']==527,'P0 imported lifecycle')
        require([r['measured_requests'] for r in block['checkpoints']]==[0,32,64,128,256,512],'P0 checkpoints')
    growth = native_growth(p0)
    power = load(bundle/'power-status.json')
    require(power['status']=='power_energy_undetermined' and not power['alignment']['accepted'], 'unearned energy claim')
    require({'truncated_tail','constant_origin_intersection_empty','wall_clock_discontinuity'} <= set(power['alignment']['failures']), 'power failure lost')
    valid = [b for g in starts for b in g.get('blocks', {}).values() if b['valid']]
    fan_floor = {'blocks': len(valid), **{field: statistics.median(b['medians'][field] for b in valid)
                                          for field in FIELDS[2:]}}
    return {'protocol':protocol,'p2':p2,'medians':medians,'slots':slots,'starts':starts,
            'sensors':sensors,'thermal':thermal,'resources':resources,'p0':p0,
            'saturated_ratio':sat_ratio,'expected':expected,'events_checked':events_count,
            'native_growth':growth,'p3_fans':p3_fans,'fan_floor':fan_floor}


def table(headers, records):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                       *['| '+' | '.join(map(str,row))+' |' for row in records]])


def measurements(data):
    """Short generated tables; all independent blocks remain in the evidence bundle."""
    fans = {r['group']: r for r in data['p3_fans']}; floor = data['fan_floor']
    lines = ['## G2: native W4A16 service observations','',
        'Imported from a separate 2026-09-11 run. One first-layer MLP; positions are not tokens. '
        'Ventura dynamic screensaver reported after the run; effect unquantified. '
        '[Finding](../findings/w4a16-service-tradeoffs/) · [Scope](SCOPE.md#g2-service-observations)','',
        table(['Positions','ANE positions/s','GPU positions/s','ANE / GPU'],[
            [r['positions'],f"{r['C']['positions_per_second']:,.2f}",f"{r['G']['positions_per_second']:,.2f}",f"{r['C_over_G']:.4f}×"] for r in data['medians']]),'',
        'Each rate is the median of three independent host rates; each host rate uses 30 measured calls.','',
        table(['Block','Observed / all completed','Requests/s','Tail CPU / GPU °C','Tail fan 0 / 1 RPM'],[
            [name,f"{s['service']['inference']['observed_completed']} / {s['service']['inference']['completed']}",
             f"{s['service']['inference']['observation_throughput_per_second']:.3f}",
             f"{s['tail_medians'][FIELDS[0]]:.2f} / {s['tail_medians'][FIELDS[1]]:.2f}",
             f"{s['tail_medians'][FIELDS[2]]:.0f} / {s['tail_medians'][FIELDS[3]]:.0f}"]
            for name,s in data['slots'].items() if name.startswith('SAT-')]),'',
        'Tail values are medians of the final five minutes. Saturated blocks complete unequal work; energy remains undetermined.','',
        table(['Coexistence slot','p95 ms','p99 ms','Violations / all arrivals','Thermal start'],[
            [name,f"{s['service']['foreground']['response_seconds_completed_only']['p95']*1000:.3f}",
             f"{s['service']['foreground']['response_seconds_completed_only']['p99']*1000:.3f}",
             f"{s['service']['foreground']['deadline_violations']} / {s['service']['foreground']['arrived']}",
             next(g['status'] for g in data['starts'] if name in g['blocks'])]
            for name,s in data['slots'].items() if name.startswith('P1-')]),'',
        table(['Matched-rate group','Requests per side','GPU − ANE CPU / GPU mean °C',
               'Fan 0 mean ANE / GPU RPM','Fan 1 mean ANE / GPU RPM','ANE / GPU platform'],[
            [r['group'],r['requests_each'],f"{r['cpu_mean_G_minus_C']:.2f} / {r['gpu_mean_G_minus_C']:.2f}",
             f"{fans[r['group']]['C_fan0_rpm']:.0f} / {fans[r['group']]['G_fan0_rpm']:.0f}",
             f"{fans[r['group']]['C_fan1_rpm']:.0f} / {fans[r['group']]['G_fan1_rpm']:.0f}",
             f"{r['C_platform']} / {r['G_platform']}"] for r in data['expected']['P3_pairs']]),'',
        'Six-minute observation means; these are not long-run equilibrium differences. All six thermal starts matched. '
        f"Idle fan reference: the median block-start speed across {floor['blocks']} blocks is "
        f"{floor['fan0_rpm']:.0f} / {floor['fan1_rpm']:.0f} RPM.",'']
    return '\n'.join(lines)


def sig3(value):
    """Three significant figures for prose; host-to-host spread makes more digits noise."""
    return f"{float(f'{value:.3g}'):,.0f}"


def quoted_values(data):
    """Bounded prose drift checks for G2; not a claim to parse every sentence."""
    point = next(r for r in data['medians'] if r['positions']==1024)
    slots = data['slots']
    rate = lambda s: s['service']['inference']['observation_throughput_per_second']
    sat = {e: [s for n, s in slots.items() if n.startswith('SAT-') and s['config']['engine']==e] for e in 'CG'}
    p3 = [s for n, s in slots.items() if n.startswith('P3-')]
    native = [r for r in data['native_growth'] if r['engine']=='C']
    change = sorted(abs(r['footprint_change_bytes'])/2**20 for r in native)
    diffs = [r['gpu_mean_G_minus_C'] for r in data['expected']['P3_pairs']]
    gpu_fan0 = [s['tail_medians']['fan0_rpm'] for s in sat['G']]
    calls = f"{native[0]['stage_calls']:,}"
    shrink = f'{change[0]:.0f}–{change[-1]:.0f}'
    projected = f"{native[0]['projected_retained_bytes']/2**30:.1f}"
    common = [sig3(point['C']['positions_per_second']), sig3(point['G']['positions_per_second']),
              f"{point['C_over_G']:.3f}×", f"{data['saturated_ratio']:.3f}×",
              calls, shrink, f'{min(diffs):.1f}–{max(diffs):.1f}',
              f"{round(data['fan_floor']['fan0_rpm'], -1):,.0f}",
              f'{max(rate(s) for s in p3):.1f}', f"{statistics.mean(rate(s) for s in sat['C']):.1f}",
              f"{statistics.mean(rate(s) for s in sat['G']):.1f}",
              f'{round(min(gpu_fan0), -2):,.0f}–{round(max(gpu_fan0), -2):,.0f}']
    tile_ratios=[]
    for rnd in range(3):
        cells={r['tile']:r for r in data['p2'] if r['engine']=='C' and r['round']==rnd and r['positions']==1024}
        tile_ratios.append(cells[256]['rate']/cells[64]['rate'])
    extra=[f'{min(tile_ratios):.3f}–{max(tile_ratios):.3f}×',
           f"{sum(s['service']['inference']['completed'] for n,s in slots.items() if n.startswith('P3-')):,}"]
    for suffix in ('alone','C','G'):
        s=slots['P1-matrix_050-r1-'+suffix]['service']['foreground']
        extra.append(f"{s['response_seconds_completed_only']['p95']*1000:.3f}")
    extra.append(f'{min(diffs):.2f}–{max(diffs):.2f}')
    extra.extend(f"{max(r[field] for r in data['resources'])/2**30:.3f}" for field in ('component_footprint_bytes','all_owned_rss_bytes'))
    offsets, penalties = [], []
    for case in ('matrix_050', 'matrix_075'):
        for rnd in (0, 1):
            p95 = lambda e: slots[f'P1-{case}-r{rnd}-{e}']['service']['foreground']['response_seconds_completed_only']['p95']*1000
            offsets.append(p95('alone')-p95('C')); penalties.append(p95('G')-p95('alone'))
    extra += [f'{min(offsets):.3f}–{max(offsets):.3f}', f'{min(penalties):.1f}–{max(penalties):.1f}', projected]
    # Article detail uses the same checked records as the summary and figures.
    article = []
    rates = [p['C']['positions_per_second'] for p in data['medians'] if p['positions'] >= 256]
    article.append(f'{min(rates):,.0f}–{max(rates):,.0f}')
    for engine in 'CG':
        article.extend(f'{rate(s):.3f}' for s in sat[engine])
        temperatures = [s['tail_medians']['macmon_gpu_sensor_mean_c'] for s in sat[engine]]
        article.append(f'{min(temperatures):.1f}–{max(temperatures):.1f}')
    for engine in ('alone', 'C', 'G'):
        for rnd in (0, 1):
            service = slots[f'P1-memory_050-r{rnd}-{engine}']['service']['foreground']
            article.append(f'{100*service["deadline_violation_fraction"]:.2f}%')
    article.extend([f'{len(data["resources"]):,}', str(len(data['p2']))])
    for block in data['p0']:
        article.append(str(block['checkpoints'][-1]['measured_requests']))
    for name, slot in slots.items():
        if name.startswith('P1-') and 'inference' in slot['service']:
            article.append(str(slot['service']['inference']['completed']))
    gpu_ends = [r['last_footprint_bytes']/2**30 for r in data['native_growth'] if r['engine']=='G']
    article.append(f'{min(gpu_ends):.2f}–{max(gpu_ends):.2f}')
    article.extend(f'{r["last_footprint_bytes"]/2**20:.0f}' for r in native)
    article = list(dict.fromkeys(article))
    return {'README.md':common,'README.zh-CN.md':common,
            'findings/w4a16-service-tradeoffs/README.md':common+extra,
            'articles/04-w4a16-service-tradeoffs.md':common+extra+article,
            'articles/zh/04-W4A16留在ANE上值得吗.md':common+extra+article,
            'findings/iosurface-per-call-growth/README.md':[calls, shrink, projected],
            'findings/README.md':[calls]}

"""Classify frozen groups; raw blocks are retained irrespective of classification."""
import math
import statistics

FIELDS = ('macmon_cpu_sensor_mean_c', 'macmon_gpu_sensor_mean_c', 'fan0_rpm', 'fan1_rpm')


def summarize(block, config):
    start, end = block['window_start_ns'], block['window_end_ns']
    if end-start != config['window_seconds']*10**9:
        return {'valid': False, 'reason': 'wrong_window'}
    medians, missing = {}, []
    for field in FIELDS:
        unique = {}
        conflict = False
        for row in block.get('samples', []):
            t, value = row['sample_monotonic_ns'], row.get(field)
            if start <= t <= end and isinstance(value, (int, float)) and math.isfinite(value):
                if t in unique and unique[t] != value:
                    conflict = True
                unique[t] = value
        ticks = sorted(unique)
        valid = (not conflict and len(ticks) >= config['minimum_distinct_samples_per_field']
                 and ticks[-1]-ticks[0] >= config['minimum_sample_span_seconds']*10**9
                 and max(b-a for a,b in zip(ticks, ticks[1:])) <= config['maximum_gap_seconds']*10**9)
        if not valid:
            missing.append(field)
        else:
            medians[field] = statistics.median(unique.values())
    states = {}
    for row in block.get('thermal_samples', []):
        tick, state = row['sample_monotonic_ns'], row.get('thermal_state')
        if start <= tick <= end:
            if tick in states and states[tick] != state:
                missing.append('conflicting_thermal_state')
            states[tick] = state
    ticks = [start, *sorted(states), end]
    if (not states or any(v not in ('nominal','fair','serious','critical') for v in states.values())
            or max(b-a for a,b in zip(ticks,ticks[1:])) > config['thermal_state_maximum_gap_seconds']*10**9):
        missing.append('thermal_state_coverage')
    return {'valid': not missing, 'missing': missing, 'medians': medians,
            'thermal_levels': sorted({str(v) for v in states.values()})}


def classify(group, blocks, config):
    # group membership comes from the preregistered manifest; never search for a match.
    members = group['members']
    if not group['frozen_before_results'] or len(members) < 2 or len(set(members)) != len(members):
        raise ValueError('invalid frozen group')
    if any(member not in blocks for member in members):
        return {'group_id': group['id'], 'status': 'undetermined', 'reason': 'incomplete_group'}
    summary = {member: summarize(blocks[member], config) for member in members}
    if any(not value['valid'] for value in summary.values()):
        return {'group_id': group['id'], 'status': 'undetermined', 'blocks': summary}
    failures = []
    for field in FIELDS:
        values = [value['medians'][field] for value in summary.values()]
        tolerance = (config['temperature_group_max_minus_min_c'] if field.endswith('_c') else
                     max(config['fan_group_max_minus_min_rpm_floor'],
                         config['fan_group_relative_tolerance']*min(values)))
        if max(values)-min(values) > tolerance:
            failures.append(field)
    levels = [value['thermal_levels'] for value in summary.values()]
    if any(len(value) != 1 for value in levels) or any(value != levels[0] for value in levels):
        failures.append('thermal_state')
    return {'group_id': group['id'], 'status': 'mismatched' if failures else 'matched',
            'failures': failures, 'blocks': summary}

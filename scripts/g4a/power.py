"""Energy over the union of measured windows, as the G4 A runner integrated it."""
import math

from g3.power import endpoint


def merge(windows):
    result = []
    for start, end in sorted(windows):
        if not start < end:
            raise ValueError('phase_window_order')
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    if not result:
        raise ValueError('missing_phase_windows')
    return result


def integrate(rows, windows, lag=2_000_000_000):
    """Count each sample once; a sample's share is its overlap with any window."""
    windows = merge(windows)
    totals = {d: dict(lower_J=0., estimate_J=0., upper_J=0.) for d in ('cpu', 'gpu', 'ane', 'components')}
    failures = []; used = 0; support = []
    for row in rows:
        try:
            lo, hi, dt = endpoint(row, lag)
            if not any(hi > a and lo - dt < b for a, b in windows):
                continue
            d = row['decoded']
            v = {k: float(x['energy_mj']) / 1000 for k, x in d['domains'].items()}
            if d.get('issues') or set(v) != {'cpu', 'gpu', 'ane'} or not all(math.isfinite(x) and x >= 0 for x in v.values()):
                raise ValueError('invalid_energy_domains')
            v['components'] = sum(v.values()); middle = (lo + hi) / 2
            fraction = sum(max(0, min(b, middle) - max(a, middle - dt)) for a, b in windows) / dt
            definite = any(lo - dt >= a and hi <= b for a, b in windows)
            for key, value in v.items():
                totals[key]['lower_J'] += value if definite else 0
                totals[key]['estimate_J'] += value * fraction
                totals[key]['upper_J'] += value
            used += 1; support.append([lo - dt, hi])
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(str(exc))
    if used < 2:
        failures.append('insufficient_samples')
    coverage = merge(support) if support else []
    if any(not any(c <= a and d >= b for c, d in coverage) for a, b in windows):
        failures.append('phase_not_covered')
    return dict(domains=totals, windows=windows, seconds=sum(b - a for a, b in windows) / 1e9,
                samples=used, failures=sorted(set(failures)), timing_coverage_passed=not failures,
                lag_assumption_ns=lag)

import datetime
import plistlib
"""Software energy estimates with explicit sample-boundary sensitivity."""
import json
import math
from pathlib import Path
import statistics


def power_rows(capture):
    path=Path(capture)/'power/records.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def endpoint(row,extra_lag_ns=2_000_000_000):
    d=row['decoded'];dt=d['elapsed_ns'];low,high=d['source_timestamp_bin_monotonic_ns']
    receipt=row['receipt']['monotonic_after_ns']
    # Union of timestamp-at-start and timestamp-at-end conventions. The extra
    # lag is a sensitivity assumption checked against the admission transitions,
    # not a calibrated bound on Apple's hardware power estimator.
    lo=low-extra_lag_ns;hi=min(high+dt,receipt)
    if not 0<dt<=3e9 or lo>hi or receipt-high>5e9:
        raise ValueError('sample_clock_or_delay')
    return lo,hi,dt


def energy_window(rows,start,end,extra_lag_ns=2_000_000_000):
    if not start<end:raise ValueError('energy_window_order')
    totals={d:{'lower_J':0.,'estimate_J':0.,'upper_J':0.} for d in ('cpu','gpu','ane','components')}
    failures=[];used=0;receipts=[];covered=[];durations=[]
    for row in rows:
        try:
            lo,hi,dt=endpoint(row,extra_lag_ns)
            if hi<=start or lo-dt>=end:continue
            d=row['decoded']
            if d.get('issues'):raise ValueError('power_domains:'+','.join(d['issues']))
            values={domain:float(v['energy_mj'])/1000 for domain,v in d['domains'].items()}
            if set(values)!={'cpu','gpu','ane'} or not all(math.isfinite(v) and v>=0 for v in values.values()):
                raise ValueError('invalid_energy_domains')
            values['components']=sum(values.values())
            middle=(lo+hi)/2
            fraction=max(0,min(end,middle)-max(start,middle-dt))/dt
            definite=(lo-dt>=start and hi<=end)
            for domain,value in values.items():
                totals[domain]['lower_J']+=value if definite else 0
                totals[domain]['estimate_J']+=fraction*value
                totals[domain]['upper_J']+=value
            used+=1;receipts.append(row['receipt']['monotonic_after_ns']);covered.append((lo-dt,hi));durations.append(dt)
        except (KeyError,ValueError,TypeError) as exc:failures.append(str(exc))
    if used<2:failures.append('insufficient_samples')
    if any(b-a>3e9 or b<=a for a,b in zip(receipts,receipts[1:])):failures.append('receipt_gap_or_reorder')
    if len(receipts)>1:
        observed=receipts[-1]-receipts[0]+statistics.median(durations)
        if abs(sum(durations)/observed-1)>.02:failures.append('elapsed_coverage_mismatch')
    if not covered or min(a for a,b in covered)>start or max(b for a,b in covered)<end:
        failures.append('window_not_covered')
    return {'start_ns':start,'end_ns':end,'samples':used,'domains':totals,'failures':sorted(set(failures)),
        'integrity_passed':not failures,'extra_counter_lag_assumption_ns':extra_lag_ns,
        'interval_mean_power_assumption':True,'bounds_scope':'sample attribution under stated timing assumptions; not sensor accuracy',
        'wall_input_energy':False}


def response(rows,start,end,arm):
    idle=[];active=[];times=[];failures=[]
    for row in rows:
        try:
            lo,hi,dt=endpoint(row,0);middle=(lo+hi)/2
            if row['decoded'].get('issues'):raise ValueError('invalid_power_domain')
            power=row['decoded']['domains'][arm]['power_mw']/1000
            if start-25e9<middle<start-5e9:idle.append(power)
            if start+5e9<middle<end-3e9:active.append(power)
            times.append((middle,power))
        except (KeyError,ValueError,TypeError) as exc:failures.append(str(exc))
    if len(idle)<5 or len(active)<5:
        return {'passed':False,'reason':'insufficient_idle_or_active_samples','idle_samples':len(idle),'active_samples':len(active)}
    base=statistics.median(idle);load=statistics.median(active)
    noise=statistics.median(abs(x-base) for x in idle)
    threshold=base+max(.2,5*noise,.1*(load-base))
    rise=next((t for t,p in times if start<=t<=end and p>threshold),None)
    delay=(rise-start)/1e9 if rise is not None else None
    passed=not failures and load-base>max(.2,5*noise) and delay is not None and delay<=5
    return {'passed':passed,'idle_median_W':base,'active_median_W':load,'idle_MAD_W':noise,
        'response_threshold_W':threshold,'observed_rise_delay_seconds':delay,
        'hardware_estimator_latency_calibrated':False,'failures':failures}

def epoch_bin_to_monotonic(epoch_low, epoch_high, anchor):
    """Local mapping only; does not assume one wall-clock origin for an entire run."""
    return [epoch_low - anchor['epoch_ns'] + anchor['monotonic_before_ns'],
            epoch_high - anchor['epoch_ns'] + anchor['monotonic_after_ns']]


def decode_power(frame, anchor):
    obj = plistlib.loads(frame[:-1])
    issues = []
    dt = obj.get('elapsed_ns')
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('invalid_elapsed_ns')
    p = obj.get('processor', {})
    domains = {}
    for domain in ('cpu', 'gpu', 'ane'):
        power, energy = p.get(domain + '_power'), p.get(domain + '_energy')
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in (power, energy))
        if not valid:
            issues.append('missing_or_invalid_domain:' + domain)
        domains[domain] = dict(power_mw=power if valid else None, energy_mj=energy if valid else None)
        # Integer energy values have at least 1 mJ quantization uncertainty.
        if valid and abs(power * dt / 1e9 - energy) > max(2, abs(energy) * .001):
            issues.append('power_energy_mismatch:' + domain)
    combined = p.get('combined_power')
    if not issues and (not isinstance(combined, (int, float)) or not math.isfinite(combined)
                       or abs(sum(d['power_mw'] for d in domains.values()) - combined) >= 10):
        issues.append('power_domains_do_not_sum')
    if obj.get('invalid') or p.get('invalid'):
        issues.append('source_invalid')
    stamp = obj.get('timestamp')
    if not isinstance(stamp, datetime.datetime):
        raise ValueError('missing_source_timestamp')
    stamp_ns = round(stamp.replace(tzinfo=datetime.timezone.utc).timestamp() * 1e9)
    bounds = epoch_bin_to_monotonic(stamp_ns, stamp_ns + 10**9, anchor)
    # The raw plist timestamp is quantized. No cumulative elapsed/origin fit.
    return dict(elapsed_ns=dt, domains=domains, issues=issues,
                source_timestamp=stamp.isoformat(), thermal=obj.get('thermal_pressure'),
                source_timestamp_bin_monotonic_ns=bounds,
                time_status='local_wall_mapping_only_sample_endpoint_unadmitted')

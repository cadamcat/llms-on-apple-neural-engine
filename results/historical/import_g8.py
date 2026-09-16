"""Import the closed G8 four-bit representation run and its clock-mapping follow-up into a portable bundle; never execute a device."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('import_g4a', HERE / 'import_g4a.py')
g4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4a)

PLAN = Path('workplans/g8-four-bit-representations-20260915')
PHASES = {'original': (Path('results/g8-night-20260916-011943'), PLAN),
          'follow-up': (Path('results/g8-clock-recovery-20260916-081936'), Path('workplans/g8-clock-recovery-20260916'))}
FINAL = Path('workplans/g8-analysis-20260916/final')
G7_PLAN = Path('workplans/g7-coreml-a8w4-20260914')
ANE_OK = 'ANEProgramProcessRequestDirect() status=0x0'
CONTROLS = ('original', 'zero', 'negative', 'repeat', 'benchmark', 'benchmark_repeat')
TOLERANCE_NS = 100_000
FAILURE_WORDS = ('Falling back', 'ANE compilation failed', 'Failed with status')


class Writer(g4a.Writer):
    def provenance(self, transformations):
        self.dump('provenance.json', {'importer': 'results/historical/import_g8.py', 'sources': sorted(self.reader.sources),
                                      'transformations': transformations})


def portable(reader, path):
    """Workspace files by workspace-relative path, this repository's scripts by repository-relative path."""
    path = Path(path)
    try:
        return path.relative_to(reader.workspace).as_posix()
    except ValueError:
        return 'ane-scope/' + path.relative_to(reader.workspace.parent / 'ane-scope').as_posix()


def bridge_offsets(samples):
    """Mach continuous minus absolute ticks, bracketed by absolute readings taken before and after each continuous reading."""
    low = max(s['continuous'] - s['absolute_after'] for s in samples)
    high = min(s['continuous'] - s['absolute_before'] for s in samples)
    if low > high:
        raise ValueError('mach_bridge')
    return low, high


def ane_requests(reader, folder, bridge):
    """Count successful direct ANE requests inside each control call from the host-PID log, under both clock mappings.

    Wall: the log's microsecond wall time moved onto the host clock through the epoch/monotonic pair taken at host start.
    Mach: the log's continuous ticks moved onto host absolute time through a continuous/absolute bridge; the interval
    must lie inside the call window with the same tolerance."""
    controls = reader.json(folder / 'host/controls.json')
    anchor = reader.json(folder / 'CLOCK.json')
    pid = reader.json(folder / 'host/ready.json')['pid']
    if [c['phase'] for c in controls] != list(CONTROLS):
        raise ValueError('control_phases:' + folder.name)
    low, high = bridge_offsets(anchor['samples']) if 'samples' in anchor else bridge
    if 'after_controls' in anchor:
        after = bridge_offsets(anchor['after_controls']['samples'])
        low, high = max(low, after[0]), min(high, after[1])
        if low > high:
            raise ValueError('mach_bridge_discontinuity:' + folder.name)
    wall, mach, failures = [0] * 6, [0] * 6, 0
    for line in reader.raw(folder / 'unified.ndjson').decode(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if row.get('processID') != pid:
            continue
        message = row.get('eventMessage', '')
        failures += any(k in message for k in FAILURE_WORDS)
        if ANE_OK not in message:
            continue
        epoch = int(datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S.%f%z').timestamp() * 1e9)
        ns = epoch - anchor['epoch_ns'] + anchor['monotonic_ns']
        lower, upper = (row['machTimestamp'] - high) * 125 // 3, -(-(row['machTimestamp'] - low) * 125 // 3)
        for i, c in enumerate(controls):
            if c['start_ns'] - TOLERANCE_NS <= ns <= c['end_ns'] + TOLERANCE_NS:
                wall[i] += 1
            if c['start_ns'] - TOLERANCE_NS <= lower and upper <= c['end_ns'] + TOLERANCE_NS:
                mach[i] += 1
    return {'wall': wall, 'mach': mach}, failures, controls


def plan_devices(plan):
    """Core ML compute plan reduced to operator, output and preferred device class; Core AI has no per-operator mapping."""
    if isinstance(plan, dict):
        return plan
    return [{'operator': p['operator'], 'output': p['outputs'][0], 'preferred': p['preferred'].split(':')[0].lstrip('<')}
            for p in plan if p['operator'] != 'const']


def admission(reader, folder, recorded, bridge):
    """Recorded numeric and placement result, the control-output hashes, and both recounts of ANE requests."""
    counts, failures, controls = ane_requests(reader, folder, bridge)
    placement = recorded['placement']
    mapping = 'mach' if placement.get('clock_mapping') == 'bracketed_mach_continuous_to_absolute' else 'wall'
    if counts[mapping] != placement['requests_per_control'] or bool(failures) != bool(placement['failures']):
        raise ValueError('ane_request_recount:' + folder.name)
    return {'passed': recorded['passed'], 'numeric': recorded['numeric'],
            'placement': {k: placement[k] for k in ('passed', 'reasons', 'requests_per_control', 'failures')} |
                         {'clock_mapping': mapping, 'compute_plan': plan_devices(placement['compute_plan'])},
            'requests_per_control_by_mapping': counts,
            'control_output_sha256': {c['phase']: c['sha256'] for c in controls}}


def asset_record(reader, plan, name):
    asset = reader.json(plan / 'assets' / name / 'ASSET.json')
    audit = asset['audit']
    if not audit['passed']:
        raise ValueError('asset_audit:' + name)
    # Codes and decoded-weight hashes are kept: the verifier compares them across representations and with the source codes.
    weights = [w if isinstance(w, str) else {k: w[k] for k in ('name', 'codes_sha256', 'decoded_fp16_sha256')} for w in audit['weights']]
    return {k: asset.get(k) for k in ('runtime', 'workload', 'positions', 'representation', 'format', 'shape', 'output_shape', 'entrypoint')} | \
           {'audit': {'counts': audit['counts'], 'weights': weights}}


def check_launch(reader, run, plan):
    state = reader.json(run / 'RUN.json')
    launch = reader.json(plan / 'LAUNCH.json')
    if state['status'] != 'closed' or state['unrun']:
        raise ValueError('run_not_closed:' + run.name)
    if hashlib.sha256(reader.raw(plan / 'LAUNCH.json')).hexdigest() != state['launch_sha256']:
        raise ValueError('launch_record_changed:' + run.name)
    for path, digest in launch['files'].items():
        # Workspace files must still be the launch copies; this repository's scripts keep evolving under Git.
        if not portable(reader, path).startswith('ane-scope/') and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('launch_identity:' + portable(reader, path))
    return state, launch


def extract(workspace, output):
    reader = g4a.Reader(workspace)
    writer = Writer(reader, output)
    diagnosis = reader.json(PHASES['follow-up'][1] / 'DIAGNOSIS.json')
    # The follow-up's retrospective bridge, read during the original run; the original hosts recorded no bridge of their own.
    bridge = bridge_offsets([{'absolute_before': b['a_before'], 'continuous': b['continuous'], 'absolute_after': b['a_after']}
                             for b in diagnosis['bridges']])

    states, launches = {}, {}
    for phase, (run, plan) in PHASES.items():
        states[phase], launches[phase] = check_launch(reader, run, plan)
    original, follow = states['original'], states['follow-up']
    frozen = {j['id']: j for j in launches['original']['jobs']}
    rejected = sorted(k for k, v in original['jobs'].items() if v['status'] == 'failed')
    skipped = sorted(k for k, v in original['jobs'].items() if v['status'] == 'skipped_after_candidate_failure')
    completed = sorted(k for k, v in original['jobs'].items() if v['status'] == 'completed')
    if len(completed) + len(rejected) + len(skipped) != len(frozen) or set(original['jobs']) != set(frozen):
        raise ValueError('original_inventory')
    if sorted(original['cancelled_assets']) != sorted({frozen[k]['asset'] for k in rejected}):
        raise ValueError('original_cancellations')
    if sorted(j['id'] for j in launches['follow-up']['jobs']) != sorted(rejected + skipped) or \
            any(j != frozen[j['id']] for j in launches['follow-up']['jobs']) or follow['cancelled_assets'] or \
            [j['id'] for j in launches['follow-up']['jobs']] != [j['id'] for j in launches['original']['jobs'] if j['id'] in rejected + skipped]:
        raise ValueError('follow_up_selection')
    selection = reader.json(PHASES['follow-up'][1] / 'SELECTION.json')
    if sorted(selection['retained_original_jobs']) != completed:
        raise ValueError('follow_up_retained')

    # Every prepared configuration with its asset audit and admission.
    preparation = reader.json(PLAN / 'PREPARATION.json')
    if preparation['status'] != 'closed':
        raise ValueError('preparation_open')
    configs = {}
    for name, cell in preparation['cells'].items():
        folder = PLAN / 'admission' / Path(cell['admission_path']).name
        recorded = reader.json(folder / 'ADMISSION.json')
        if recorded['passed'] != cell['passed'] or recorded['numeric'] != cell['numeric']:
            raise ValueError('preparation_record:' + name)
        configs[name] = {'asset': asset_record(reader, PLAN, name), 'admission': admission(reader, folder, recorded, bridge)}

    # Timed blocks from both phases, and the original rejections.
    blocks, captures, rejections = [], {}, []
    for phase, (run, plan) in PHASES.items():
        state = states[phase]
        for job in launches[phase]['jobs']:
            if phase == 'original' and job['id'] in skipped:
                continue
            entry = state['jobs'][job['id']]
            folder = run / 'jobs' / job['id']
            result = reader.json(folder / 'ADMISSION.json')
            if phase == 'original' and job['id'] in rejected:
                if result['passed'] or not result['numeric']['passed'] or result['placement']['reasons'] != ['missing_ane_control_request'] \
                        or result['host_returncode'] != 0 or not result['closed'] or entry['result'] != result:
                    raise ValueError('rejection_record:' + job['id'])
                rejections.append({'id': job['id'], 'config': job['asset'], 'round': job['round'], 'seed': job['seed'],
                                   'admission': admission(reader, folder, result, bridge)})
                continue
            measurement = reader.json(folder / 'MEASUREMENT.json')
            block = reader.json(folder / 'host/BLOCK.json')
            if (entry['status'] != 'completed' or not result['passed'] or result['host_returncode'] != 0 or not result['closed']
                    or not result['final_output_matches_control'] or result['block'] != block or entry['measurement'] != measurement):
                raise ValueError('job_not_completed:' + job['id'])
            if not configs[job['asset']]['admission']['passed']:
                raise ValueError('job_without_admission:' + job['id'])
            clocks = list(struct.iter_unpack('<QQ', reader.raw(folder / 'host/clocks.u64le')))
            last = block['start_ns']
            for a, z in clocks:
                if not last <= a < z <= block['end_ns']:
                    raise ValueError('clock_order:' + job['id'])
                last = z
            durations = sorted(z - a for a, z in clocks)
            p50 = statistics.median(durations) / 1e6
            if len(clocks) != block['calls'] or p50 != measurement['API_p50_ms']:
                raise ValueError('clock_summary:' + job['id'])
            capture = phase + '/' + Path(entry['capture']).name
            captures.setdefault(capture, job['group'])
            blocks.append({'id': job['id'], 'phase': phase, 'config': job['asset'], 'round': job['round'], 'seed': job['seed'],
                           'group': job['group'], 'seconds_requested': job['seconds'], 'capture': capture,
                           'admission': admission(reader, folder, result, bridge),
                           'block': g4a.without_hashes(block),
                           'api_latency_ms': {'p50': p50, 'p90': durations[int(0.9 * (len(durations) - 1))] / 1e6, 'max': durations[-1] / 1e6,
                                              'busy_fraction': sum(durations) / (block['end_ns'] - block['start_ns'])},
                           'recorded_measurement': measurement})
    audit = reader.json(FINAL / 'AUDIT.json')
    audit_phase = {'original': 'original', 'recovery': 'follow-up'}
    if sorted(b['id'] + '@' + audit_phase[b['phase']] for b in audit['blocks']) != sorted(b['id'] + '@' + b['phase'] for b in blocks):
        raise ValueError('final_audit_inventory')
    writer.dump('configs.json', configs)
    writer.dump('blocks.json', blocks)
    writer.dump('rejections.json', {'rejected_first_round': rejections, 'cancelled_later_rounds': skipped,
                                    'retrospective_bridge_offset_ticks': list(bridge)})

    idle = {}
    for phase, (run, _) in PHASES.items():
        recorded = audit['phases']['original' if phase == 'original' else 'recovery']['idle_mean_W']
        for name in ('idle-start', 'idle-end'):
            record = reader.json(run / 'captures' / name / 'IDLE.json')
            if record != states[phase][name.replace('-', '_')] or record['status'] != 'completed':
                raise ValueError('idle_record:' + phase + '/' + name)
            idle[phase + '/' + name] = {'window_ns': record['window_ns'], 'recorded_mean_W': recorded[name]}
            captures[phase + '/' + name] = name

    status, last = {}, {}

    def frames():
        for key in captures:
            phase, name = key.split('/')
            for frame in g4a.power_frames(reader, PHASES[phase][0] / 'captures' / name / 'capture/power'):
                last[key] = max(last.get(key, 0), frame['receipt']['monotonic_after_ns'])
                yield dict(frame, capture=key)
    writer.compressed('power.jsonl.gz', frames())
    for key in captures:
        phase, name = key.split('/')
        folder = PHASES[phase][0] / 'captures' / name
        audit_record = reader.json(folder / 'audit.json')
        finish = states[phase][name.replace('-', '_')]['capture'] if name.startswith('idle-') else reader.json(folder / 'FINISH.json')
        if finish['audit'] != audit_record or finish['returncode'] != 0 or \
                not all(audit_record[k] for k in ('integrity_passed', 'capture_completed', 'cleanup_passed')):
            raise ValueError('capture_integrity:' + key)
        public_audit = dict(audit_record, source_close={k: v for k, v in audit_record['source_close'].items() if k != 'sampler_pid'})
        status[key] = dict(public_audit, returncode=finish['returncode'], group=captures[key], last_power_receipt_ns=last[key])
    writer.dump('capture-status.json', status)
    writer.dump('idle.json', idle)

    closures = {}
    for phase, (run, _) in PHASES.items():
        closure = reader.json(FINAL / ('original-closure.json' if phase == 'original' else 'recovery-closure.json'))
        if not closure['passed'] or closure['still_present'] or closure['loggers_still_present']:
            raise ValueError('process_closure:' + phase)
        closures[phase] = {'run': {k: states[phase][k] for k in ('status', 'cancelled_assets', 'unrun', 'started_at', 'finished_at')},
                           'processes': {k: closure[k] for k in ('recorded_pids_checked', 'native_hosts_checked', 'ownership_receipts_checked',
                                                                 'still_present', 'samplers_reaped')}}
    writer.dump('closures.json', closures)

    matrix = reader.json(PLAN / 'MATRIX.json')
    inputs = reader.json(PLAN / 'INPUTS.json')
    dependencies = reader.json(PLAN / 'DEPENDENCIES.json')
    g7_matrix = reader.json(G7_PLAN / 'MATRIX.json')
    g7_inputs = reader.json(G7_PLAN / 'INPUTS.json')
    references = {n: reader.json(PLAN / 'inputs' / n / 'REFERENCE.json') for n in ('n64', 'n1024')}
    for n, r in references.items():
        if r['source_codes'] != inputs['sets'][n]['source_codes']:
            raise ValueError('reference_codes:' + n)
    writer.dump('protocol.json', {
        'historical_import': True, 'date': '2026-09-16', 'machine': {'chip': 'Apple M5 Pro', 'memory_GiB': 48},
        'model': 'google/gemma-4-E4B-it-qat-mobile-ct', 'revision': g7_matrix['source_revision'],
        'weights': {k: {x: v[x] for x in ('shape', 'parameters', 'scale_shape', 'scale_dtype')} for k, v in g7_inputs['weights'].items()},
        'software': dependencies['versions'] | {'macOS': '27.0 (26A428)', 'xcode': '27.0'},
        'representations': {
            'fp16': 'Core ML: the four-bit codes times the FP16 per-output-channel scale, stored as FP16 constants',
            'int8_lut': 'Core ML: four-bit indices into a 16-entry palette built from INT8 −8…7 by constexpr_blockwise_shift_scale with the '
                        'per-output-channel scale, then constexpr_lut_to_dense; the G7 W4A16 form',
            'int4': 'Core ML: the codes stored as signed INT4 with the per-output-channel scale, constexpr_blockwise_shift_scale only',
            'fp16_lut': 'Core ML: four-bit indices into a per-output-channel FP16 palette with the scale already applied, constexpr_lut_to_dense only',
            'coreai_int8_lut': 'Core AI: the G7 W4A16 export, coreai.lut_to_dense and coreai.blockwise_shift_scale'},
        'activations': matrix['activations'],
        'workloads': {'gate': 'gate_proj as a 1×1 convolution, 2560 → 10240',
                      'mlp': 'down(gelu_tanh(gate(x)) * up(x)), no attention, residual or normalization'},
        'references': {n: r for n, r in references.items()}, 'numeric_relative_L2_limit': matrix['numeric_relative_L2_limit'],
        'ordinary_rows': 'rows 6 onward; rows 0-5 are the boundary controls',
        'rounds': matrix['rounds'], 'benchmark_seeds': matrix['benchmark_seeds'],
        'block_seconds': matrix['seconds_per_block'], 'warm_seconds': 5, 'minimum_warm_calls': 10, 'quiet_seconds': 25,
        'order': 'per round sorted by workload, positions, representation and runtime; the middle round reversed; one new host per block',
        'phases': {'original': 'all 20 configurations; the first round of three 1024-position Core ML MLP configurations was rejected by the '
                               'placement clock check before timing, and their later rounds were cancelled',
                   'follow-up': 'only those nine untimed blocks, same assets, inputs, host, timing and energy rules, with a Mach-clock '
                                'placement check; no completed block repeated'},
        'placement_clock': {'tolerance_ns': TOLERANCE_NS,
                            'wall': 'original phase: log wall time with microsecond resolution mapped through an epoch/monotonic pair taken at host start',
                            'mach': 'follow-up phase: log Mach continuous ticks mapped to host absolute time through continuous/absolute '
                                    'bridges read before and after the controls; the mapped interval must lie inside the window'},
        'energy': {'lags_ns': [0, 10 ** 9, 2 * 10 ** 9, 5 * 10 ** 9], 'primary_lag': '2', 'interior_margin_seconds': 15,
                   'response_arm': 'ane', 'power_domains': ['cpu', 'gpu', 'ane'], 'idle_baseline_subtracted': False},
        'host': {'source': f'{G7_PLAN}/host.swift', 'coreml_compute_units': 'cpuAndNeuralEngine', 'coreai_preferred_compute_unit': 'neuralEngine',
                 'timing': 'synchronous MLModel.prediction or awaited function.run per call; API start and end clocks'},
        'scope': 'Real E4B weights with seeded RMS-normalized synthetic activations. Per-block wall-clock speed and software component '
                 'energy; no model quality, attention, KV cache, A8 or exclusive per-operation placement.'})
    writer.provenance([
        'Strip the workspace prefix; refuse any other personal path. Check both phases\' launch records and every frozen workspace file.',
        'Keep every prepared configuration: asset audit reduced to operator counts and, per projection, the codes and decoded-FP16 hashes '
        '(compared across representations and with the reference codes); recorded numeric comparisons, placement result and control-output '
        'hashes, which stand in for the unbundled outputs; Core ML compute plans reduced to operator, first output and preferred device class.',
        'Recount successful direct ANE requests per control call from each host-PID log stream under the wall-clock and the Mach-clock mapping; '
        'refuse a mismatch between the recount under the phase\'s own mapping and the recorded count. Original hosts use the retrospective '
        'bridge read during the original run. The logs are not bundled.',
        'Keep the three rejected first-round admissions and the six cancelled job IDs; keep each timed block record and recorded measurement; '
        'reduce per-call API clocks to p50, p90, maximum and busy fraction after checking their order, count and recorded p50; the clock files are not bundled.',
        'Re-serialize CPU/GPU/ANE power fields from each plist frame of the nineteen captures, as the G4 A importer does, tagged by phase and capture.',
        'Keep capture audits without sampler PIDs, idle windows with the recorded idle means, both run closures and the process-closure counts; drop process IDs, argv and output tensors.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

"""Import the closed G7 Core ML / Core AI same-code run into a portable bundle; never execute a device."""
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

RUN = Path('results/g7-night-20260915-011532')
PLAN = Path('workplans/g7-coreml-a8w4-20260914')
ANE_OK = 'ANEProgramProcessRequestDirect() status=0x0'
PHASES = ('original', 'zero', 'negative', 'repeat', 'benchmark', 'benchmark_repeat')


class Writer(g4a.Writer):
    def provenance(self, transformations):
        self.dump('provenance.json', {'importer': 'results/historical/import_g7.py', 'sources': sorted(self.reader.sources),
                                      'transformations': transformations})


def portable(reader, path):
    """Workspace files by workspace-relative path, this repository's scripts by repository-relative path."""
    path = Path(path)
    try:
        return path.relative_to(reader.workspace).as_posix()
    except ValueError:
        return 'ane-scope/' + path.relative_to(reader.workspace.parent / 'ane-scope').as_posix()


def ane_requests(reader, folder):
    """Recount successful direct ANE requests inside each control call from the host-PID log stream."""
    controls = reader.json(folder / 'host/controls.json')
    anchor = reader.json(folder / 'CLOCK.json')
    pid = reader.json(folder / 'host/ready.json')['pid']
    counts, failures = [0] * len(controls), 0
    for line in reader.raw(folder / 'unified.ndjson').decode(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        message = row.get('eventMessage', '')
        if row.get('processID') != pid:
            continue
        failures += any(k in message for k in ('Falling back', 'ANE compilation failed', 'Failed with status'))
        if ANE_OK not in message:
            continue
        epoch = int(datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S.%f%z').timestamp() * 1e9)
        ns = epoch - anchor['epoch_ns'] + anchor['monotonic_ns']
        for i, c in enumerate(controls):
            if c['start_ns'] - 100_000 <= ns <= c['end_ns'] + 100_000:
                counts[i] += 1
    return counts, failures, controls


def plan_devices(plan):
    """Core ML compute plan reduced to operator, output and preferred device class; Core AI has no per-operator mapping."""
    if isinstance(plan, dict):
        return plan
    return [{'operator': p['operator'], 'output': p['outputs'][0], 'preferred': p['preferred'].split(':')[0].lstrip('<')}
            for p in plan if p['operator'] != 'const']


def admission(reader, folder, recorded):
    """Recorded numeric and placement result, with the control-output hashes and a recount of ANE requests."""
    counts, failures, controls = ane_requests(reader, folder)
    placement = recorded['placement']
    if counts != placement['requests_per_control'] or bool(failures) != bool(placement['failures']):
        raise ValueError('ane_request_recount:' + folder.name)
    if [c['phase'] for c in controls] != list(PHASES):
        raise ValueError('control_phases:' + folder.name)
    return {'passed': recorded['passed'], 'numeric': recorded['numeric'],
            'placement': {k: placement[k] for k in ('passed', 'reasons', 'requests_per_control', 'failures')} |
                         {'compute_plan': plan_devices(placement['compute_plan'])},
            'control_output_sha256': {c['phase']: c['sha256'] for c in controls}}


def extract(workspace, output):
    reader = g4a.Reader(workspace)
    writer = Writer(reader, output)
    state = reader.json(RUN / 'RUN.json')
    launch = reader.json(PLAN / 'LAUNCH.json')
    if state['status'] != 'closed' or state['cancelled_assets'] or state['unrun']:
        raise ValueError('run_not_closed_cleanly')
    if hashlib.sha256(reader.raw(PLAN / 'LAUNCH.json')).hexdigest() != state['launch_sha256']:
        raise ValueError('launch_record_changed')
    frozen = {portable(reader, path): digest for path, digest in launch['files'].items()}
    for path, digest in launch['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('launch_identity:' + portable(reader, path))

    # Every prepared configuration, admitted or not, with its asset audit.
    preparation = reader.json(PLAN / 'PREPARATION.json')
    synthetic = reader.json(PLAN / 'SYNTHETIC-PREPARATION.json')
    clamp = reader.json(PLAN / 'CLAMP-PREPARATION.json')
    configs = {}
    entries = [(name, cell, Path(cell['admission_path']).name) for name, cell in preparation['cells'].items()]
    entries += [(name, cell, name) for name, cell in list(synthetic.items()) + list(clamp.items())]
    for name, cell, folder_name in entries:
        asset_folder = PLAN / 'assets' / name
        asset = reader.json(asset_folder / 'ASSET.json')
        folder = PLAN / 'admission' / folder_name
        recorded = reader.json(folder / 'ADMISSION.json')
        recheck = folder / 'PLACEMENT-RECHECK.json'
        if (reader.workspace / recheck).exists():
            # The first check missed the ios18.conv prefix; the offline recheck of the same log is the admission of record.
            recorded = dict(recorded, placement=reader.json(recheck), passed=recorded['numeric']['passed'] and reader.json(recheck)['passed'])
        if recorded['passed'] != cell['passed'] or recorded['numeric'] != cell['numeric']:
            raise ValueError('preparation_record:' + name)
        configs[name] = {'asset': {k: asset.get(k) for k in ('runtime', 'workload', 'positions', 'format', 'clip', 'depth', 'shape',
                                                            'output_shape', 'entrypoint')} |
                                  {'audit': g4a.without_hashes(asset['audit'])},
                         'placement_recheck': (reader.workspace / recheck).exists(),
                         'admission': admission(reader, folder, recorded)}

    # Formal blocks.
    jobs, captures = [], {}
    for job in launch['jobs']:
        entry = state['jobs'][job['id']]
        folder = RUN / 'jobs' / job['id']
        result = reader.json(folder / 'ADMISSION.json')
        measurement = reader.json(folder / 'MEASUREMENT.json')
        block = reader.json(folder / 'host/BLOCK.json')
        if (entry['status'] != 'completed' or not result['passed'] or result['host_returncode'] != 0 or not result['closed']
                or not result['final_output_matches_control'] or result['block'] != block or entry['measurement'] != measurement):
            raise ValueError('job_not_completed:' + job['id'])
        if not configs[job['asset']]['admission']['passed']:
            raise ValueError('job_without_admission:' + job['id'])
        payload = reader.raw(folder / 'host/clocks.u64le')
        clocks = list(struct.iter_unpack('<QQ', payload))
        last = block['start_ns']
        for a, z in clocks:
            if not last <= a < z <= block['end_ns']:
                raise ValueError('clock_order:' + job['id'])
            last = z
        durations = sorted(z - a for a, z in clocks)
        p50 = statistics.median(durations) / 1e6
        if len(clocks) != block['calls'] or p50 != measurement['API_p50_ms']:
            raise ValueError('clock_summary:' + job['id'])
        capture = Path(entry['capture']).name
        captures.setdefault(capture, job['group'])
        jobs.append({'id': job['id'], 'config': job['asset'], 'round': job['round'], 'seed': job['seed'], 'group': job['group'],
                     'seconds_requested': job['seconds'], 'capture': capture,
                     'admission': admission(reader, folder, result),
                     'block': g4a.without_hashes(block), 'api_latency_ms': {'p50': p50, 'p90': durations[int(0.9 * (len(durations) - 1))] / 1e6,
                                                        'max': durations[-1] / 1e6, 'busy_fraction': sum(durations) / (block['end_ns'] - block['start_ns'])},
                     'recorded_measurement': measurement})
    writer.dump('configs.json', configs)
    writer.dump('blocks.json', jobs)

    idle = {}
    for name in ('idle-start', 'idle-end'):
        record = reader.json(RUN / 'captures' / name / 'IDLE.json')
        if record != state[name.replace('-', '_')] or record['status'] != 'completed':
            raise ValueError('idle_record:' + name)
        idle[name] = {'window_ns': record['window_ns']}
        captures[name] = name

    status, last = {}, {}

    def frames():
        for name in captures:
            for frame in g4a.power_frames(reader, RUN / 'captures' / name / 'capture/power'):
                last[name] = max(last.get(name, 0), frame['receipt']['monotonic_after_ns'])
                yield dict(frame, capture=name)
    writer.compressed('power.jsonl.gz', frames())
    for name in captures:
        audit = reader.json(RUN / 'captures' / name / 'audit.json')
        if name.startswith('idle-'):
            finish = state[name.replace('-', '_')]['capture']
        else:
            finish = reader.json(RUN / 'captures' / name / 'FINISH.json')
        if finish['audit'] != audit or finish['returncode'] != 0 or not all(audit[k] for k in ('integrity_passed', 'capture_completed', 'cleanup_passed')):
            raise ValueError('capture_integrity:' + name)
        status[name] = dict(audit, returncode=finish['returncode'], group=captures[name], last_power_receipt_ns=last[name])
    writer.dump('capture-status.json', status)
    idle_recorded = reader.json('workplans/g7-analysis-20260915/final/IDLE.json')
    writer.dump('idle.json', {name: dict(v, recorded=idle_recorded[name]) for name, v in idle.items()})

    closure = reader.json('workplans/g7-analysis-20260915/final/PROCESS-CLOSURE.json')
    writer.dump('closures.json', {'run': {k: state[k] for k in ('status', 'cancelled_assets', 'unrun')} |
                                         {'started_at': state['started_at'], 'finished_at': state['finished_at']},
                                  'processes': {k: closure[k] for k in ('recorded_pids_checked', 'ownership_receipts_checked', 'still_present', 'samplers_reaped')}})

    matrix = reader.json(PLAN / 'MATRIX.json')
    inputs = reader.json(PLAN / 'INPUTS.json')
    references = {n: reader.json(PLAN / 'inputs' / n / 'REFERENCE.json') for n in ('n64', 'n1024')}
    build = reader.json(PLAN / 'bin/BUILD.json')
    if build['source_sha256'] != frozen[f'{PLAN}/host.swift']:
        raise ValueError('host_build_source')
    writer.dump('protocol.json', {
        'historical_import': True, 'date': '2026-09-15', 'machine': {'chip': 'Apple M5 Pro', 'memory_GiB': 48},
        'model': 'google/gemma-4-E4B-it-qat-mobile-ct', 'revision': matrix['source_revision'],
        'weights': {k: {x: v[x] for x in ('shape', 'parameters', 'scale_shape', 'scale_dtype')}
                    for k, v in inputs['weights'].items()},
        'software': inputs['installed_metadata_only'] | {'macOS': '27.0 (26A428)', 'xcode': '27.0'},
        'formats': {'fp16_from_w4_codes': 'the same four-bit codes decoded to FP16 constants; FP16 activations',
                    'w4a16': 'four-bit indices, INT8 palette, per-output-channel FP16 scale; FP16 activations',
                    'w8a8_same_codes': 'the same codes stored as INT8 with the same scale; static per-tensor A8 QDQ at each projection',
                    'a8w4_int8_lut': 'the W4A16 weight representation with the W8A8 activation QDQ'},
        'workloads': {'gate': 'gate_proj as a 1×1 convolution, 2560 → 10240',
                      'mlp': 'down(gelu_tanh(gate(x)) * up(x)), no attention, residual or normalization',
                      'synthetic': 'the K512 128-layer 1×1 convolution chain over 64×64 positions, Core ML only'},
        'references': {n: {k: v for k, v in r.items() if k != 'source_codes'} for n, r in references.items()}, 'numeric_relative_L2_limits': matrix['numeric_relative_L2_limits'],
        'ordinary_rows': 'rows 6 onward; rows 0-5 are the boundary controls',
        'rounds': matrix['independent_rounds'], 'benchmark_seeds': matrix['benchmark_seeds'],
        'block_seconds': {'real': 120, 'synthetic': 60}, 'warm_seconds': 5, 'minimum_warm_calls': 10, 'quiet_seconds': 25,
        'order': 'per round sorted by workload, positions, format and runtime; odd rounds reversed; one new host per block',
        'energy': {'lags_ns': [0, 10 ** 9, 2 * 10 ** 9, 5 * 10 ** 9], 'primary_lag': '2', 'interior_margin_seconds': 15,
                   'response_arm': 'ane', 'power_domains': ['cpu', 'gpu', 'ane'], 'idle_baseline_subtracted': False},
        'host': {'source': f'{PLAN}/host.swift',
                 'coreml_compute_units': 'cpuAndNeuralEngine', 'coreai_preferred_compute_unit': 'neuralEngine',
                 'timing': 'synchronous MLModel.prediction or awaited function.run per call; API start and end clocks'},
        'scope': 'Real E4B weights with seeded RMS-normalized synthetic activations. Per-block wall-clock speed and software '
                 'component energy; no model quality, attention, KV cache or exclusive per-operation placement.'})
    writer.provenance([
        'Strip the workspace prefix; this repository\'s scripts frozen at launch are named ane-scope/<path>; refuse any other personal path.',
        'Keep every prepared configuration: asset audit, recorded numeric comparisons, placement result and control-output hashes, which stand in for the unbundled outputs; '
        'Core ML compute plans reduced to operator, first output and preferred device class. For the one configuration with an offline placement recheck, keep the recheck.',
        'Recount successful direct ANE requests per control call from each host-PID log stream and refuse a mismatch with the recorded count; the logs are not bundled.',
        'Keep each block record and recorded measurement; reduce per-call API clocks to p50, p90, maximum and busy fraction after checking their order, '
        'count and recorded p50; the clock files are not bundled.',
        'Re-serialize CPU/GPU/ANE power fields from each plist frame of the seventeen captures, as the G4 A importer does, tagged by capture.',
        'Keep capture audits, idle windows, the run closure and the process-closure counts; drop process IDs, argv and output tensors.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

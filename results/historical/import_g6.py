"""Import the closed G6 night run into a portable bundle; never execute a device."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('import_g4a', HERE / 'import_g4a.py')
g4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4a)

RUN = 'results/g6-night-20260914-002535'
PLAN = 'workplans/g6-night-20260913'
G4A_PLAN = 'workplans/g4-ready-20260913'
G4_TIERS = 'workplans/g4-cplus256-20260912'
HOST_SWIFT = 'swift/Sources/Tools/g3-flow-host/G3Host.swift'
ENGINE_SWIFT = 'swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAIStaticShapeEngine.swift'
ADMISSIONS = ('q4-c1280-reading_1024', 'q4-c768-quality_short', 'q4-c2304-quality_short-switch',
              'q4-c4352-quality_short-switch', 'q4-c8448-quality_short-switch', 'q4-c16640-quality_short-switch',
              'w4-c1280-reading_1024-switch', 'w4-c4352-quality_short-switch')
ANE_OK = 'ANEProgramProcessRequestDirect() status=0x0'
METAL = 'Metal Compiling Shader'


class Writer(g4a.Writer):
    def provenance(self, transformations):
        products = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.output.iterdir())}
        self.dump('provenance.json', {'importer': 'results/historical/import_g6.py', 'sources': self.reader.sources,
                                      'products': products, 'transformations': transformations})


def log_counts(reader, path):
    """Counts from a unified-log stream filtered to the host PID; the log itself is not bundled."""
    ok = failed = metal = 0
    functions = []
    for line in reader.raw(path).decode(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        message = json.loads(line).get('eventMessage', '')
        ok += ANE_OK in message
        failed += 'Failed with status' in message
        metal += METAL in message
        if message.startswith("Loading function '"):
            functions.append(message.split("'")[1])
    return {'ane_direct_request_success_rows': ok, 'ane_request_failure_rows': failed,
            'metal_shader_compile_rows': metal, 'loaded_functions': functions}


def request_rows(reader, root, names, inputs, n, record):
    rows = []
    for phase, ids in names.items():
        for name in ids:
            result = reader.json(root / f'native/host/{name}/result.json')
            command = reader.json(root / f'native/{name}-command.json')
            if 'input_ids' in result:
                if result['input_ids'] != inputs[f'reading_{n}'][:len(result['input_ids'])]:
                    raise ValueError(f'input_ids:{root.name}:{name}')
                result = {k: v for k, v in result.items() if k != 'input_ids'}
            rows.append(dict(record, phase=phase, command=command, result=result))
    return rows


def closure(reader, root, key):
    closed = reader.json(root / 'native/CLOSED.json')
    owned = reader.json(root / 'native/OWNED.json')
    if closed != {'returncode': 0, 'closed': True} or owned['returncode'] != 0:
        raise ValueError('host_not_closed:' + key)
    return ({'host': closed, 'reason': owned['reason'], 'cleanup_error': owned['cleanup_error']},
            {'segment': key, 'host_started_monotonic_s': owned['started_monotonic'], 'host_ended_monotonic_s': owned['ended_monotonic']})


def extract(workspace, output):
    reader = g4a.Reader(workspace)
    writer = Writer(reader, output)
    run, plan = Path(RUN), Path(PLAN)
    record = reader.json(run / 'RUN.json')
    config = record['config']
    if record['status'] != 'closed' or record['stop_reason'] is not None or record['failed_segments']:
        raise ValueError('run_not_closed_cleanly')
    terminal = reader.json(run / 'TERMINAL.json')
    if (terminal['status'] != 'closed' or terminal['returncode'] != 0 or terminal['cleanup_errors']
            or len(terminal['attempts']) != 1):
        raise ValueError('terminal_not_closed')
    launch = reader.json(run / 'LAUNCH.json')
    if launch != reader.json(plan / 'LAUNCH.json'):
        raise ValueError('launch_record_changed')
    frozen = reader.json(plan / 'CONFIG.json')
    if frozen != config:
        raise ValueError('run_config_differs_from_frozen')
    g4_config = reader.json(Path(G4A_PLAN) / 'CONFIG.json')
    g4a.launch_hash(launch, reader, f'{G4A_PLAN}/CONFIG.json')
    g4a.launch_hash(launch, reader, f'{PLAN}/CONFIG.json')

    tiers = []
    for weights, assets in (('fp16', config['fp16_q4_assets']), ('w4', config['w4_assets'])):
        for capacity, path in sorted(assets.items(), key=lambda x: int(x[0])):
            base = Path(path).resolve().relative_to(reader.workspace)
            model = base / f'{base.name}.aimodel'
            tier = reader.json(base / 'TIER.json')
            if tier['capacity'] != int(capacity):
                raise ValueError('tier_capacity:' + base.name)
            tiers.append({'weights': weights, 'capacity': int(capacity), 'tier': tier,
                          'metadata': reader.json(base / 'metadata.json'),
                          'metadata_sha256': g4a.launch_hash(launch, reader, (base / 'metadata.json').as_posix()),
                          'main_hash': reader.raw(model / 'main.hash').hex(),
                          'main_hash_file_sha256': g4a.launch_hash(launch, reader, (model / 'main.hash').as_posix()),
                          'main_mlirb_bytes': launch['asset_stats'][str(reader.workspace / model / 'main.mlirb')][0]})
    codes = Path(config['w4_codes']).resolve().relative_to(reader.workspace)
    inventory = reader.json(codes / 'INVENTORY.json')
    g4a.launch_hash(launch, reader, (codes / 'INVENTORY.json').as_posix())
    reference = reader.json(plan / 'w4-reference/REFERENCE.json')
    g4a.launch_hash(launch, reader, f'{PLAN}/w4-reference/REFERENCE.json')
    retired = reader.json(plan / 'assets/CANONICAL-RETIRED.json')
    admissions = {}
    for name in ADMISSIONS:
        root = plan / 'admission' / name
        probe = reader.json(root / 'PROBE.json')
        if not probe['passed'] or probe['status'] != 'closed':
            raise ValueError('admission_not_passed:' + name)
        logs = {p.name: log_counts(reader, p.relative_to(reader.workspace)) for p in
                sorted((reader.workspace / root).glob('*-log.ndjson'))}
        admissions[name] = {k: probe.get(k) for k in ('capacity', 'input', 'input_tokens', 'queries', 'outputs', 'same_host', 'truncated', 'asset')} | {
            'placement_logs': logs,
            'results': {q: {k: v for k, v in r.items() if k != 'protocol'} for q, r in probe['results'].items()}}
    writer.dump('asset-identity.json', {
        'tiers': tiers,
        'w4_preset': {'exporter_compression': 'coreai-models IOS_PRESETS 4bit_weight_palettized_group32',
                      'n_bits': 4, 'granularity': 'per_grouped_channel', 'axis': 0, 'group_size': 32,
                      'embedding': 'float16 (disable_embedding_quantization)', 'activations_and_kv': 'float16',
                      'canonical_metadata': retired['qwen3-4b-w4g32-ane-g6-canonical']['metadata']},
        'w4_codes': {'count': inventory['count'],
                     'modules': {k: {x: v[x] for x in ('indices_shape', 'lut_shape', 'group_size', 'vector_axis', 'max_index')}
                                 for k, v in inventory['modules'].items()},
                     'inventory_sha256': g4a.launch_hash(launch, reader, (codes / 'INVENTORY.json').as_posix())},
        'w4_reference': reference,
        'fp16_canonical_metadata': retired['qwen3-4b-fp16-ane-g6-q4-canonical']['metadata'],
        'admissions': admissions,
        'scope': 'Asset identity by recorded hashes, graph inventories and W4 code shapes; model payloads, codes and logs are not bundled. '
                 'Placement is counted from unified-log rows filtered to each admission host PID after the host reported ready.'})

    changes = {}
    for relative in (HOST_SWIFT, ENGINE_SWIFT):
        new = reader.raw(plan / 'q1-diagnosis/host-src' / relative).decode().splitlines()
        old = reader.raw(Path(G4_TIERS) / 'vendor/coreai-models' / relative).decode().splitlines()
        changes[relative] = {'g6_sha256': g4a.launch_hash(launch, reader, f'{PLAN}/q1-diagnosis/host-src/{relative}'),
                             'g4a_sha256': g4a.launch_hash(launch, reader, f'{G4_TIERS}/vendor/coreai-models/{relative}'),
                             'diff': [l for l in difflib.unified_diff(old, new, lineterm='', n=0)
                                      if l[:1] in '+-' and not l.startswith(('+++', '---'))]}
    writer.dump('runtime-implementation.json', {
        'g6_query_host_sha256': g4a.launch_hash(launch, reader, reader.relative(config['g6_host'])),
        'g4a_host_sha256': {arm: g4a.launch_hash(launch, reader, reader.relative(path)) for arm, path in g4_config['hosts'].items()},
        'changes_from_g4a_host': changes,
        'scope': 'Query sessions use the G4 A host rebuilt with a wider decode-query allow-list; the repeat arms use the G4 A binaries.'})

    inputs = reader.json(Path(g4_config['inputs']).resolve().relative_to(reader.workspace))
    requests, sessions, arms, idle, closures, windows = [], [], [], [], {}, []
    for capture in config['captures']:
        for s in capture:
            if s['kind'] == 'idle':
                report = reader.json(run / 'segments' / s['id'] / 'IDLE.json')
                if not report['passed']:
                    raise ValueError('idle_not_passed:' + s['id'])
                idle.append({'id': s['id'], 'seconds': s['seconds'], 'monitor': report['monitor'], 'window_ns': report['window_ns']})
            elif s['kind'] == 'query':
                root = run / 'segments' / s['id']
                report = reader.json(root / 'SESSION.json')
                if not report['passed'] or report['segment'] != s:
                    raise ValueError('session_not_passed:' + s['id'])
                names = {'boundary': [f'boundary-q{q}' for q in s['queries']]}
                for q in s['queries']:
                    names[f'decode_q{q}'] = [Path(r['path']).parent.name for r in report['decode'][str(q)]['requests']]
                if s['prefill']:
                    names['prefill'] = [Path(r['path']).parent.name for r in report['prefill']['requests']]
                requests += request_rows(reader, root, names, inputs, s['input_N'],
                                         {'segment': s['id'], 'weights': s['weights'], 'arm': 'ane',
                                          'input_N': s['input_N'], 'capacity': s['capacity']})
                closures[s['id']], window = closure(reader, root, s['id'])
                windows.append(window)
                sessions.append({'id': s['id'], 'weights': s['weights'], 'input_N': s['input_N'], 'capacity': s['capacity'],
                                 'queries': s['queries'], 'prefill': s['prefill'], 'monitor': report['monitor'],
                                 'requests': names, 'warmup': {q: {k: v for k, v in w.items() if k != 'protocol'}
                                                               for q, w in report['warmup'].items()},
                                 'prefill_decode_query': report['prefill']['decode_query'] if report['prefill'] else None})
            else:
                root = run / 'groups' / f"N{s['input_N']:05d}" / s['arm']
                report = reader.json(root / 'ARM.json')
                if not report['passed']:
                    raise ValueError('arm_not_passed:' + s['id'])
                names = {'boundary': ['boundary']} | {p: [Path(r['path']).parent.name for r in report[p]['requests']]
                                                     for p in ('full', 'prefill')}
                requests += request_rows(reader, root, names, inputs, s['input_N'],
                                         {'segment': s['id'], 'weights': 'fp16', 'arm': s['arm'],
                                          'input_N': s['input_N'], 'capacity': report['capacity']})
                closures[s['id']], window = closure(reader, root, s['id'])
                windows.append(window)
                arms.append({'id': s['id'], 'input_N': s['input_N'], 'arm': s['arm'], 'capacity': report['capacity'],
                             'monitor': report['monitor'], 'requests': names,
                             'counts': {p: report[p]['count'] for p in ('full', 'prefill')},
                             'warmup': report['warmup'], 'reference_1k': report.get('reference_1k')})
    writer.compressed('requests.jsonl.gz', requests)
    writer.dump('sessions.json', {'query_sessions': sessions, 'repeat_arms': arms, 'idle': idle})

    summary = reader.json(run / 'SUMMARY.json')
    for entry in summary['segments']:
        entry.pop('path')
        for phase in entry.get('phases', {}).values():
            phase['requests'] = [{'id': Path(r['path']).parent.name, 'stats': r['stats']} for r in phase['requests']]
    for entry in summary['g4a_repeat_arms']:
        entry.pop('path')
        for phase in entry['phases'].values():
            phase['requests'] = [{'id': Path(r['path']).parent.name, 'stats': r['stats']} for r in phase['requests']]
    writer.dump('summary-recorded.json', summary)

    status = {}
    names = sorted({s['monitor'] for s in sessions + arms + idle})
    last = {}
    def frames():
        for name in names:
            for frame in g4a.power_frames(reader, run / name / 'capture/power'):
                last[name] = max(last.get(name, 0), frame['receipt']['monotonic_after_ns'])
                yield dict(frame, capture=name)
    writer.compressed('power.jsonl.gz', frames())
    for name in names:
        capture = reader.json(run / f'{name}-CAPTURE.json')
        audit = capture['audit']
        if (capture['returncode'] != 0 or not audit['integrity_passed'] or not audit['capture_completed']
                or not audit['cleanup_passed'] or audit['clock_issues']):
            raise ValueError('capture_integrity:' + name)
        # The run summary recorded the last receipt only for captures holding query or idle segments.
        recorded = summary['captures'].get(name, {}).get('last_power_receipt_ns')
        if recorded is not None and recorded != last[name]:
            raise ValueError('last_power_receipt:' + name)
        status[name] = dict(audit, returncode=capture['returncode'], last_power_receipt_ns=last[name])
    writer.dump('capture-status.json', status)
    writer.dump('closures.json', {'run': {k: record[k] for k in ('status', 'stop_reason', 'failed_segments')},
                                  'terminal': {k: terminal[k] for k in ('status', 'returncode', 'cleanup_pending', 'cleanup_errors')},
                                  'hosts': closures})

    load_space = [json.loads(line) for line in reader.raw(run / 'LOAD-SPACE.jsonl').decode().splitlines()]
    resources = [json.loads(line) for line in reader.raw(run / 'resources.jsonl').decode().splitlines()]
    reclaim = [reader.json(run / f'RECLAIM-{i}.json') for i in range(1, len(config['captures']))]
    writer.dump('disk.json', {'load_gate': load_space,
        'observer': [{k: r[k] for k in ('monotonic_ns', 'disk_free_bytes', 'disk_reserve_bytes', 'memory_pressure')} for r in resources],
        'reclaim_breaks': reclaim, 'host_windows': windows,
        'scope': 'statfs free bytes from the run observer and the load gate; important-usage capacity at each load; the reclaim wait between captures.'})

    writer.dump('protocol.json', {
        'historical_import': True, 'date': '2026-09-14', 'model': 'Qwen/Qwen3-4B',
        'revision': reader.json(Path(G4_TIERS) / 'SOURCE-CHECK.json')['revision'],
        'captures': [[{k: v for k, v in s.items() if k not in ('asset', 'reference_1k')} for s in c] for c in config['captures']],
        'full_repetitions': g4_config['full_repetitions'], 'full_output_tokens': 257, 'decode_steps': 256,
        'prefill_counts': g4_config['prefill_counts'], 'quiet_seconds': g4_config['quiet_seconds'],
        'tail_seconds': g4_config['tail_seconds'], 'capture_tail_seconds': g4_config['capture_tail_seconds'],
        'energy_interior_margin_seconds': g4_config['energy_interior_margin_seconds'],
        'primary_extra_lag_ns': 2000000000, 'sensitivity_lag_ns': [0, 1000000000, 2000000000, 5000000000],
        'power_domains': ['cpu', 'gpu', 'ane'], 'idle_baseline_subtracted': False,
        'idle_seconds': config['idle_seconds'], 'reclaim_wait_seconds': config['reclaim_wait_seconds'],
        'window_seconds': config['window_seconds'], 'load_free_bytes': config['load_free_bytes'],
        'continuation': g4_config['continuation'], 'quality': g4_config['quality'],
        'repeat_decode_query': 8, 'gpu_capacity': 32768,
        'inputs_source': {'path': reader.relative(g4_config['inputs']), **reader.sources[reader.relative(g4_config['inputs'])]},
        'machine': {'chip': 'Apple M5 Pro', 'memory_GiB': 48},
        'runtime_sources': {reader.relative(k): v for k, v in launch['files'].items()},
        'scope': 'One night: FP16 decode with query 8 and 4 on six matched graphs in one host session each; the upstream iOS 4-bit '
                 'palettized preset at 1K and 4K; a repeat of the G4 A arms with the arm order reversed; idle captures. '
                 'Per-block component energy admission; no exclusive per-operation placement.'})
    writer.provenance([
        'Strip the workspace prefix; refuse any remaining personal or temporary-directory path.',
        'Retain every boundary, full and prefill request result with its command for query sessions and repeat arms; drop input_ids after checking them; no logits.',
        'Retain tier graph inventories, recorded asset hashes, the W4 code shapes and the dequantized-reference summary; no model payloads or codes.',
        'Count ANE direct-request, request-failure and Metal shader-compile rows in each admission log; the logs are not bundled.',
        'Re-serialize CPU/GPU/ANE power fields from each plist frame of the four captures, as the G4 A importer does, tagged by capture.',
        'Retain the recorded summary so recomputation can be compared with what the run wrote.',
        'Retain load-gate, observer and reclaim-wait disk fields and host start/end times; drop process IDs and argv.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

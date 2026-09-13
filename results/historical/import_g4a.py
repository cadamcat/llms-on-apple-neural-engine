"""Import the closed G4 A run and its disk observations into portable bundles; never execute a device."""
import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import plistlib
import re

RUN = 'results/g4-a-20260913-124133'
TIERS = 'workplans/g4-cplus256-20260912'
G3_PREP = 'workplans/g3-night-20260912-energy-r5'
MODEL_SOURCE = 'workplans/g3-qwen3-4b-20260911/model-source'
GPU_ASSET = 'workplans/g3-qwen3-4b-20260911/assets/qwen3-4b-fp16-gpu-c32768'
LEAK = 'workplans/ane-compiler-reclaim-20260913/evidence'
HOST_SWIFT = 'vendor/coreai-models/swift/Sources/Tools/g3-flow-host/G3Host.swift'
ENGINE_SWIFT = 'vendor/coreai-models/swift/Sources/CoreAILanguageModels/InferenceEngines/'


class Reader:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.sources = {}

    def raw(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.workspace / path
        data = path.read_bytes()
        self.sources[path.resolve().relative_to(self.workspace).as_posix()] = {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return data

    def json(self, path):
        return json.loads(self.raw(path))

    def relative(self, path):
        return Path(path).resolve().relative_to(self.workspace).as_posix()

    def clean(self, value):
        if isinstance(value, dict):
            return {self.clean(k): self.clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            value = value.replace(str(self.workspace) + '/', '')
            if '/Users/' in value or '/var/folders/' in value:
                raise ValueError('personal_path')
        return value


class Writer:
    def __init__(self, reader, output):
        self.reader = reader
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)

    def dump(self, name, value):
        (self.output / name).write_text(json.dumps(self.reader.clean(value), indent=2,
                                                   ensure_ascii=False, allow_nan=False) + '\n')

    def compressed(self, name, records):
        with (self.output / name).open('xb') as target:
            with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
                for row in records:
                    stream.write((json.dumps(self.reader.clean(row), separators=(',', ':'),
                                             allow_nan=False) + '\n').encode())

    def provenance(self, transformations):
        products = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(self.output.iterdir())}
        self.dump('provenance.json', {'importer': 'results/historical/import_g4a.py',
            'sources': self.reader.sources, 'products': products,
            'transformations': transformations})


def launch_hash(launch, reader, path):
    """The file read now must be the file frozen at launch."""
    absolute = str(reader.workspace / path)
    digest = hashlib.sha256(reader.raw(path)).hexdigest()
    if launch['files'].get(absolute) != digest:
        raise ValueError('launch_identity:' + path)
    return digest


def power_frames(reader, power):
    index = [json.loads(line) for line in reader.raw(power / 'records.jsonl').decode().splitlines()]
    current = data = None
    for row in index:
        if current != row['file']:
            current = row['file']
            data = reader.raw(power / current)
        frame = data[row['byte_start']:row['byte_end']]
        obj = plistlib.loads(frame[:-1])
        processor = obj['processor']
        fields = {'elapsed_ns': obj['elapsed_ns'], 'timestamp': obj['timestamp'],
                  'thermal_pressure': obj.get('thermal_pressure'),
                  'processor': {k: processor[k] for k in
                      ('cpu_power', 'cpu_energy', 'gpu_power', 'gpu_energy',
                       'ane_power', 'ane_energy', 'combined_power')}}
        if 'invalid' in obj:
            fields['invalid'] = obj['invalid']
        if 'invalid' in processor:
            fields['processor']['invalid'] = processor['invalid']
        yield {'index': row['index'], 'receipt': row['receipt'],
               'source': {k: row[k] for k in ('file', 'byte_start', 'byte_end')},
               'source_frame_sha256': hashlib.sha256(frame).hexdigest(),
               'plist_fields': plistlib.dumps(fields, sort_keys=True).decode(),
               'recorded_decoded': row['decoded']}


def extract(workspace, output):
    reader = Reader(workspace)
    writer = Writer(reader, output)
    run = Path(RUN)
    record = reader.json(run / 'RUN.json')
    config = record['config']
    if record['status'] != 'closed' or record['attempt'] != 1 or record['stop_reason'] is not None:
        raise ValueError('run_not_closed_in_one_attempt')
    terminal = reader.json(run / 'TERMINAL.json')
    if terminal['status'] != 'closed' or terminal['returncode'] != 0 or terminal['cleanup_errors']:
        raise ValueError('terminal_not_closed')
    launch = reader.json(run / 'LAUNCH.json')

    # Model source: the same checkout G3 used; configuration and safetensors headers only.
    model_config = reader.json(Path(MODEL_SOURCE) / 'config.json')
    weight_index = reader.json(Path(MODEL_SOURCE) / 'model.safetensors.index.json')
    headers = {}
    for name in sorted(set(weight_index['weight_map'].values())):
        with (reader.workspace / MODEL_SOURCE / name).open('rb') as stream:
            length = int.from_bytes(stream.read(8), 'little')
            raw = stream.read(length)
        headers[name] = {'header_bytes': length, 'header_sha256': hashlib.sha256(raw).hexdigest()}

    active = reader.json(Path(TIERS) / 'ACTIVE-ASSETS.json')
    tiers = []
    for n, c in zip(config['inputs_N'], config['capacities'], strict=True):
        base = Path(config['ane_assets'][str(c)]).resolve().relative_to(reader.workspace)
        model = base / f'{base.name}.aimodel'
        tier = reader.json(base / 'TIER.json')
        compiled = reader.json(Path(TIERS) / f'COMPILED-{c}.json')
        listed = next(a for a in active['active'] if a['capacity'] == c)
        if (tier['capacity'] != c or listed['input_tokens'] != n or not compiled['complete']
                or compiled['inference_executed']
                or reader.raw(model / 'main.hash').hex() != listed['native_main_hash']):
            raise ValueError(f'tier_identity:{c}')
        tiers.append({'input_N': n, 'capacity': c, 'tier': tier,
            'metadata': reader.json(base / 'metadata.json'),
            'metadata_sha256': launch_hash(launch, reader, (base / 'metadata.json').as_posix()),
            'main_hash': reader.raw(model / 'main.hash').hex(),
            'main_hash_file_sha256': launch_hash(launch, reader, (model / 'main.hash').as_posix()),
            'main_mlirb_bytes': launch['asset_stats'][str(reader.workspace / model / 'main.mlirb')][0],
            'compile_receipt': {k: compiled[k] for k in ('complete', 'inference_executed', 'preferred_compute')} |
                {'functions': [r['name'] for r in compiled['records']]}})
    gpu = Path(GPU_ASSET)
    admissions = {}
    for path in launch['admissions']:
        relative = reader.relative(path)
        check = reader.json(relative)
        if not check['passed']:
            raise ValueError('admission_not_passed:' + relative)
        admissions[relative] = {k: check[k] for k in
            ('capacity', 'arm', 'passed', 'formal_benchmark', 'quality', 'placement', 'load_seconds')}
    writer.dump('asset-identity.json', {
        'model_source': {'model': 'Qwen/Qwen3-4B', 'config': model_config, 'source_headers': headers,
                         'source_check': reader.json(Path(TIERS) / 'SOURCE-CHECK.json')},
        'ane_tiers': tiers,
        'ane_stride_basis': active['ANE_stride_basis'],
        'gpu': {'capacity': 32768, 'export': reader.json(gpu / 'export.json'),
                'export_sha256': launch_hash(launch, reader, (gpu / 'export.json').as_posix()),
                'main_hash': reader.raw(gpu / f'{gpu.name}.aimodel/main.hash').hex()},
        'flow_admissions': admissions,
        'scope': 'Asset identity by recorded hashes and graph inventories; bytecode payloads were not rehashed at launch and are not bundled.'})

    host = Path(TIERS) / HOST_SWIFT
    host_source = reader.raw(host).decode()
    begin = host_source.index('        let engine=try await EngineFactory.createEngine')
    end = host_source.index('        guard engine.supportsLogits', begin)
    selection = host_source[begin:end]
    if '"static-shape":"coreai-sequential"' not in selection:
        raise ValueError('host_engine_selection_changed')
    changes = {}
    for relative in (HOST_SWIFT, ENGINE_SWIFT + 'CoreAIStaticShapeEngine.swift', ENGINE_SWIFT + 'CoreAISequentialEngine.swift'):
        new = reader.raw(Path(TIERS) / relative).decode().splitlines()
        old = reader.raw(Path(G3_PREP) / relative).decode().splitlines()
        diff = [line for line in difflib.unified_diff(old, new, lineterm='', n=0)
                if line[:1] in '+-' and not line.startswith(('+++', '---'))]
        changes[relative] = {'g4_sha256': launch_hash(launch, reader, f'{TIERS}/{relative}'),
                             'lines_changed_from_g3': len(diff), 'diff': diff}
    writer.dump('runtime-implementation.json', {
        'source': f'{TIERS}/{HOST_SWIFT}', 'engine_selection_excerpt': selection,
        'ane': 'Core AI StaticShapeEngine', 'gpu': 'Core AI CoreAISequentialEngine',
        'changes_from_g3_host': changes,
        'host_sha256': {arm: launch_hash(launch, reader, reader.relative(path))
                        for arm, path in config['hosts'].items()},
        'scope': 'Host engine selection and the G4 fixed-capacity additions; not exclusive per-operation device placement.'})

    inputs = reader.json(Path(config['inputs']).resolve().relative_to(reader.workspace))
    continuation_source = reader.json(reader.relative(config['continuation_source']))
    if continuation_source['generated_ids'] != config['continuation']:
        raise ValueError('continuation_source')

    requests, arms, closures, windows = [], [], {}, []
    for n in config['inputs_N']:
        for arm in ('ane', 'gpu'):
            root = run / f'groups/N{n:05d}/{arm}'
            report = reader.json(root / 'ARM.json')
            if not report['passed'] or report['monitor'] != 'monitor-1':
                raise ValueError(f'arm_not_passed:{n}:{arm}')
            closed = reader.json(root / 'native/CLOSED.json')
            owned = reader.json(root / 'native/OWNED.json')
            if closed != {'returncode': 0, 'closed': True} or owned['returncode'] != 0:
                raise ValueError(f'host_not_closed:{n}:{arm}')
            closures[f'{n}:{arm}'] = {'host': closed, 'reason': owned['reason'], 'cleanup_error': owned['cleanup_error']}
            windows.append({'input_N': n, 'arm': arm, 'host_started_monotonic_s': owned['started_monotonic'],
                            'host_ended_monotonic_s': owned['ended_monotonic']})
            ids = {'boundary': ['boundary']}
            for phase in ('full', 'prefill'):
                ids[phase] = [Path(r['path']).parent.name for r in report[phase]['requests']]
            for phase, names in ids.items():
                for name in names:
                    result = reader.json(root / f'native/host/{name}/result.json')
                    command = reader.json(root / f'native/{name}-command.json')
                    if 'input_ids' in result:
                        if result['input_ids'] != inputs[f'reading_{n}'][:len(result['input_ids'])]:
                            raise ValueError(f'input_ids:{n}:{arm}:{name}')
                        result = {k: v for k, v in result.items() if k != 'input_ids'}
                    requests.append({'input_N': n, 'arm': arm, 'capacity': report['capacity'],
                                     'phase': phase, 'command': command, 'result': result})
            arms.append({'input_N': n, 'arm': arm, 'capacity': report['capacity'], 'passed': True,
                         'requests': ids, 'counts': {p: report[p]['count'] for p in ('full', 'prefill')},
                         'warmup': report['warmup'], 'reference_1k': report.get('reference_1k')})
    writer.compressed('requests.jsonl.gz', requests)
    writer.dump('arms.json', arms)

    summary = reader.json(run / 'SUMMARY.json')
    for entry in summary['arms']:
        entry.pop('path')
        for phase in entry['phases'].values():
            phase['requests'] = [{'id': Path(r['path']).parent.name, 'stats': r['stats']} for r in phase['requests']]
    writer.dump('summary-recorded.json', summary)

    capture = reader.json(run / 'monitor-1-CAPTURE.json')
    audit = capture['audit']
    if (capture['returncode'] != 0 or not audit['integrity_passed'] or not audit['capture_completed']
            or not audit['cleanup_passed'] or audit['clock_issues']):
        raise ValueError('capture_integrity')
    writer.compressed('power.jsonl.gz', power_frames(reader, run / 'monitor-1/capture/power'))
    writer.dump('capture-status.json', dict(audit, returncode=capture['returncode'],
        last_power_receipt_ns=summary['captures']['monitor-1']['last_power_receipt_ns']))
    writer.dump('closures.json', {'run': {k: record[k] for k in ('status', 'attempt', 'stop_reason')},
        'terminal': {k: terminal[k] for k in ('status', 'returncode', 'cleanup_pending', 'cleanup_errors')},
        'hosts': closures})

    load_space = [json.loads(line) for line in reader.raw(run / 'LOAD-SPACE.jsonl').decode().splitlines()]
    resources = [json.loads(line) for line in reader.raw(run / 'resources.jsonl').decode().splitlines()]
    writer.dump('disk.json', {'load_gate': load_space,
        'observer': [{k: r[k] for k in ('monotonic_ns', 'disk_free_bytes', 'disk_reserve_bytes', 'memory_pressure')}
                     for r in resources],
        'host_windows': windows,
        'scope': 'statfs free bytes from the run observer and the load gate; important-usage capacity from NSURLVolumeAvailableCapacityForImportantUsageKey at each load.'})

    writer.dump('protocol.json', {
        'historical_import': True, 'date': '2026-09-13', 'model': 'Qwen/Qwen3-4B',
        'revision': reader.json(Path(TIERS) / 'SOURCE-CHECK.json')['revision'], 'precision': 'fp16',
        'inputs_N': config['inputs_N'], 'ane_capacities': config['capacities'], 'gpu_capacity': 32768,
        'run_order': config['run_order'], 'full_repetitions': config['full_repetitions'],
        'full_output_tokens': 257, 'decode_steps': 256, 'prefill_counts': config['prefill_counts'],
        'prefill_count_basis': 'at least 45 seconds of the faster arm from an earlier interrupted run',
        'quiet_seconds': config['quiet_seconds'], 'tail_seconds': config['tail_seconds'],
        'capture_tail_seconds': config['capture_tail_seconds'],
        'energy_interior_margin_seconds': config['energy_interior_margin_seconds'],
        'primary_extra_lag_ns': 2000000000, 'sensitivity_lag_ns': [0, 1000000000, 2000000000, 5000000000],
        'power_domains': ['cpu', 'gpu', 'ane'], 'idle_baseline_subtracted': False,
        'continuation': config['continuation'], 'continuation_scope': config['continuation_scope'],
        'decode_query': 8, 'quality': config['quality'], 'load_free_bytes': config['load_free_bytes'],
        'load_space_metric': config['load_space_metric'],
        'inputs_source': {'path': reader.relative(config['inputs']),
                          **reader.sources[reader.relative(config['inputs'])]},
        'machine': {'chip': 'Apple M5 Pro', 'memory_GiB': 48},
        'runtimes': {'ane': 'Core AI StaticShapeEngine', 'gpu': 'Core AI CoreAISequentialEngine'},
        'runtime_sources': {reader.relative(k): v for k, v in launch['files'].items()},
        'scope': 'One host session per arm: boundary warmup, three teacher-forced full requests, one prefill block; per-block component energy admission. No exclusive per-operation placement or direct memory-bandwidth measurement.'})
    writer.provenance([
        'Strip the workspace prefix; refuse any remaining personal or temporary-directory path.',
        'Retain every boundary, full and prefill request result with its command; drop input_ids after checking them against the recorded input IDs; no logits.',
        'Retain per-tier graph inventories, recorded asset hashes, compile receipts and short ANE flow admissions; no model payloads.',
        'Re-serialize only elapsed time, timestamp, thermal state, invalid flags and CPU/GPU/ANE power/energy fields from each plist frame; retain frame ranges, hashes, receipts and recorded decoded values.',
        'Retain the recorded summary so recomputation can be compared with what the run wrote.',
        'Retain load-gate and observer disk fields and host start/end times; drop process IDs and argv.'])


TMP = re.compile(r'/private/var/folders/[^/\s]+/[^/\s]+/T/')
LSOF = re.compile(r'^(?P<command>\S+)\s+(?P<pid>\d+)\s+(?P<user>\S+)\s+(?P<fd>\S+)\s+(?P<type>\S+)\s+'
                  r'(?P<device>\S+)\s+(?P<size>\d+)\s+(?:(?P<nlink>\d+)\s+)?(?P<node>\d+)\s+(?P<name>/\S+)$')


def listing(text, owner):
    rows = []
    for line in text.splitlines():
        match = LSOF.match(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append({'command': row['command'], 'pid': int(row['pid']),
                     'user': 'root' if row['user'] == 'root' else 'session user',
                     'fd': row['fd'], 'size_bytes': int(row['size']),
                     'nlink': None if row['nlink'] is None else int(row['nlink']),
                     'name': TMP.sub('$TMPDIR/', row['name'])})
    if not rows or any(owner in json.dumps(r) for r in rows):
        raise ValueError('listing_redaction')
    return rows


def extract_leak(workspace, output, owner):
    reader = Reader(workspace)
    writer = Writer(reader, output)
    leak = Path(LEAK)
    first = reader.raw(leak / 'LSOF-20260913-1335.txt').decode()
    second = reader.raw(leak / 'LSOF-20260913-1400.txt').decode()
    df_ps = reader.raw(leak / 'DF-PS-20260913.txt').decode()
    log = reader.raw(leak / 'RECLAIM-LOG-20260913.txt').decode().strip().splitlines()
    samples = []
    lines = df_ps.splitlines()
    for i, line in enumerate(lines):
        if re.fullmatch(r'\d\d:\d\d:\d\d', line):
            fields = lines[i + 1].split()
            samples.append({'local_time': line, 'filesystem_kib': int(fields[1]), 'used_kib': int(fields[2]),
                            'available_kib': int(fields[3]),
                            'service_1720_running': any(l.split()[:1] == ['1720'] for l in lines[i + 2:i + 4])})
    build = re.search(r'BuildVersion:\s+(\S+)', df_ps).group(1)
    version = re.search(r'ProductVersion:\s+(\S+)', df_ps).group(1)
    started = re.search(r'1720 \d+ (\w{3} \w{3} +\d+ [\d:]+ \d{4})', df_ps).group(1)
    writer.dump('observations.json', {
        'macos': {'version': version, 'build': build},
        'service_1720_started_local': started,
        'listing_1335': {'command': "sudo lsof -nP +L1 | awk 'NR==1 || $7>100000000'",
                         'during': 'G4 A, ANE 4K prefill block', 'rows': listing(first, owner)},
        'listing_1400': {'command': "sudo lsof -nP +L1 | awk 'NR==1 || $7>100000000'",
                         'during': 'G4 A, GPU 500 arm, after the service was restarted at 13:46:16',
                         'rows': listing(second, owner)},
        'free_space': samples,
        'reclaim_log': log,
        'scope': 'One machine, one macOS build, Qwen3-4B FP16 ANE assets. The user temporary directory is written as $TMPDIR.'})
    writer.provenance([
        'Parse lsof rows from the saved terminal output; drop prompts, the wrapped command echo, inode numbers and the session user name; rewrite the per-user temporary directory as $TMPDIR.',
        'Keep df available/used KiB and ANECompilerService PIDs at each recorded time; keep the macOS product and build version.',
        'Keep the reclaim daemon log line unchanged.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--leak-output', type=Path)
    parser.add_argument('--owner', help='session user name that must not survive redaction')
    args = parser.parse_args()
    if args.output:
        extract(args.workspace, args.output)
    if args.leak_output:
        extract_leak(args.workspace, args.leak_output, args.owner or Path.home().name)

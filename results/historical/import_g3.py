"""Import the closed G3 runs into a portable bundle; never execute a device."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import plistlib


def extract(workspace, output):
    workspace = Path(workspace).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def source(path):
        path = Path(path)
        if not path.is_absolute():
            path = workspace / path
        raw = path.read_bytes()
        sources[path.relative_to(workspace).as_posix()] = {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return raw

    def read(path):
        return json.loads(source(path))

    def clean(value):
        if isinstance(value, dict):
            return {clean(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            value = value.replace(str(workspace) + '/', '')
            if '/Users/' in value:
                raise ValueError('personal_path')
        return value

    def dump(name, value):
        (output / name).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                             allow_nan=False) + '\n')

    def compressed(name, records):
        with (output / name).open('wb') as target:
            with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
                for row in records:
                    stream.write((json.dumps(clean(row), separators=(',', ':'),
                                             allow_nan=False) + '\n').encode())

    prep = workspace / 'workplans/g3-night-20260912-energy-r5'
    cfg = read(prep / 'CONFIG.json')
    model_source = workspace / 'workplans/g3-qwen3-4b-20260911/model-source'
    model_config = read(model_source / 'config.json')
    weight_index = read(model_source / 'model.safetensors.index.json')
    tensors, headers = {}, {}
    for name in sorted(set(weight_index['weight_map'].values())):
        with (model_source / name).open('rb') as stream:
            length = int.from_bytes(stream.read(8), 'little')
            raw = stream.read(length)
        header = json.loads(raw)
        headers[name] = {'header_bytes': length, 'header_sha256': hashlib.sha256(raw).hexdigest()}
        tensors.update({key: {'shape': value['shape'], 'dtype': value['dtype']}
                        for key, value in header.items() if key != '__metadata__'})
    if set(tensors) != set(weight_index['weight_map']):
        raise ValueError('model_tensor_inventory')
    dump('model-structure.json', {'model': cfg['model'], 'revision': cfg['revision'],
        'config': model_config, 'tensors': tensors, 'source_headers': headers,
        'source_total_bytes': weight_index['metadata']['total_size'],
        'scope': 'Source tensor metadata only. G3 runtime weights are FP16; source weights are BF16.'})
    launch = read(prep / 'LAUNCH.json')
    env = read(prep / 'ENVIRONMENT.json')
    pins = read(prep / 'vendor/coreai-models/Package.resolved')
    dump('asset-identity.json', read(prep / 'ASSET-IDENTITY.json'))
    host_path = prep / 'vendor/coreai-models/swift/Sources/Tools/g3-flow-host/G3Host.swift'
    host_source = source(host_path).decode()
    engine_begin = host_source.index('        let engine=try await EngineFactory.createEngine')
    engine_end = host_source.index('        guard engine.supportsLogits', engine_begin)
    selection = host_source[engine_begin:engine_end]
    if '"static-shape":"coreai-sequential"' not in selection:
        raise ValueError('host_engine_selection_changed')
    dump('runtime-implementation.json', {
        'source': host_path.relative_to(workspace).as_posix(),
        'source_sha256': hashlib.sha256(host_source.encode()).hexdigest(),
        'engine_selection_excerpt': selection,
        'ane': 'Core AI StaticShapeEngine', 'gpu': 'Core AI CoreAISequentialEngine',
        'scope': 'Host engine selection, not exclusive per-operation device placement.'})
    input_dir = Path(cfg['inputs']).parent
    inputs = read(input_dir / 'inputs.json')
    manifest = read(input_dir / 'INPUT-MANIFEST.json')
    selected = {k: v for k, v in inputs.items() if k.startswith('reading_') or k == 'quality_short'}
    dump('inputs.json', selected)
    dump('input-identity.json', {'model': manifest['model'], 'revision': manifest['revision'],
         'inputs': {k: v for k, v in manifest['inputs'].items() if k in selected},
         'sources': manifest['sources'], 'source_text_in_bundle': False})
    configurations, closures, blocks, records, quality = {}, {}, [], [], {}
    for run in ('r4', 'r5', 'r6'):
        root = workspace / f'results/g3-night-20260912-{run}'
        config = read(root / 'CONFIG.json')
        configurations[run] = {k: config[k] for k in ('model', 'revision', 'precision', 'capacity')}
        index = [json.loads(line) for line in source(root / 'requests.jsonl').splitlines()]
        for row in index:
            path = root / row['row']
            raw = read(path)
            command = read(path.parents[2] / (raw['id'] + '-command.json'))
            records.append({'run': run, 'host': path.parents[1].parent.name,
                            'stage': row['stage'], 'group': row['group'],
                            'command': command, 'result': raw})
        if (root / 'BLOCKS.json').exists():
            blocks.extend(dict(b, run=run, capture='r5') for b in read(root / 'BLOCKS.json'))
        guard = read(root / 'guard/result.json')
        closures[run] = {k: guard[k] for k in ('returncode', 'reason', 'remaining_pids', 'passed')}
        if run != 'r4' and (not guard['passed'] or guard['remaining_pids']):
            raise ValueError('unclosed_run:' + run)
        if run == 'r4':
            quality = {p.name: read(p) for p in sorted((root / 'admission').glob('*.json'))}
    compressed('requests.jsonl.gz', records)
    dump('blocks.json', blocks)
    dump('quality-records.json', {'controls': quality,
         'scope': 'Recorded first-output and cache checks; logits and model assets are not bundled. Not a full-model quality evaluation.'})
    power = workspace / 'results/g3-night-20260912-r5/monitors/main/capture/power'
    power_index = [json.loads(line) for line in source(power / 'records.jsonl').splitlines()]

    def frames():
        current = None
        data = None
        for row in power_index:
            if current != row['file']:
                current = row['file']
                data = source(power / current)
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
    compressed('power.jsonl.gz', frames())
    audit = read(power.parent.parent / 'audit.json')
    if not audit['integrity_passed'] or not audit['cleanup_passed'] or audit['clock_issues']:
        raise ValueError('capture_integrity')
    dump('capture-status.json', audit)
    dump('closures.json', closures)
    dump('protocol.json', {
        'historical_import': True, 'date': '2026-09-12',
        'model': cfg['model'], 'revision': cfg['revision'], 'precision': cfg['precision'],
        'capacity': cfg['capacity'], 'lengths': cfg['lengths'], 'families': cfg['families'],
        'ANE_shapes': cfg['ANE_shapes'], 'quality': cfg['quality'],
        'coverage_decode_steps': cfg['decode_forward_steps'], 'stage_decode_steps': 1024,
        'stage_prefill_counts': {str(b['context']): b['requested_count'] for b in blocks
                                 if b['run'] == 'r5' and b['mode'] == 'prefill'},
        'supplement_prefill_count': next(b['requested_count'] for b in blocks if b['run'] == 'r6'),
        'runs': configurations, 'roles': {'r4': 'single coverage requests and output controls',
            'r5': 'warmed stage blocks and shared capture', 'r6': 'longer 500-token prefill pair'},
        'preferred_500_prefill_run': 'r6', 'primary_extra_lag_ns': 2000000000,
        'sensitivity_lag_ns': [0, 1000000000, 2000000000, 5000000000],
        'power_domains': ['cpu', 'gpu', 'ane'], 'idle_baseline_subtracted': False,
        'recovery_in_primary_energy': False, 'decode_forced_ids': False,
        'machine': {'chip': 'Apple M5 Pro', 'memory_GiB': 48},
        'runtimes': {'ane': 'Core AI StaticShapeEngine', 'gpu': 'Core AI CoreAISequentialEngine'},
        'environment': env, 'swift_dependencies': pins['pins'],
        'host_sha256': launch['files']['bin/g3-flow-host'],
        'runtime_sources': launch['files'],
        'scope': 'Two complete FP16 model implementations; host and framework work included. No exclusive per-operation placement or direct UMA bandwidth measurement.'})
    products = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    dump('provenance.json', {'sources': sources, 'products': products,
        'transformations': [
            'Strip the workspace prefix; omit model assets, logits arrays, machine name and process inventories.',
            'Retain complete request/command timing and graph records under separate run IDs.',
            'Re-serialize only elapsed time, timestamp, thermal state, invalid flags and CPU/GPU/ANE power/energy fields from each original plist frame; retain original frame ranges and hashes.',
            'Retain receipt anchors and recorded decoded values so portable parsing can check every selected source field.',
            'Import source model configuration and tensor shapes/dtypes from safetensors headers; do not read tensor payloads for this metadata.',
            'Input IDs are from this repository documentation; text snapshots remain identified by hash.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

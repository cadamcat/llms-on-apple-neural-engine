"""Extract a portable G2 evidence bundle from the closed research workspace.

Only explicit invocation reads the workspace. No models, arrays, device calls or
system process listings are copied. Every read source is identified in provenance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

BASE = Path('results/g2-w4a16-night-20260910')
ANALYSIS = BASE / 'analysis-20260911-r4'
RUN = BASE / 'night-20260911-r4'
META_FIELDS = ('engine', 'pid', 'request', 'request_id', 'positions', 'tile',
               'outer_ns', 'worker_ns', 'input_index', 'input_file_sha256',
               'output_hash', 'numerical_passed', 'stage_calls')
EVENT_FIELDS = ('id', 'index', 'stream', 'status', 'arrival_ns', 'submitted_ns',
                'service_start_ns', 'completed_ns', 'deadline_ns', 'deadline_violation')


def extract(workspace, output):
    sources = {}

    def source(path):
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(workspace)
        raw = (workspace / path).read_bytes()
        sources[path.as_posix()] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return raw

    def read(path):
        return json.loads(source(path))

    def rows(path):
        # One closed stream at a time; never retain all request logs in memory.
        for line in source(path).splitlines():
            if line.strip():
                yield json.loads(line)

    def clean(obj):
        if isinstance(obj, dict):
            return {clean(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, str):
            text = obj.replace(str(workspace) + '/', '')
            if '/Users/' in text:
                raise ValueError('Unexpected personal path in selected evidence')
            return text
        return obj

    def dump(name, obj):
        (output / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2,
                                             allow_nan=False) + '\n')

    def compressed(name, records):
        with (output / name).open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
                for record in records:
                    stream.write((json.dumps(clean(record), separators=(',', ':'),
                                              allow_nan=False) + '\n').encode())

    audit = read(ANALYSIS / 'device-audit-v2/audit.json')
    closure = read(ANALYSIS / 'final-closure/closure.json')
    assert all(r.get('passed') for r in audit['checks'].values()) and closure['passed']
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(BASE / 'execution-20260911-r4/manifest.json')
    run = read(RUN / 'run.json')
    criteria = read(ANALYSIS / 'criteria.json')['criteria']
    speed = read(ANALYSIS / 'device-audit-v2/speed.json')
    slots = read(ANALYSIS / 'device-audit-v2/slots.json')
    p2 = []
    for name, host in speed.items():
        requests = list(rows(RUN / name / 'requests.jsonl'))
        for index, cell in enumerate(host['cells']):
            group = requests[index * 45:(index + 1) * 45]
            assert len(group) == 45
            p2.append(dict(cell, host_id=name, engine=host['engine'], round=host['round'],
                           host=host['host'], records=[{k: r.get(k) for k in
                           (*META_FIELDS, 'category', 'index')} for r in group]))
    compressed('p2.jsonl.gz', p2)
    for name, slot in slots.items():
        slot['events_file'] = name + '.jsonl.gz'
        if slot['config']['mode'] == 'arrivals':
            slot['offsets'] = {stream: read(slot['config'][stream + '_arrivals'])['offsets_ns']
                               for stream in slot['service']}
        def events():
            for r in rows(RUN / name / 'requests.jsonl'):
                entry = {k: r[k] for k in EVENT_FIELDS if k in r}
                if 'metadata' in r:
                    entry['metadata'] = {k: r['metadata'][k] for k in META_FIELDS if k in r['metadata']}
                yield entry
        compressed(slot['events_file'], events())
    compressed('arrivals.jsonl.gz', ({'slot_id': name, 'stream': stream, 'offsets_ns': values}
        for name, slot in slots.items() for stream, values in slot.pop('offsets', {}).items()))
    dump('slots.json', slots)
    telemetry = read(ANALYSIS / 'device-audit-v2/telemetry.json')
    sensor_fields = ('sample_monotonic_ns', 'macmon_cpu_sensor_mean_c',
                     'macmon_gpu_sensor_mean_c', 'fan0_rpm', 'fan1_rpm')
    compressed('sensors.jsonl.gz', ({k: r[k] for k in sensor_fields} for r in telemetry['sensors']))
    compressed('resources.jsonl.gz', ({k: v for k, v in r.items() if k != 'epoch_ns'}
                                    for r in telemetry['resources']))
    compressed('thermal.jsonl.gz', ({k: r[k] for k in ('sample_monotonic_ns', 'thermal_state')}
                                  for r in telemetry['thermal']))
    p0 = read(ANALYSIS / 'device-audit-v2/p0.json')
    dump('p0.json', [{'engine': r['engine'], 'host': r['host'], 'passed': r['passed'],
                     'checkpoints': [{'measured_requests': c['measured_requests'],
                                     'monotonic_ns': c['monotonic_ns'],
                                     'rss_bytes': c['memory']['rss_bytes'],
                                     'footprint_bytes': c['memory']['footprint_bytes']}
                                    for c in r['checkpoints']]} for r in p0])
    dump('thermal-starts.json', read(ANALYSIS / 'device-audit-v2/thermal-starts.json'))
    dump('thermal-windows.json', read(ANALYSIS / 'device-audit-v2/thermal-windows.json'))
    summary = read(ANALYSIS / 'SUMMARY.json')
    env = read(ANALYSIS / 'environment-followup-20260911/ventura-evidence.json')
    # Personal usage is not evidence: keep the display condition, drop process rankings and message text.
    display = {'date': env['owner_report']['date'],
               'source': 'Statement recorded after G2 completion; not contemporaneous telemetry.',
               'text': 'The Mac stayed awake and the Ventura dynamic screensaver loaded automatically.'}
    summary['background'] = {k: v for k, v in summary['background'].items() if k != 'top5_appearance_counts'}
    summary['environment_followup'] = {k: v for k, v in summary['environment_followup'].items()
                                       if k != 'owner_report'} | {'post_run_report': display}
    dump('expected-summary.json', summary)
    dump('environment.json', {k: v for k, v in env.items() if k not in ('snapshots', 'owner_report')}
         | {'post_run_report': display})
    power = read(ANALYSIS / 'power-audit-v1/power-audit.json')
    dump('power-status.json', {k: power[k] for k in ('status', 'scope', 'capture', 'source_identity',
         'alignment', 'epoch_monotonic_offset_bounds_seconds', 'matched_energy_pairs')})
    prepared = read(BASE / 'preflight/implementation-20260911/bundle/prepared.json')
    packages = read(BASE / 'preflight/relocation/post-repair-verification.json')
    assert packages['distribution_metadata_unchanged_since_snapshot'] is True
    xcode = read(BASE / 'preflight/swift-recheck/recheck.json')['toolchain_now']
    system = read(BASE / 'night-20260911-r3/environment.json')
    client_raw = source('scripts/g2_client.py')
    assert hashlib.sha256(client_raw).hexdigest() == manifest['source_hashes'][str(workspace / 'scripts/g2_client.py')]
    assert "performance/.venv/bin/python'), '-B', str(ROOT/'scripts/g2_gpu.py')" in client_raw.decode()
    source(ANALYSIS / 'REPORT.md')
    source('scripts/g2_thermal.py')
    dump('protocol.json', {
        'historical_import': True, 'id': 'g2-w4a16-night-20260911-r4',
        'host': 'Apple M5 Pro, 48 GiB, macOS 27; recorded workspace report',
        'workload': 'Gemma 4 12B first-layer full MLP; H3840/I15360; same-source Q4_0; FP16 I/O',
        'engines': {'C': 'native Swift Core AI, requested neuralEngine', 'G': 'MLX GPU'},
        'toolchain': {
            'macos_sw_vers': system['os']['stdout'],
            'xcode': xcode['xcode'], 'macosx_sdk': xcode['macosx_sdk'],
            'python_environments': [{'venv': e['venv'], 'python': '.'.join(map(str, e['version'])),
                                     'packages': {k: v for k, v in e['packages'].items() if v is not None}}
                                    for e in packages['environments']],
            'gpu_host_environment': 'performance/.venv: scripts/g2_client.py (frozen source hash) launches scripts/g2_gpu.py with it',
            'recorded_by': 'Same-day preflight package verification and Xcode recheck; OS build from the r3 attempt. The r4 run itself recorded no versions.',
            'post_run_report': {'date': '2026-09-11',
                                'source': 'Statement recorded after G2 completion; not contemporaneous telemetry.',
                                'text': 'The software environment matched preflight; no software changed that day.'},
        },
        'unit': 'MLP positions/s, not tokens/s or complete-model context',
        'origin_monotonic_ns': run['phases'][0]['start_ns'], 'phases': run['phases'],
        'groups': manifest['groups'], 'criteria': criteria,
        'source_hashes': manifest['source_hashes'], 'binary_hashes': manifest['binary_hashes'],
        'asset_and_reference_identity': prepared,
        'imported_audit': {'checks': len(audit['checks']), 'passed': True,
                           'scope': 'Original closed-workspace audit; not rerun by the portable verifier.'},
        'closure': {'passed': True, 'lifecycle': closure['lifecycle']},
        'reproduction': {'portable': 'Selected scalar records, timing and sensor recomputation',
                         'full_raw_audit': 'Requires closed workspace logs and assets',
                         'device_rerun': 'G2 orchestrator and model preparation not packaged in this CLI'},
        'prior_attempts': 'r4 reuses the original completed 20-minute baseline only; earlier device results are not pooled.',
        'thermal_platform': 'Last five minutes: both temperature OLS slopes <=0.2 C/min in magnitude, completion range/mean <=5%, no adjacent thermal upgrade; coverage checked.',
    })
    products = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(output.iterdir())}
    dump('provenance.json', {'historical_import': True, 'schema_version': 1,
         'sources': sources, 'products': products,
         'extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'selection': 'All r4 P2 control/warmup/measured records; every P1/P3/SAT arrival or completion with selected scalar fields; all audited r4 temperature, fan, thermal-state and resource samples; P0 natural checkpoints. No weights, activations, system process lists or original power byte stream.',
         'transformations': 'Whitespace compacted in deterministic gzip JSONL; absolute workspace prefixes removed from metadata paths; post-run statements reduced to their display and software-change content; background process rankings omitted. Numeric times and values unchanged; no averaging, downsampling, interpolation or outlier removal.',
         'limits': 'Hashes identify unavailable original files; they do not allow a reader without those files to rerun the original 87-check device audit.'})
    print(json.dumps({'output': str(output), 'files': len(products) + 1,
                      'bytes': sum(v['bytes'] for v in products.values()), 'sources': len(sources)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.workspace.resolve(), args.output.resolve())

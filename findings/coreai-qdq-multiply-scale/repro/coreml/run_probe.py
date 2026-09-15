"""Run each exported Core ML QDQ probe on CPU_AND_NE and CPU_ONLY with a target-PID log; one host per run."""
import argparse, datetime, json, math, os, struct, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOST = HERE / 'bin/probe-host'
PHASES = ['original', 'zero', 'negative', 'repeat', 'benchmark', 'benchmark_repeat']
UNITS = ['cpuAndNeuralEngine', 'cpuOnly']


def qdq(x, scale):
    code = max(-128, min(127, math.floor(abs(x) / scale + 0.5) * (1 if x >= 0 else -1)))
    return code * scale


def expectations(s_out, s_in=1 / 16):
    # (a, b) per phase; reference follows the declared graph, substitution dequantizes b's code with the output scale.
    pairs = {'original': (1, 1), 'zero': (0, 0), 'negative': (-1, -1), 'benchmark': (1, .5)}
    reference = {k: qdq(a * qdq(b, s_in), s_out) for k, (a, b) in pairs.items()}
    substitution = {k: qdq(a * round(b / s_in) * s_out, s_out) for k, (a, b) in pairs.items()}
    return reference, substitution


def values(path):
    return sorted(set(v[0] for v in struct.iter_unpack('<e', path.read_bytes())))


def ane_requests(log, pid, anchor, controls):
    counts = [0] * len(controls)
    failures = []
    for line in log.read_text(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        message = row.get('eventMessage', '')
        if row.get('processID') != pid:
            continue
        if 'Falling back' in message or 'ANE compilation failed' in message or 'Failed with status' in message:
            failures.append(message)
        if 'ANEProgramProcessRequestDirect() status=0x0' not in message:
            continue
        epoch = int(datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S.%f%z').timestamp() * 1e9)
        ns = epoch - anchor['epoch_ns'] + anchor['monotonic_ns']
        for i, c in enumerate(controls):
            if c['start_ns'] - 100_000 <= ns <= c['end_ns'] + 100_000:
                counts[i] += 1
    return counts, failures


def run_one(export, arm, record, units, out, shape, output_shape):
    out.mkdir(parents=True, exist_ok=False)
    model = Path(record['model_path'])
    config = {'runtime': 'coreml', 'model_path': str(model), 'compiled_path': str(model.parent / 'compiled.mlmodelc'),
              'entrypoint': 'main', 'input_name': 'x', 'output_name': record['output_name'],
              'input_path': str(export / 'benchmark.raw'), 'control_path': str(export / 'control.raw'),
              'shape': shape, 'output_shape': output_shape, 'seconds': 0, 'warm_seconds': 0,
              'quiet_seconds': 0, 'compute_units': units}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    child = logger = None
    report = {'arm': arm, 'compute_units': units}
    try:
        with (out / 'stdout.log').open('w') as so, (out / 'stderr.log').open('w') as se, (out / 'unified.ndjson').open('w') as logs:
            child = subprocess.Popen([str(HOST), str(out / 'config.json'), str(out / 'host')], stdout=so, stderr=se,
                                     env=dict(os.environ, OS_ACTIVITY_DT_MODE='YES'), start_new_session=True)
            deadline = time.monotonic() + 60
            while not (out / 'host/ready.json').exists():
                assert child.poll() is None and time.monotonic() < deadline, 'host_ready'
                time.sleep(.05)
            anchor = {'epoch_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
            logger = subprocess.Popen(['/usr/bin/log', 'stream', '--predicate', f'processIdentifier == {child.pid}',
                                       '--level', 'debug', '--style', 'ndjson'], stdout=logs, stderr=subprocess.STDOUT)
            time.sleep(1)
            assert logger.poll() is None, 'log_capture_started'
            (out / 'host/go').touch()
            report['host_returncode'] = child.wait(timeout=600)
            time.sleep(1)
    finally:
        if logger and logger.poll() is None:
            logger.terminate()
            logger.wait()
        if child and child.poll() is None:
            child.kill()
            child.wait()
    assert report['host_returncode'] == 0, (arm, units, (out / 'stderr.log').read_text())
    host = out / 'host'
    controls = json.loads((host / 'controls.json').read_text())
    assert [c['phase'] for c in controls] == PHASES
    counts, failures = ane_requests(out / 'unified.ndjson', child.pid, anchor, controls)
    reference, substitution = expectations(record['output_scale'])
    observed = {p: values(host / f'{p}.raw') for p in PHASES}
    plan = json.loads((host / 'plan.json').read_text())
    report.update(
        ane_requests_per_control=counts, ane_failures=failures,
        preferred={f"{p['operator']}:{p['outputs'][0]}": p['preferred'].split(':')[0].lstrip('<') for p in plan if p['operator'] != 'const'},
        observed={p: observed[p] for p in PHASES},
        reference=reference, substitution=substitution,
        repeat_exact=(host / 'original.raw').read_bytes() == (host / 'repeat.raw').read_bytes()
        and (host / 'benchmark.raw').read_bytes() == (host / 'benchmark_repeat.raw').read_bytes(),
        matches_reference=all(observed[p] == [reference[p]] for p in reference),
        matches_substitution=all(observed[p] == [substitution[p]] for p in substitution))
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    export, out = args.export.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((export / 'EXPORT.json').read_text())
    results = []
    for arm, record in manifest['arms'].items():
        assert record['audit']['passed'], arm
        for units in UNITS:
            r = run_one(export, arm, record, units, out / f'{arm}-{units}', manifest.get('shape', [1, 32, 1, 64]),
                        manifest.get('output_shape', [1, 16, 1, 64]))
            results.append(r)
            print(arm, units, 'ANE', r['ane_requests_per_control'], 'original', r['observed']['original'],
                  'benchmark', r['observed']['benchmark'], 'ref' if r['matches_reference'] else '',
                  'subst' if r['matches_substitution'] else '', flush=True)
    (out / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')

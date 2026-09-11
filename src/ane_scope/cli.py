"""The reproducibility CLI.

Every generated file goes into a new run directory, one suite runs at a time
behind a host-wide lock, and each stage — prepare, export, execute, verify,
report — is a separate child process with its own recorded command and log.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from .common import PACKAGE, ROOT, dump, environment, read, sha, source_identity
from .exporting import CASES, CHAINS, GROUPS, MULTIPLY, SPLIT, describe

PROCESSES_PER_ARM = 3
WARMUP = 10
SAMPLES_PER_PROCESS = 30
JOB_SECONDS = 300
ADDRESS = re.compile(r': 0x[0-9a-fA-F]+')


def worker(args):
    """The child-process entry point for one stage of a suite."""
    if args.action == 'prepare':
        from .references.prepare import prepare_data
        prepare_data(Path(args.destination), args.profile)
    elif args.action == 'export':
        from .exporting import export_case
        export_case(args.case, Path(args.data), Path(args.destination))
    elif args.action == 'execute':
        from .controller import execute
        execute(args.asset, args.data, args.host, args.destination, args.bench)
    elif args.action == 'verify':
        verify(Path(args.destination))


@contextlib.contextmanager
def serial_lock():
    """Host-wide lock for this tool, shared across independent checkouts."""
    with open(f'/tmp/ane-scope-{os.getuid()}.lock', 'a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another ANE Scope run is active')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def suite_cases(suite):
    """The depth-2 control cases each suite checks before it measures anything."""
    shallow = [c for c in CHAINS if c.endswith('-2')]
    if suite == 'smoke':
        return shallow + [SPLIT] + GROUPS + MULTIPLY
    if suite == 'compatibility':
        return GROUPS + MULTIPLY
    if suite == 'throughput':
        return shallow
    return ['coreai-a8w4-2', SPLIT]


def run(args):
    from .guard import run_child

    if sys.platform != 'darwin':
        raise RuntimeError('device suites require macOS; reference tests are portable')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + args.minutes * 60
    experiments_end = deadline - 60

    shutil.copytree(PACKAGE, out / 'source-snapshot' / 'ane_scope',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml', 'uv.lock'):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, out / 'source-snapshot' / name)
    dump(out / 'manifest.json', {
        'schema_version': 1, 'fresh_run': True, 'suite': args.suite,
        'started_epoch': time.time(), 'source_identity': source_identity(),
        'environment': environment(),
        'protocol': {
            'warmup': WARMUP, 'samples_per_process': SAMPLES_PER_PROCESS,
            'processes_per_arm': PROCESSES_PER_ARM,
            'timing': 'synchronous prediction only; FP16 copying, hashing, persistence '
                      'and debug capture outside timer',
            'MAC_counts_as_ops': 2, 'physical_INT8_proven': False,
            'fixture': 'synthetic Hadamard/sign/permutation; 16 distinct spatial vectors '
                       'repeated 256 times',
            'time_limit_minutes': args.minutes,
        },
        'state': 'running',
    })
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false')

    failures, runs, assets, hosts = [], [], set(), set()
    counter = 0

    def job(name, command, seconds=JOB_SECONDS):
        return run_child(command, out / 'jobs' / name, result_root=out, seconds=seconds,
                         deadline=experiments_end)

    def py(name, *arguments):
        return job(name, [sys.executable, '-m', 'ane_scope', '_worker', *arguments])

    def execute(case, bench, label):
        nonlocal counter
        if time.time() >= experiments_end:
            raise TimeoutError('suite cutoff reached')
        if case not in assets:
            py('export-' + case, 'export', '--case', case, '--data', str(out / 'data'),
               '--destination', str(out / 'assets' / case))
            assets.add(case)
        runtime = describe(case)['runtime']
        host = out / 'bin' / runtime
        if runtime not in hosts:
            host.parent.mkdir(exist_ok=True)
            job('compile-' + runtime,
                ['xcrun', 'swiftc', '-O', '-parse-as-library',
                 '-module-cache-path', str(out / 'module-cache'),
                 str(PACKAGE / 'native' / f'{runtime}.swift'), '-o', str(host)])
            hosts.add(runtime)
        counter += 1
        name = f'{counter:02d}-{label}-{case}'
        arguments = ['execute', '--asset', str(out / 'assets' / case),
                     '--data', str(out / 'data'), '--host', str(host),
                     '--destination', str(out / 'runs' / name)]
        if bench:
            arguments += ['--bench']
        py(name, *arguments)
        runs.append(name)
        result = read(out / 'runs' / name / 'run.json')
        print(json.dumps({'run': name, 'numerical': result['numerical_passed'],
                          'ANE_controls': result['ANE_controls'],
                          'benchmark': result['benchmark_status'],
                          'p50_ms': result.get('p50_ms')}), flush=True)
        return result

    with serial_lock():
        try:
            py('prepare', 'prepare', '--profile',
               'full' if args.suite == 'throughput' else 'smoke',
               '--destination', str(out / 'data'))
            cases = suite_cases(args.suite)
            allowed = {}
            for case in cases:
                try:
                    allowed[case] = execute(case, False, 'control')['admitted']
                except (RuntimeError, TimeoutError) as error:
                    failures.append({'case': case, 'error': str(error)})
                    # A resource or timeout failure stops the suite; an ordinary
                    # export error is isolated to its own case.
                    commands = list((out / 'jobs').glob('*/command.json'))
                    if 'error' in read(max(commands, key=lambda p: p.stat().st_mtime)):
                        raise
            if args.suite in ('throughput', 'split'):
                perf = [c[:-1] + '128' for c in cases] if args.suite == 'throughput' else cases
                for repeat in range(PROCESSES_PER_ARM):
                    # Forward, reverse, forward, so arm order cannot drive the result.
                    for case in perf if repeat % 2 == 0 else reversed(perf):
                        base = (case.rsplit('-', 1)[0] + '-2'
                                if args.suite == 'throughput' else case)
                        if not allowed.get(base):
                            continue
                        if not execute(case, True, f'process{repeat + 1}')['admitted']:
                            allowed[base] = False
        except BaseException as error:
            failures.append({'suite_error': repr(error)})
        finally:
            manifest = read(out / 'manifest.json')
            manifest.update(finished_epoch=time.time(),
                            state='failed' if failures else 'completed',
                            failures=failures, runs=runs)
            dump(out / 'manifest.json', manifest)

    if (out / 'data/manifest.json').exists():
        run_child([sys.executable, '-m', 'ane_scope', '_worker', 'verify',
                   '--destination', str(out)],
                  out / 'jobs' / 'verify', result_root=out, seconds=60, deadline=deadline)
        report(out)
    if failures:
        raise RuntimeError('suite incomplete; see manifest.json and jobs logs')


def verify(out):
    """Recheck stored evidence, and report whether the current source produced it."""
    import numpy as np

    from .common import file_hashes
    from .evidence import placement, verify_outputs

    out = Path(out)
    manifest = read(out / 'manifest.json')
    data = out / 'data'
    checks = []
    source_matches = manifest['source_identity'] == source_identity()
    for name, digest in read(data / 'manifest.json')['files'].items():
        assert sha(data / name) == digest

    for name in manifest['runs']:
        folder = out / 'runs' / name
        record = read(folder / 'run.json')
        asset = out / 'assets' / record['case_id']
        meta = read(asset / 'export.json')
        audit = read(asset / 'asset-audit.json')
        assert sha(asset / 'export.json') == record['export_sha256']
        assert sha(asset / 'asset-audit.json') == record['asset_audit_sha256']
        assert file_hashes(Path(meta['model_path'])) == audit['asset_files']
        assert sha(out / 'bin' / meta['runtime']) == record['host_sha256']

        calls = read(folder / 'controls.json')['records']
        numeric = verify_outputs(folder, meta, data, calls)
        device = placement(folder, record['pid'], calls, meta['expected_conv_count'],
                           meta['runtime'])
        assert numeric == read(folder / 'numerical-control.json')
        assert device == read(folder / 'placement.json')

        if record['benchmark_status'] == 'measured':
            timed = read(folder / 'result.json')['records']
            final = verify_outputs(folder, meta, data, timed)
            assert final == read(folder / 'numerical-final.json') and final['passed']
            times = [x['duration_ns'] / 1e6 for x in timed if x['phase'] == 'measured']
            assert len(times) == SAMPLES_PER_PROCESS
            assert float(np.median(times)) == record['p50_ms']
            assert float(np.percentile(times, 95)) == record['p95_ms']
            assert all(x['output_sha256'] == calls[0]['output_sha256'] for x in timed
                       if x['phase'] in ('warmup', 'measured'))
        checks.append({'run': name, 'evidence_consistent': True,
                       'numeric_passed': numeric['passed'],
                       'ANE_controls': device['every_control_has_ANE']})

    dump(out / 'verification.json', {
        'passed': True,
        'meaning': 'saved evidence rechecked; findings are not required to be numerically '
                   'correct or execute on ANE',
        'checked_runs': checks,
        'source_matches_run': source_matches,
        'verifier_source_identity': source_identity(),
    })


def report(out):
    """Write the human-readable report and the compact public selection."""
    out = Path(out)
    manifest = read(out / 'manifest.json')
    rows, groups = [], {}
    for name in manifest['runs']:
        record = read(out / 'runs' / name / 'run.json')
        rows.append(f"| {name} | {record['numerical_passed']} | {record['ANE_controls']} "
                    f"| {record['benchmark_status']} | {record.get('p50_ms', '—')} |")
        if record['benchmark_status'] == 'measured':
            groups.setdefault(record['case_id'], []).append(record)

    verification = (read(out / 'verification.json')['passed']
                    if (out / 'verification.json').exists() else 'not run')
    lines = ['# Local experiment report', '',
             f"State: {manifest['state']}. Evidence verification: {verification}.", '',
             '| Process | Numeric | ANE controls | Timing | p50 ms |',
             '|---|---|---|---|---|', *rows, '',
             '## Independent-process summaries', '',
             '| Case | Processes | p50 range ms | Source-equivalent T ops/s range |',
             '|---|---:|---:|---:|']
    for case, records in groups.items():
        p50 = [r['p50_ms'] for r in records]
        tops = [r['source_equivalent_Tops'] for r in records]
        lines.append(f'| {case} | {len(records)} | {min(p50):.6f}–{max(p50):.6f} '
                     f'| {min(tops):.3f}–{max(tops):.3f} |')
    lines += ['', 'MAC = 2 source operations. These are controlled convolution workloads, not '
                  'LLM tokens/s or proof of physical INT8 instructions. Separate calls within '
                  'one process are correlated. Core AI has no per-operation placement proof in '
                  'this host. Energy was not measured.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))

    published = []
    for name in manifest['runs']:
        folder = out / 'runs' / name
        record = read(folder / 'run.json')
        meta = read(out / 'assets' / record['case_id'] / 'export.json')
        timed = folder / 'result.json'
        calls = read(timed)['records'] if timed.exists() else read(folder / 'controls.json')['records']
        measured = [{k: c[k] for k in ('index', 'duration_ns', 'start_epoch_ns',
                                       'end_epoch_ns', 'output_sha256')}
                    for c in calls if c['phase'] == 'measured']
        device = read(folder / 'placement.json')
        # Device descriptions carry unstable object addresses; keep the class only.
        device['conv_plan'] = [
            {**x, 'preferred': ADDRESS.sub('', x.get('preferred') or ''),
             'supported': [ADDRESS.sub('', v) for v in x.get('supported', [])]}
            for x in device['conv_plan']]
        row = {
            'run': name, 'case_id': record['case_id'], 'source_ops': meta['source_ops'],
            'shape': meta['shape'], 'output_shape': meta['output_shape'],
            'representation': meta['representation'], 'pid': record['pid'],
            'benchmark_status': record['benchmark_status'], 'p50_ms': record.get('p50_ms'),
            'p95_ms': record.get('p95_ms'),
            'source_equivalent_Tops': record.get('source_equivalent_Tops'),
            'source_records_sha256': sha(timed if timed.exists() else folder / 'controls.json'),
            'asset_audit': read(out / 'assets' / record['case_id'] / 'asset-audit.json'),
            'numerical': read(folder / 'numerical-control.json'), 'placement': device,
            'admission': read(folder / 'admission.json'), 'measured_records': measured,
        }
        for optional in ('flattened-scale-hypothesis.json', 'RNE-not-gate.json'):
            if (folder / optional).exists():
                row[optional[:-5]] = read(folder / optional)
        published.append(row)

    dump(out / 'public-results.json', {
        'schema_version': 1, 'fresh_run': True, 'historical_import': False,
        'suite': manifest['suite'], 'environment': manifest['environment'],
        'protocol': manifest['protocol'], 'source_identity': manifest['source_identity'],
        'verification': (read(out / 'verification.json')
                         if (out / 'verification.json').exists() else None),
        'runs': published,
    })
    return out / 'REPORT.md'


def build_parser():
    parser = argparse.ArgumentParser(prog='ane-scope')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Print the observed environment as JSON')
    sub.add_parser('list', help='List every available case')

    runner = sub.add_parser('run', help='Run one suite into a new output directory')
    runner.add_argument('--suite', choices=['smoke', 'throughput', 'split', 'compatibility'],
                        default='smoke')
    runner.add_argument('--output', required=True)
    runner.add_argument('--minutes', type=int, choices=range(3, 61), default=50)
    for name in ('verify', 'report'):
        sub.add_parser(name).add_argument('path')

    child = sub.add_parser('_worker', help=argparse.SUPPRESS)
    child.add_argument('action', choices=['prepare', 'export', 'execute', 'verify'])
    child.add_argument('--destination', required=True)
    child.add_argument('--profile', default='smoke')
    for name in ('--case', '--data', '--asset', '--host'):
        child.add_argument(name)
    child.add_argument('--bench', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    try:
        if args.command == 'doctor':
            print(json.dumps(environment(), indent=2))
        elif args.command == 'list':
            print('\n'.join(CASES))
        elif args.command == 'run':
            run(args)
        elif args.command == 'verify':
            from .guard import run_child
            out = Path(args.path).resolve()
            with serial_lock():
                run_child([sys.executable, '-m', 'ane_scope', '_worker', 'verify',
                           '--destination', str(out)],
                          out / 'jobs' / f'verify-{time.time_ns()}', result_root=out,
                          seconds=120)
        elif args.command == 'report':
            print(report(Path(args.path)))
        else:
            worker(args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()

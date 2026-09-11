"""One Swift host: four controls, an evidence gate, then optional timed predictions.

Nothing is timed until the numerical controls, the device evidence and the
runtime or plan check have all passed.  A case that fails any of them is
recorded as an observation with no benchmark number.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .common import dump, file_hashes, read, sha
from .evidence import placement, verify_outputs

WARMUP = 10
REPEATS = 30
FOOTPRINT_LIMIT_BYTES = 32 * 1024 ** 2
CONTROL_TIMEOUT_SECONDS = 220
READY_TIMEOUT_SECONDS = 30


def execute(asset, data, host, out, bench):
    """Run one exported case through its host and record every layer of evidence."""
    asset, data, out = Path(asset), Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = read(asset / 'export.json')
    audit = read(asset / 'asset-audit.json')
    assert audit['passed'] and file_hashes(Path(meta['model_path'])) == audit['asset_files']
    for name, digest in read(data / 'manifest.json')['files'].items():
        assert sha(data / name) == digest

    config = {k: meta[k] for k in ('model_path', 'shape', 'output_shape', 'input_name',
                                   'output_name', 'entrypoint')}
    config.update(input_path=str(data / meta['input_file']), warmup=WARMUP, repeats=REPEATS,
                  controls_only=not bench)
    dump(out / 'config.json', config)

    record = {'case_id': meta['case_id'], 'status': 'started', 'bench_requested': bool(bench),
              'host_sha256': sha(host),
              'asset_audit_sha256': sha(asset / 'asset-audit.json'),
              'export_sha256': sha(asset / 'export.json'),
              'input_sha256': sha(data / meta['input_file']),
              'started_epoch': time.time()}
    child = capture = None

    def wait_file(name, seconds):
        end = time.monotonic() + seconds
        while not (out / name).exists():
            if child.poll() is not None:
                raise RuntimeError(f'host exited {child.returncode} before {name}')
            if time.monotonic() > end:
                raise TimeoutError(name)
            time.sleep(.05)

    def stop(process):
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    try:
        with (out / 'host.log').open('x') as log, (out / 'unified.ndjson').open('x') as events:
            child = subprocess.Popen([str(host), str(out / 'config.json'), str(out)],
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, OS_ACTIVITY_DT_MODE='YES'))
            record['pid'] = child.pid
            dump(out / 'run.json', record)
            wait_file('ready.json', READY_TIMEOUT_SECONDS)
            assert read(out / 'ready.json')['pid'] == child.pid

            capture = subprocess.Popen(
                ['/usr/bin/log', 'stream', '--predicate',
                 f'processIdentifier == {child.pid}', '--level', 'debug', '--style', 'ndjson'],
                stdout=events, stderr=subprocess.STDOUT)
            time.sleep(.7)
            if capture.poll() is not None:
                raise RuntimeError('unified log collector unavailable')

            (out / 'go').touch()
            wait_file('controls.json', CONTROL_TIMEOUT_SECONDS)
            calls = read(out / 'controls.json')['records']
            numeric = verify_outputs(out, meta, data, calls)
            dump(out / 'numerical-control.json', numeric)

            # Log capture stops before any timing, so it never sits inside the timer.
            time.sleep(.5)
            stop(capture)
            capture = None
            events.flush()

            device = placement(out, child.pid, calls, meta['expected_conv_count'], meta['runtime'])
            dump(out / 'placement.json', device)
            runtime = read(out / 'runtime.json') if meta['runtime'] == 'coreai' else None
            if runtime:
                runtime_ok = (runtime['preferred_compute_unit'] == 'neuralEngine'
                              and runtime['entrypoint'] == meta['entrypoint']
                              and runtime['function_names'] == [meta['entrypoint']])
            else:
                runtime_ok = device['conv_plan_passed']
            admitted = bool(numeric['passed'] and device['every_control_has_ANE'] and runtime_ok
                            and not device['compiler_or_fallback_messages'])
            dump(out / 'admission.json', {
                'numerical_passed': numeric['passed'],
                'ANE_control_participation': device['every_control_has_ANE'],
                'runtime_or_plan_passed': bool(runtime_ok),
                'admitted': admitted,
                'benchmark_admitted': bool(bench and admitted),
                'coreai_per_operation_mapping': 'unverified' if runtime else None,
            })

            if bench and not admitted:
                stop(child)
                record.update(status='observed', benchmark_status='not_admitted', measurements=0)
            else:
                if bench:
                    (out / 'bench-go').touch()
                child.wait(timeout=CONTROL_TIMEOUT_SECONDS)
                if child.returncode:
                    raise RuntimeError(f'host failed {child.returncode}')
                result = read(out / 'result.json')
                final = verify_outputs(out, meta, data, result['records'])
                dump(out / 'numerical-final.json', final)
                measured = [r for r in result['records'] if r['phase'] == 'measured']
                original = calls[0]['output_sha256']
                hash_ok = all(r['output_sha256'] == original for r in result['records']
                              if r['phase'] in ('warmup', 'measured'))
                if bench and (not final['passed'] or not hash_ok):
                    raise RuntimeError('timed output correctness or repeatability failed')
                record.update(status='observed',
                              benchmark_status='measured' if bench else 'not_requested',
                              measurements=len(measured), timed_output_hashes_equal=hash_ok)
                if bench:
                    record.update(_timing(result, measured, meta))
                    if not record['memory_gate_passed']:
                        raise RuntimeError('post-warmup footprint gate failed')

            # Diagnostic comparisons kept beside, never inside, the admission gate.
            if '-group-' in meta['case_id']:
                alt = dict(meta, reference_file='reference-group-wrong.npy')
                dump(out / 'flattened-scale-hypothesis.json',
                     verify_outputs(out, alt, data, calls))
            if '-w8a8-' in meta['case_id'] or ('-a8w4-' in meta['case_id']
                                               and 'split32' not in meta['case_id']):
                alt = dict(meta, reference_file=meta['reference_file'].replace('.npy', '-rne.npy'))
                dump(out / 'RNE-not-gate.json', verify_outputs(out, alt, data, calls))

            record.update(numerical_passed=numeric['passed'],
                          ANE_controls=device['every_control_has_ANE'], admitted=admitted)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        stop(capture)
        stop(child)
        record.update(finished_epoch=time.time(),
                      child_returncode=child.returncode if child else None)
        dump(out / 'run.json', record)
    return record


def _timing(result, measured, meta):
    """Summarise the measured calls and check the post-warmup footprint gate."""
    import numpy as np

    times = [r['duration_ns'] / 1e6 for r in measured]
    assert len(times) == REPEATS
    snapshot = {x['point']: x['physical_footprint'] for x in result['snapshots']}
    warm, after = snapshot.get('warm'), snapshot.get('measured')
    growth = after - warm if after is not None and warm is not None else None
    p50 = float(np.median(times))
    return {
        'p50_ms': p50,
        'p95_ms': float(np.percentile(times, 95)),
        'source_ops': meta['source_ops'],
        'footprint_growth_bytes': growth,
        'memory_gate_passed': growth is not None and growth <= FOOTPRINT_LIMIT_BYTES,
        'source_equivalent_Tops': meta['source_ops'] / p50 / 1e9,
    }

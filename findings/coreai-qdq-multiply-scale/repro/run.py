"""Run the exported fixture with native Core AI and retain original/zero/repeat."""
import argparse, json, os, struct, subprocess, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--host', type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    here = Path(__file__).resolve().parent
    assert (out / 'config.json').is_file(), 'export_config_missing'
    host = a.host.resolve() if a.host else out / 'host'
    if not a.host:
        subprocess.run(['xcrun', 'swiftc', '-parse-as-library', '-O', '-target', 'arm64-apple-macos27.0',
                        '-module-cache-path', str(out / 'module-cache'), str(here / 'host.swift'),
                        '-o', str(host)], check=True)
    target = out / 'host-output'
    assert not target.exists(), 'fresh_output_directory'
    child = None
    log = None
    with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr, (out / 'unified.ndjson').open('w') as events:
        try:
            child = subprocess.Popen([str(host), str(out / 'config.json'), str(target)], stdout=stdout, stderr=stderr,
                                     env=dict(os.environ, OS_ACTIVITY_DT_MODE='YES'))
            deadline = time.monotonic() + 60
            while not (target / 'ready.json').exists():
                assert child.poll() is None and time.monotonic() < deadline, 'host_ready'
                time.sleep(.05)
            log = subprocess.Popen(['/usr/bin/log', 'stream', '--predicate', f'processIdentifier == {child.pid}',
                                    '--level', 'debug', '--style', 'ndjson'], stdout=events, stderr=subprocess.STDOUT)
            time.sleep(1)
            assert log.poll() is None
            (target / 'go').touch()
            assert child.wait(timeout=120) == 0, 'host_exit'
            time.sleep(1)
        finally:
            if child and child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
            if log:
                log.terminate()
                log.wait(timeout=10)
            (out / 'command.json').write_text(json.dumps({'pid': child.pid if child else None,
                                                          'log_pid': log.pid if log else None,
                                                          'returncode': child.returncode if child else None}, indent=2) + '\n')
    cfg = json.loads((out / 'config.json').read_text())
    result = {}
    for arm in cfg['order']:
        first = (target / f'{arm}-control-0.raw').read_bytes()
        assert first == (target / f'{arm}-control-2.raw').read_bytes(), 'repeat'
        zero = [v[0] for v in struct.iter_unpack('<e', (target / f'{arm}-control-1.raw').read_bytes())]
        assert all(v == 0 for v in zero), 'zero'
        values = sorted(set(v[0] for v in struct.iter_unpack('<e', first)))
        result[arm] = {'expected': 1.0, 'actual_unique_values': values, 'repeat_exact': True, 'zero_exact': True}
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

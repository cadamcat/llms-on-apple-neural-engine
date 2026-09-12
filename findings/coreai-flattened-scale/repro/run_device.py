"""Explicitly export and run just the native K64 and split K32 Core AI probes."""
from pathlib import Path
import argparse
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for assets, compilation cache, logs and outputs.')
    args = parser.parse_args()
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        parser.error('requires an Apple silicon Mac with the Core AI macOS 27 SDK/runtime')
    from verify import verify
    verify()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'frozen'))
    from ane_scope.common import environment
    from ane_scope.exporting import export_case
    (out / 'environment.json').write_text(json.dumps(environment(), indent=2) + '\n')
    host = out / 'coreai-host'
    command = ['xcrun', 'swiftc', '-O', '-parse-as-library', '-module-cache-path', str(out / 'module-cache'),
               str(ROOT / 'frozen/ane_scope/native/coreai.swift'), '-o', str(host)]
    with (out / 'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    for case in ['coreai-group-native64', 'coreai-group-split32']:
        directory = out / case
        directory.mkdir()
        info = export_case(case, ROOT / 'fixture', out / 'assets' / case)
        config = {key: info[key] for key in ['model_path', 'shape', 'output_shape', 'input_name', 'output_name', 'entrypoint']}
        config.update(input_path=str(ROOT / 'fixture/group_input.raw'), warmup=0, repeats=0, controls_only=True)
        path = directory / 'config.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        (directory / 'go').touch()
        with (directory / 'host.log').open('w') as log:
            subprocess.run([str(host), str(path), str(directory)], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = verify(out)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

"""Build a reproducible ZIP from the archived probe's explicit file inventory."""
from pathlib import Path
import argparse
import json
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'coreai-grouped-lut-repro'


def archive(output, root=ROOT):
    root, output = Path(root), Path(output)
    record = json.loads((root / 'recorded/results.json').read_text())
    fixture = json.loads((root / 'fixture/manifest.json').read_text())
    names = {'README.md', 'LICENSE', 'requirements-device.txt', 'verify.py',
             'run_device.py', 'archive.py', 'fixture/manifest.json', 'recorded/results.json'}
    names.update('fixture/' + name for name in fixture['files'])
    names.update('frozen/ane_scope/' + name for name in record['bundled_source_files'])
    for case in record['runs']:
        prefix = 'recorded/' + case['case_id'] + '/'
        names.add(prefix + 'persisted-graph.mlir')
        names.update(prefix + row['output_file'] for row in case['numerical']['comparisons'])
    for name in names:
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('archive.invalid_source: ' + name)
        if path.resolve() == output.resolve():
            raise ValueError('archive.output_overwrites_source: ' + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(names):
            info = zipfile.ZipInfo(PREFIX + '/' + name, (2026, 9, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, (root / name).read_bytes())
    return len(names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = archive(args.output)
    print(f'{args.output}: {count} files')

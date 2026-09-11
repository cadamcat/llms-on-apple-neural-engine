"""Compare today's source bytes with the identities stored in the selected runs.

This is an identity check, not a device rerun or a semantic equivalence proof.
Old verification fields remain historical records of their original checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/validation/source-identity.json'


def build(root):
    suites = {}
    for name in ('smoke', 'throughput', 'split'):
        rel = 'results/fresh/' + name + '.json'
        raw = (root / rel).read_bytes()
        data = json.loads(raw)
        rows = []
        for key, expected in data['source_identity'].items():
            path = (root / 'src/ane_scope' / key).resolve()
            normalized = str(path.relative_to(root.resolve()))
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            rows.append({'path': normalized, 'run_sha256': expected,
                         'current_sha256': actual, 'matches': actual == expected})
        suites[name] = {'record': rel, 'record_sha256': hashlib.sha256(raw).hexdigest(),
                        'current_source_matches_run': all(r['matches'] for r in rows),
                        'files': rows}
    return {'schema_version': 1, 'method': 'SHA256 of current source versus recorded run identity',
            'device_rerun': False, 'semantic_equivalence_tested': False, 'suites': suites}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write', action='store_true')
    modes.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = build(ROOT)
    text = json.dumps(data, indent=2) + '\n'
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
    elif not OUTPUT.exists() or OUTPUT.read_text() != text:
        raise SystemExit('Current identity report is stale; review the changes before --write.')
    for name, suite in data['suites'].items():
        changed = [r['path'] for r in suite['files'] if not r['matches']]
        print(f'{name}: current_source_matches_run={suite["current_source_matches_run"]}; '
              f'{len(changed)} source files differ')


if __name__ == '__main__':
    main()

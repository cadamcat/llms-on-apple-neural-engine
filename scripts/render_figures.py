"""Generate the documentation SVGs, or check the identity of the published ones.

Both modes use only the Python standard library, so regeneration is a plain
text transformation with no plotting stack, no network access and no device
work.  ``--write`` is deterministic: the same repository state always produces
byte-identical files, which is what lets CI regenerate and compare.
"""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
GENERATOR_FILES = ['scripts/render_figures.py', 'scripts/figures/canvas.py',
                   'scripts/figures/architecture.py', 'scripts/figures/arithmetic.py',
                   'scripts/figures/measurements.py', 'scripts/figures/g2.py',
                   'scripts/g2/evidence.py', 'scripts/g2/thermal.py',
                   'scripts/figures/g3.py', 'scripts/g3/evidence.py',
                   'scripts/g3/protocol.py', 'scripts/g3/power.py',
                   'scripts/figures/g4a.py', 'scripts/g4a/evidence.py', 'scripts/g4a/protocol.py',
                   'scripts/g4a/power.py', 'scripts/g4a/disk.py']
SVG = '{http://www.w3.org/2000/svg}'
FORBIDDEN_TAGS = ('script', 'foreignObject', 'image', 'use')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_svg(path: Path) -> None:
    """Reject anything active, external or inaccessible in a published figure."""
    text = path.read_text(encoding='utf-8')
    root = ET.fromstring(text)
    assert root.tag == SVG + 'svg' and root.get('viewBox'), path.name
    assert root.get('role') == 'img', path.name
    assert root.find(SVG + 'title') is not None and root.find(SVG + 'desc') is not None, path.name
    ids = {e.get('id') for e in root.iter() if e.get('id')}
    assert set(root.get('aria-labelledby', '').split()) <= ids, path.name
    for element in root.iter():
        tag = element.tag.rsplit('}', 1)[-1]
        assert tag not in FORBIDDEN_TAGS, f'{path.name}: <{tag}>'
        for key, value in element.attrib.items():
            assert not key.lower().startswith('on'), path.name
            if key.rsplit('}', 1)[-1] == 'href':
                assert value.startswith('#'), path.name
    for style in root.iter(SVG + 'style'):
        body = (style.text or '').lower()
        for token in ('@import', 'url(', 'expression('):
            assert token not in body, f'{path.name}: external or active style ({token})'


def write() -> int:
    sys.path.insert(0, str(ROOT / 'scripts'))
    from figures import architecture, arithmetic, measurements, g2, g3, g4a

    OUT.mkdir(parents=True, exist_ok=True)
    figures = []
    for module in (architecture, arithmetic, measurements, g2, g3, g4a):
        for item in module.generate(ROOT, OUT):
            item['source_hashes'] = {name: sha(ROOT / name) for name in item.get('sources', [])}
            item['svg_sha256'] = sha(OUT / item['filename'])
            inspect_svg(OUT / item['filename'])
            figures.append(item)
    for path in sorted(OUT.glob('*.svg')):
        if path.name not in {item['filename'] for item in figures}:
            raise SystemExit(f'Stale figure with no generator: {path.name}')
    manifest = {
        'schema_version': 2,
        'scope': 'Figures derived from the bundled evidence and the documented methods; '
                 'zero new device calls.',
        'generation': 'Python standard library only; deterministic and byte-reproducible.',
        'generator_hashes': {name: sha(ROOT / name) for name in GENERATOR_FILES},
        'figures': figures,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    return len(figures)


def check() -> int:
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema_version'] == 2, 'Unexpected manifest schema'
    for name, expected in manifest['generator_hashes'].items():
        assert sha(ROOT / name) == expected, f'Generator changed; review and regenerate: {name}'
    expected_files = {item['filename'] for item in manifest['figures']}
    assert {p.name for p in OUT.glob('*.svg')} == expected_files, 'SVG inventory changed'
    for item in manifest['figures']:
        for name, expected in item['source_hashes'].items():
            assert sha(ROOT / name) == expected, f'Figure source changed; regenerate: {name}'
        path = OUT / item['filename']
        assert sha(path) == item['svg_sha256'], f'Figure changed; regenerate: {path.name}'
        inspect_svg(path)
    return len(expected_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write', action='store_true',
                       help='Regenerate every SVG and its source manifest')
    modes.add_argument('--check', action='store_true',
                       help='Check the published SVG, source and generator identities')
    args = parser.parse_args()
    count = write() if args.write else check()
    verb = 'generated' if args.write else 'verified'
    print(f'{count} documentation SVGs {verb}; no device execution')


if __name__ == '__main__':
    main()

"""Generate the documentation SVGs, or check that the published ones regenerate unchanged.

Both modes use only the Python standard library, so regeneration is a plain
text transformation with no plotting stack, no network access and no device
work.  ``--write`` is deterministic: the same repository state always produces
byte-identical files, and ``--check`` regenerates into a temporary directory
and compares every byte with the published figures and manifest.
"""

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
SVG = '{http://www.w3.org/2000/svg}'
FORBIDDEN_TAGS = ('script', 'foreignObject', 'image', 'use')


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


def generate(out: Path) -> dict:
    sys.path.insert(0, str(ROOT / 'scripts'))
    from figures import architecture, arithmetic, measurements, g2, g3, g4a

    out.mkdir(parents=True, exist_ok=True)
    figures = []
    for module in (architecture, arithmetic, measurements, g2, g3, g4a):
        for item in module.generate(ROOT, out):
            inspect_svg(out / item['filename'])
            figures.append(item)
    return {
        'schema_version': 3,
        'scope': 'Figures derived from the bundled evidence and the documented methods; '
                 'zero new device calls.',
        'generation': 'Python standard library only; deterministic and byte-reproducible.',
        'figures': figures,
    }


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'


def write() -> int:
    manifest = generate(OUT)
    names = {item['filename'] for item in manifest['figures']}
    for path in sorted(OUT.glob('*.svg')):
        if path.name not in names:
            raise SystemExit(f'Stale figure with no generator: {path.name}')
    (OUT / 'manifest.json').write_text(manifest_text(manifest), encoding='utf-8')
    return len(names)


def check() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        fresh = Path(temporary)
        manifest = generate(fresh)
        names = {item['filename'] for item in manifest['figures']}
        assert {p.name for p in OUT.glob('*.svg')} == names, 'SVG inventory changed; regenerate'
        assert (OUT / 'manifest.json').read_text(encoding='utf-8') == manifest_text(manifest), \
            'Figure manifest differs from the evidence; regenerate'
        for name in sorted(names):
            assert (OUT / name).read_bytes() == (fresh / name).read_bytes(), f'Figure differs from the evidence; regenerate: {name}'
            inspect_svg(OUT / name)
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write', action='store_true',
                       help='Regenerate every SVG and its source manifest')
    modes.add_argument('--check', action='store_true',
                       help='Regenerate in a temporary directory and compare with the published figures')
    args = parser.parse_args()
    count = write() if args.write else check()
    verb = 'generated' if args.write else 'verified'
    print(f'{count} documentation SVGs {verb}; no device execution')


if __name__ == '__main__':
    main()

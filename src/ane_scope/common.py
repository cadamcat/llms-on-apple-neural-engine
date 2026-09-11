"""Path, identity and JSON helpers.

Nothing here imports a device framework, and importing this module performs no
filesystem work of its own.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]

# Recorded for every run so a result can name the stack that produced it.
TRACKED_PACKAGES = ['numpy', 'torch', 'coreai-core', 'coreai-torch', 'coremltools']
HOST_COMMANDS = [
    ('os', ['sw_vers']),
    ('xcode', ['xcodebuild', '-version']),
    ('sdk', ['xcrun', '--sdk', 'macosx', '--show-sdk-version']),
    ('chip', ['sysctl', '-n', 'machdep.cpu.brand_string']),
    ('memory', ['sysctl', '-n', 'hw.memsize']),
]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    """Write JSON atomically, refusing NaN so a record cannot become unreadable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(path):
    return {p.relative_to(path).as_posix(): sha(p)
            for p in sorted(Path(path).rglob('*')) if p.is_file()}


def environment():
    """Describe the interpreter, the tracked packages and, on macOS, the host."""
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        'python': sys.version.split()[0],
        'architecture': platform.machine(),
        'system': platform.system(),
        'packages': packages,
    }
    if sys.platform == 'darwin':
        for name, command in HOST_COMMANDS:
            probe = subprocess.run(command, text=True, capture_output=True, timeout=20)
            result[name] = {'returncode': probe.returncode,
                            'value': probe.stdout.strip(),
                            'error': probe.stderr.strip()[:1000]}
    return result


def source_identity():
    """Hash the working tree that is about to run, commit or no commit."""
    paths = list(PACKAGE.rglob('*.py')) + list((PACKAGE / 'native').glob('*.swift'))
    result = {p.relative_to(PACKAGE).as_posix(): sha(p) for p in sorted(paths)}
    for name in ['pyproject.toml', 'uv.lock']:
        if (ROOT / name).is_file():
            result['../../' + name] = sha(ROOT / name)
    return result

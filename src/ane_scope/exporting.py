"""Static, model-free experiment catalog and lazy backend dispatch.

Importing this module builds only the case list.  A backend is imported, and a
model is written, only when :func:`export_case` is actually called.
"""

from __future__ import annotations

from pathlib import Path

from .common import dump, read

CHAINS = [f'{r}-{f}-{d}'
          for d in (2, 128)
          for r, fs in [('coreml', ('fp16', 'w8a8')), ('coreai', ('fp16', 'w8a8', 'a8w4'))]
          for f in fs]
GROUPS = [f'{r}-group-{k}' for r in ('coreml', 'coreai') for k in ('native64', 'split32')]
MULTIPLY = [f'coreai-qdq-{k}' for k in ('equal', 'unequal', 'reverse_order', 'explicit_q')]
SPLIT = 'coreai-a8w4-split32-2'
CASES = CHAINS + [SPLIT] + GROUPS + MULTIPLY


def describe(case_id):
    """Return the frozen contract for one case: shapes, reference and gates."""
    if case_id not in CASES:
        raise ValueError(f'unknown case {case_id}')
    group = '-group-' in case_id
    multiply = '-qdq-' in case_id
    split = case_id == SPLIT
    depth = int(case_id.rsplit('-', 1)[1]) if not (group or multiply) else None

    if multiply:
        shape, output_shape = [2, 64, 1, 16], [1, 64, 1, 16]
        input_file, reference_file = 'mul_input.raw', 'reference-mul.npy'
    elif group:
        shape, output_shape = [1, 64, 1, 64], [1, 64, 1, 64]
        input_file, reference_file = 'group_input.raw', 'reference-group.npy'
    else:
        shape, output_shape = [1, 512, 64, 64], [1, 512, 64, 64]
        input_file = 'input.raw'
        reference_file = ('reference-a8w4-split32-2.npy' if split else
                          f'reference-{"fp16" if "-fp16-" in case_id else "w8a8"}-{depth}.npy')

    if multiply:
        expected_convs = 0
    elif group:
        expected_convs = 2 if case_id.endswith('split32') else 1
    else:
        expected_convs = 32 if split else depth

    return {
        'case_id': case_id,
        'runtime': case_id.split('-')[0],
        'shape': shape,
        'output_shape': output_shape,
        'input_file': input_file,
        'reference_file': reference_file,
        # The chain fixture repeats 16 distinct spatial vectors across 4096 positions.
        'reference_repeat_columns': 1 if group or multiply else 256,
        'negative_parity': 'even' if multiply else 'odd',
        # Exact-grid probes must match bit for bit; chains carry a relative limit.
        'l2_limit': 0 if group or multiply or split else .01 if depth == 2 else .05,
        'expected_conv_count': expected_convs,
        'benchmark_eligible': not (group or multiply),
        'source_ops': None if group or multiply else 2 * depth * 512 ** 2 * 4096,
        'entrypoint': 'scope_' + case_id.replace('-', '_'),
        'input_name': None,
        'output_name': None,
    }


def export_case(case_id: str, data: Path, out: Path) -> dict:
    """Export one case, audit the persisted asset, then record what was written."""
    from .common import sha

    data, out = Path(data), Path(out)
    info = describe(case_id)
    if out.exists():
        raise FileExistsError(out)
    if not (data / info['reference_file']).exists():
        raise ValueError('data profile lacks requested reference')
    for name, digest in read(data / 'manifest.json')['files'].items():
        if sha(data / name) != digest:
            raise ValueError('fixture identity mismatch: ' + name)

    if info['runtime'] == 'coreml':
        from ._coreml import build
    else:
        from ._coreai import build

    out.mkdir(parents=True)
    info.update(build(case_id, data, out, info))
    if not read(out / 'asset-audit.json')['passed']:
        raise ValueError('persisted asset audit failed')
    dump(out / 'export.json', info)
    return info

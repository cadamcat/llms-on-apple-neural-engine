"""Output and placement evidence.

Execution is never inferred from a requested compute unit or from a low-bit
representation.  Placement evidence here means observed ANE requests inside the
recorded control windows, plus the Core ML compute plan where one exists.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from .common import read, sha

ANE_REQUEST = 'ANEProgramProcessRequestDirect'
FALLBACK_MARKERS = ['Falling back', 'Full compile with ANE', 'Compiler internal error']
EVIDENCE_SCOPE = ('control-window participation plus Core ML preferred-device plan; '
                  'no per-kernel instruction proof')


def placement(path, pid, calls, expected_convs, runtime):
    """Count successful ANE requests inside each control window for one PID."""
    counts = [0] * len(calls)
    diagnostics, errors = [], []
    for line in Path(path, 'unified.ndjson').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('processID', event.get('processIdentifier')) != pid:
            continue
        message = event.get('eventMessage', '')
        if ANE_REQUEST in message and 'status=0x0' in message:
            when = datetime.datetime.fromisoformat(event['timestamp']).timestamp()
            for i, call in enumerate(calls):
                counts[i] += int(call['start_epoch_ns'] / 1e9 <= when <= call['end_epoch_ns'] / 1e9)
        if message.startswith('Unsupported op'):
            diagnostics.append(message)
        if any(marker in message for marker in FALLBACK_MARKERS):
            errors.append(message)

    plan = read(Path(path, 'plan.json'))
    convs = ([p for p in plan if p.get('operator_name', '').split('.')[-1] == 'conv']
             if isinstance(plan, list) else [])
    # Core AI exposes no per-operation plan, so it has no plan verdict to give.
    plan_ok = None if runtime == 'coreai' else (
        len(convs) == expected_convs
        and all('NeuralEngine' in x.get('preferred', '') for x in convs))
    return {
        'counts': counts,
        'every_control_has_ANE': bool(counts) and all(counts),
        'conv_plan': convs,
        'conv_plan_passed': plan_ok,
        'unsupported_messages': diagnostics,
        'compiler_or_fallback_messages': errors,
        'physical_INT8_proven': False,
        'evidence_scope': EVIDENCE_SCOPE,
    }


def verify_outputs(directory, metadata, data, calls):
    """Compare every recorded control output with its frozen reference."""
    import numpy as np

    data, directory = Path(data), Path(directory)
    reference = np.load(data / metadata['reference_file'])
    if metadata['reference_repeat_columns'] != 1:
        reference = np.tile(reference, (1, metadata['reference_repeat_columns']))
        reference = reference.reshape(metadata['output_shape'])
    if list(reference.shape) != metadata['output_shape']:
        raise ValueError('reference shape differs from output contract')

    rows, hashes = [], {}
    for call in calls:
        if not call['output_file']:
            continue
        path = directory / call['output_file']
        digest = sha(path)
        if digest != call['output_sha256']:
            raise ValueError('output identity mismatch')
        actual = np.fromfile(path, '<f2').reshape(reference.shape)
        if call['output_count'] != actual.size:
            raise ValueError('recorded output count mismatch')

        # A linear chain negates with its input; a multiplication probe does not.
        if call['phase'] == 'control_zero':
            target = np.zeros_like(reference)
        elif call['phase'] == 'control_negative' and metadata['negative_parity'] == 'odd':
            target = -reference
        else:
            target = reference

        wide_actual = actual.astype(np.float64)
        wide_target = target.astype(np.float64)
        norm = float(np.linalg.norm(wide_target))
        relative = float(np.linalg.norm(wide_actual - wide_target) / (norm or 1))
        finite = bool(np.isfinite(actual).all())
        nonzero = float(np.count_nonzero(actual) / actual.size)
        mismatch = int(np.count_nonzero(actual != target))
        limit = 0 if call['phase'] == 'control_zero' else metadata['l2_limit']
        passed = finite and (mismatch == 0 if limit == 0
                             else relative <= limit and nonzero >= .25)
        rows.append({
            'phase': call['phase'],
            'output_file': call['output_file'],
            'output_sha256': digest,
            'numeric_mismatches': mismatch,
            'bit_mismatches': int(np.count_nonzero(actual.view(np.uint16)
                                                  != target.view(np.uint16))),
            'relative_L2': relative if finite else None,
            'max_abs': float(np.max(np.abs(wide_actual - wide_target))) if finite else None,
            'finite': finite,
            'nonzero_fraction': nonzero,
            'limit': limit,
            'passed': passed,
        })
        hashes[call['phase']] = digest

    repeat = ('control_original' in hashes
              and hashes.get('control_original') == hashes.get('control_original_repeat'))
    return {
        'reference_file': metadata['reference_file'],
        'reference_sha256': sha(data / metadata['reference_file']),
        'comparisons': rows,
        'repeat_byte_equal': repeat,
        'passed': len(rows) >= 4 and all(x['passed'] for x in rows) and repeat,
    }

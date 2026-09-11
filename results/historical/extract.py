"""Copy metadata for ten closed runs out of the private research workspace.

This is an evidence-preparation step, not an experiment: it reads `result.json`
and its paired `run.json`, hashes both, keeps every measured record and
recomputes the median from the raw durations.  It imports no array, model or
device library, copies no weights or logs, and serialises workspace-relative
paths only.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, rel):
    return json.loads((root / rel).read_text())


CASES = {
    'coreai_w8a8_128': 'results/g1-int8-control-20260910/runs/coreai-heldout-w8a8-128',
    'coreml_w8a8_128_1': 'results/g1-int8-control-20260910/runs/heldout-w8a8-128-1',
    'coreml_w8a8_128_2': 'results/g1-int8-control-20260910/runs/heldout-w8a8-128-2',
    'coreai_fp16_128': 'results/g1-int8-control-20260910/runs/coreai-heldout-fp16-128',
    'coreml_fp16_128_1': 'results/g1-int8-control-20260910/runs/heldout-fp16-128-1',
    'coreml_fp16_128_2': 'results/g1-int8-control-20260910/runs/heldout-fp16-128-2',
    'wide_a8w4_128_1': 'results/g1-w4-bridge-20260910/runs/a8w4-128-1',
    'wide_a8w4_128_2': 'results/g1-w4-bridge-20260910/runs/a8w4-128-2',
    'wide_a8w4_2': 'results/g1-w4-bridge-20260910/runs/a8w4-2-bench',
    'split_a8w4_2': 'results/g1-w4-bridge-20260910/runs/split32-2-bench',
}

HARDWARE = ('Apple M5 Pro / Mac17,8; macOS 27.0 build 26A428 '
            '(as recorded by source workspace tooling/manifest)')
HARDWARE_SOURCE = ('results/g1-int8-control-20260910/before/TOOLING.md; '
                   'results/g1-int8-control-20260910/data/manifest.json')
ADMISSION_FIELDS = ('numerical_passed', 'every_control_has_ANE', 'all_convs_preferred_ANE',
                    'memory_gate_passed', 'repeat_hashes_passed', 'child_returncode')
RECORD_FIELDS = ('index', 'phase', 'start_epoch_ns', 'end_epoch_ns', 'duration_ns',
                 'output_sha256')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'refusing to overwrite existing output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, base_rel in CASES.items():
        base = root / base_rel
        result_path = base / 'result.json'
        run_path = base / 'run.json'
        admission_path = base / 'admission.json'
        result = load(root, base_rel + '/result.json')
        run = load(root, base_rel + '/run.json') if run_path.exists() else {}
        admission = (load(root, base_rel + '/admission.json')
                     if admission_path.exists() else {})
        records = result.get('records', [])
        measured = [r for r in records if r.get('phase') == 'measured']
        durations = [r['duration_ns'] / 1e6 for r in measured]
        rows.append({
            'label': label,
            'historical_import': True,
            'result_source': str(result_path.relative_to(root)),
            'result_sha256': sha(result_path),
            'run_source': str(run_path.relative_to(root)) if run_path.exists() else None,
            'run_sha256': sha(run_path) if run_path.exists() else None,
            'hardware_version': HARDWARE,
            'hardware_source': HARDWARE_SOURCE,
            'shape': [1, 512, 64, 64],
            'independent_process': {k: run.get(k) for k in ('pid', 'case', 'runtime', 'format')},
            'control_admission': {k: run.get(k, result.get(k)) for k in ADMISSION_FIELDS},
            'admission_source': (str(admission_path.relative_to(root))
                                 if admission_path.exists() else None),
            'admission_sha256': sha(admission_path) if admission_path.exists() else None,
            'benchmark_admitted': admission.get('benchmark_admitted'),
            'record_count_measured': len(measured),
            'duration_ms_all_measured': durations,
            'record_rows': [{k: r.get(k) for k in RECORD_FIELDS} for r in measured],
            'reported_p50_ms': run.get('p50_ms'),
            'reported_p95_ms': run.get('p95_ms'),
            'reported_equivalent_tops': run.get('equivalent_tops'),
            'source_ops': run.get('projection_ops'),
            'recomputed_median_ms': statistics.median(durations) if durations else None,
            'time_range_epoch_s': ([r.get('start_epoch_ns', 0) / 1e9 for r in records][:1]
                                   + [r.get('end_epoch_ns', 0) / 1e9 for r in records[-1:]]),
        })
    output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()

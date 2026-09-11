#!/usr/bin/env python3
"""Verify the portable historical evidence without source workspace access."""
from pathlib import Path
import argparse
import json
import math
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('selected_evidence', type=Path, nargs='?', default=Path(__file__).resolve().parents[1]/'selected-evidence.json')
    args = ap.parse_args()
    rows = json.loads(args.selected_evidence.read_text())
    by = {row['label']: row for row in rows}
    assert len(rows) >= 10
    for row in rows:
        durations = [item['duration_ns'] / 1e6 for item in row['record_rows']]
        assert len(durations) == row['record_count_measured'] == 30
        assert math.isclose(statistics.median(durations), row['recomputed_median_ms'], rel_tol=0, abs_tol=1e-12)
        if row['source_ops'] is not None and row['recomputed_median_ms']:
            tops = row['source_ops'] / (row['recomputed_median_ms'] / 1000) / 1e12
            assert math.isclose(tops, row['reported_equivalent_tops'], rel_tol=1e-9)
    ratio = by['split_a8w4_2']['recomputed_median_ms'] / by['wide_a8w4_2']['recomputed_median_ms']
    print(f'Historical split/wide p50 ratio: {ratio:.9f}')
    pairs=[('coreai_fp16_128','coreai_w8a8_128'),('coreml_fp16_128_1','coreml_w8a8_128_1'),('coreml_fp16_128_2','coreml_w8a8_128_2')]
    for fp,q in pairs:
        assert by[fp]['source_ops']==by[q]['source_ops'] and by[fp]['shape']==by[q]['shape']
        print(f'{fp} / {q}: {by[fp]["recomputed_median_ms"]/by[q]["recomputed_median_ms"]:.9f}')
    for fp in ('coreai_fp16_128', 'coreai_w8a8_128', 'coreml_fp16_128_1', 'coreml_w8a8_128_1'):
        assert by[fp]['benchmark_admitted'] is True
    print('historical evidence verification passed')


if __name__ == '__main__':
    main()

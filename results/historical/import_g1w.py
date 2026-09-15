"""Import the G1-W quantized-speed and numerical-probe evidence.

Six experiments contribute per-call timing records: exact-zero FP16 weights,
chain depth, weight codebook, scale granularity, and the Gemma 4 E4B mobile
QAT MLP stack. Numerical results are copied as scalar fields with their source
paths; the model-free QDQ multiply probe also writes its raw outputs for the
finding's standalone verifier. No weight tensor, activation array, model file
or absolute path is serialised, and no array, model or device library is
imported.
"""

import argparse
import gzip
import json
import re
from pathlib import Path

BASE = 'workplans/g1-w-e4b-mobile-qat-20260911'
CHECKPOINT = 'google/gemma-4-E4B-it-qat-mobile-ct'
ARMS = {
    'density': ['old', 'shuffled', 'sparse', 'permuted', 'fill_min', 'fill_16'],
    'boundary': ['bare-128', 'inner-128', 'input-128', 'output-128', 'both-128',
                 'bare-32', 'both-32', 'bare-8', 'both-8', 'bare-2', 'both-2',
                 'bare-1', 'both-1'],
    'codebook': ['old_fp16-128', 'old_w4a16-128', 'old_a8-128',
                 'full_fp16-128', 'full_w4a16-128', 'full_a8-128'],
    'scale': ['scalar_w4a16-128', 'scalar_a8-128', 'equal_w4a16-128', 'equal_a8-128',
              'varied_w4a16-128', 'varied_a8-128'],
    'e4b': ['bare', 'native', 'clip_product', 'split_w4', 'split_a8',
            'split_coarse_w4', 'split_coarse_a8'],
}
PROBE_ARMS = ['s16', 's16_clip', 's8', 's8_clip', 's4', 's4_clip', 's2', 's2_clip']
LOG_PATH = re.compile(r'^/\S+?\.mm:\d+: ')


class Reader:
    def __init__(self, root):
        self.root, self.sources = Path(root), set()

    def raw(self, rel):
        path = BASE + '/' + rel
        data = (self.root / path).read_bytes()
        self.sources.add(path)
        return data

    def json(self, rel):
        return json.loads(self.raw(rel))


def measured(calls, field='duration_ns'):
    rows = [c[field] for c in calls if c['phase'] == 'measured']
    assert rows and all(isinstance(v, int) and v > 0 for v in rows)
    return rows


def timings(read):
    out = {'density': {}, 'boundary': {}, 'codebook': {}, 'scale': {}, 'e4b': {}}
    for arm in ARMS['density']:
        out['density'][arm] = [measured(read.json(f'density-r1/runs/{arm}-{r}/calls.json'))
                               for r in range(3)]
    for experiment in ('boundary', 'codebook', 'scale'):
        for arm in ARMS[experiment]:
            out[experiment][arm] = [
                measured(read.json(f'{experiment}-r1/runs/{arm}-bench-{p}/calls.json'))
                for p in range(2)]
    for process in range(3):
        calls = read.json(f'coarse-r1/bench-{process}/host/calls.json')
        blocks = {}
        for c in calls:
            if c['phase'] == 'measured':
                blocks.setdefault(c['arm'], {}).setdefault(str(c['pair']), []).append(c['run_ns'])
        assert sorted(blocks) == sorted(ARMS['e4b'])
        out['e4b'][str(process)] = blocks
    return out


def reported(read):
    density = read.json('density-r1/SUMMARY.json')
    coarse = read.json('coarse-r1/SUMMARY.json')
    summaries = {name: read.json(f'{name}-r1/SUMMARY.json')
                 for name in ('boundary', 'codebook', 'scale')}
    return {
        'density': {
            'process_p50_ms': {a: [p['p50_ms'] for p in density['arms'][a]['processes']]
                               for a in ARMS['density']},
            'over_sparse_median': {a: density['comparisons'][f'{a}_over_sparse']['median']
                                   for a in ARMS['density']},
        },
        'boundary': {'p50_ms_by_pass': {k: v['p50_ms_by_repeat']
                                        for k, v in summaries['boundary']['formats'].items()},
                     'a8_over_w4a16': {k: v['geomean'] for k, v in
                                       summaries['boundary']['a8_over_w4a16_speedups'].items()},
                     'serialized_qdq_count': summaries['boundary']['serialized_qdq_count']},
        'codebook': {'p50_ms_by_pass': {k: v['p50_ms_by_repeat']
                                        for k, v in summaries['codebook']['formats'].items()},
                     'speedups': {w: {k: v['geomean'] for k, v in block.items()}
                                  for w, block in summaries['codebook']['speedups'].items()}},
        'scale': {'p50_ms_by_pass': {k: v['p50_ms_by_repeat']
                                     for k, v in summaries['scale']['formats'].items()},
                  'a8_over_w4a16': {k: v['a8_over_w4a16']['geomean']
                                    for k, v in summaries['scale']['speedups'].items()}},
        'e4b': {'block_median_ms': [[{'pair': b['pair'], 'arm': b['arm'],
                                      'run_ms': b['median_ms']['run_ns'],
                                      'calls': b['calls'], 'model_calls': b['model_calls']}
                                     for b in p['blocks']] for p in coarse['processes']],
                'speed_vs_bare_by_process': {k: v['run_ns']
                                             for k, v in coarse['speed_vs_bare'].items()}},
    }


def fixtures(read):
    density = read.json('density-r1/PREPARED.json')
    codebook = read.json('codebook-r1/SUMMARY.json')
    config = read.json('model-source/config.json')['text_config']
    state = read.json('STATE.json')
    keep = ('zero_fraction', 'row_norm_min', 'row_norm_max')
    return {
        'synthetic_chain': {'input_channels': 512, 'output_channels': 512, 'positions': 4096,
                            'kernel': '1x1 convolution'},
        'density_weights': {a: {k: density['arms'][a][k] for k in keep} for a in ARMS['density']},
        'codebook_full_zero_fraction': codebook['full_weights_zero_fraction'],
        'e4b': {'checkpoint': CHECKPOINT, 'revision': state['revision'],
                'hidden_size': config['hidden_size'],
                'intermediate_size': config['intermediate_size'],
                'hidden_activation': config['hidden_activation'],
                'positions': 64, 'layers_in_stack': 8,
                'inputs': 'synthetic controls, not text activations: positions 0-5 hold zero, '
                          'one-hot and code-boundary rows; positions 6-63 ("ordinary") are '
                          'RMS-normalized random rows',
                'stack': 'the first MLP repeated eight times, each after a shared FP16 RMS norm'},
    }


def numerics(read):
    single = read.json('coarse-r1/control-single/NUMERIC.json')
    deep = read.json('coarse-r1/control-deep/NUMERIC.json')
    composition = read.json('composition-r1/SUMMARY.json')['entries']
    fallback = {}
    for arm in ('toy_w4_f32', 'toy_w8_f32'):
        lines = set()
        for line in read.raw(f'composition-r1/control-{arm}/unified.ndjson').splitlines():
            try:
                message = json.loads(line).get('eventMessage', '')
            except json.JSONDecodeError:
                continue
            if 'Falling back' in message:
                lines.add(LOG_PATH.sub('', message.strip()))
            lines.update(re.findall(r'Compiler internal error: [^"\\\n]+', message))
        fallback[arm] = sorted(lines)
    gelu = read.json('numerics-r1/control-element-gelu/NUMERIC.json')
    unit = read.json('projection-r1/QUANT-ANALYSIS.json')['unit_qdq']
    return {
        'e4b_single_layer': {arm: {'all': v['all']['relative_l2'],
                                   'ordinary': v['ordinary']['relative_l2']}
                             for arm, v in single.items()},
        'e4b_eight_layer': {arm: {'all': v['all']['relative_l2'],
                                  'ordinary': v['ordinary']['relative_l2']}
                            for arm, v in deep.items()},
        'composition': {name: {k: v for k, v in entry.items()
                               if k not in ('reference', 'control_run')}
                        for name, entry in composition.items()},
        'fallback_log_messages': fallback,
        'gelu_zero_input': {mode: gelu[mode]['zero'] for mode in gelu},
        'unit_scale_qdq': {k: unit[k] for k in
                           ('finite_values', 'unit_even_mismatches', 'unit_away_mismatches')},
    }


def probe(read, destination):
    """Copy the model-free QDQ multiply outputs, graphs and per-call request counts."""
    trace = read.json('numerics-r1/portable-r2/TRACE.json')
    requests = {}
    for row in trace['requests_per_call']:
        requests.setdefault(row['arm'], []).append(row['requests'])
    record = {'environment': read.json('numerics-r1/AUDIT.json')['environment'],
              'input': 'all-ones FP16, shape [1,32,1,64]; a = channels 0-15, b = channels 16-31',
              'controls': ['original', 'zero', 'original repeated'],
              'input_scale': 1 / 16, 'arms': {}}
    for arm in PROBE_ARMS:
        folder = destination / arm
        folder.mkdir(parents=True)
        for index in range(3):
            data = read.raw(f'numerics-r1/portable-r2/host-output/{arm}-control-{index}.raw')
            (folder / f'control-{index}.raw').write_bytes(data)
        (folder / 'graph.mlir').write_bytes(read.raw(f'numerics-r1/portable-r2/{arm}/graph.mlir'))
        denominator = int(arm.split('_')[0][1:])
        record['arms'][arm] = {'output_scale': 1 / denominator, 'product_clamp': arm.endswith('_clip'),
                               'ane_requests_per_call': requests[arm]}
    assert trace['all_calls_have_ane'] and trace['successful_ane_requests'] == 24
    (destination / 'results.json').write_text(json.dumps(record, indent=2) + '\n')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new bundle directory')
    parser.add_argument('--probe-output', type=Path, required=True,
                        help='new directory for the finding verifier records')
    args = parser.parse_args()
    read = Reader(args.source_root)
    args.output.mkdir(parents=True)
    args.probe_output.mkdir(parents=True)
    evidence = {'schema_version': 1, 'historical_import': True, 'fresh_run': False,
                'scope': 'Component graphs on one M5 Pro. Timings are awaited function.run '
                         'calls; relative L2 compares a device output with its own '
                         'independent CPU reference. No full-model quality, physical INT8 '
                         'datapath or energy is measured.',
                'fixtures': fixtures(read), 'reported': reported(read),
                'numerics': numerics(read)}
    write(args.output / 'evidence.json', json.dumps(evidence, indent=2) + '\n')
    with gzip.GzipFile(args.output / 'timings.json.gz', 'xb', mtime=0) as handle:
        handle.write(json.dumps(timings(read), separators=(',', ':')).encode())
    probe(read, args.probe_output)
    provenance = {
        'importer': 'results/historical/import_g1w.py',
        'environment': read.json('numerics-r1/AUDIT.json')['environment'],
        'sources': sorted(read.sources),
        'transformations': [
            'per-call measured durations only; warmup, control and output hashes omitted',
            'E4B stack timings grouped by process, arm and pair; run_ns is the full pipeline',
            'numerical results copied as scalar fields; arrays stay in the workspace',
            'unified-log messages reduced to fallback lines with source-file prefixes removed',
        ],
    }
    write(args.output / 'provenance.json', json.dumps(provenance, indent=2) + '\n')
    print('wrote', args.output, 'and', args.probe_output)


if __name__ == '__main__':
    main()

"""Import the Core ML QDQ multiply probe runs into the finding's recorded evidence; never execute a device."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('import_g4a', HERE / 'import_g4a.py')
g4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4a)

PROBE = Path('workplans/g7-coreml-qdq-probe-20260915')
MAIN = ('export-bare-c512-n1024', 'runs/bare-c512-n1024-r1')
SMALL = (('export', 'runs/r1', 'none'), ('export-anchor-in', 'runs/anchor-in-r1', 'identity 1×1 convolution at the input'))
PHASES = ('original', 'zero', 'negative', 'repeat', 'benchmark', 'benchmark_repeat')


def run_record(reader, run, arm_units):
    result = reader.json(run / arm_units / 'RESULT.json')
    controls = reader.json(run / arm_units / 'host/controls.json')
    if [c['phase'] for c in controls] != list(PHASES) or reader.json(run / arm_units / 'host/CLOSED.json') != {'closed': True, 'measured': False}:
        raise ValueError('run_record:' + arm_units)
    return result


def extract(workspace, output):
    reader = g4a.Reader(workspace)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'outputs').mkdir()
    (output / 'graphs').mkdir()
    export, run = PROBE / MAIN[0], PROBE / MAIN[1]
    manifest = reader.json(export / 'EXPORT.json')
    if manifest['anchor'] != 'none' or manifest['shape'] != [1, 1024, 1, 1024]:
        raise ValueError('main_export_shape')
    arms, runs = {}, []
    for arm, record in manifest['arms'].items():
        if not record['audit']['passed']:
            raise ValueError('graph_audit:' + arm)
        text = reader.raw(Path(record['model_path']).parent.relative_to(reader.workspace) / 'graph.mil').decode()
        (output / 'graphs' / f'{arm}.mil').write_text(text)
        arms[arm] = {'output_scale': record['output_scale'], 'product_clamp': record['product_clamp'], 'audit': record['audit']}
        for units in ('cpuAndNeuralEngine', 'cpuOnly'):
            result = run_record(reader, run, f'{arm}-{units}')
            for phase in PHASES:
                data = reader.raw(run / f'{arm}-{units}' / 'host' / f'{phase}.raw')
                with (output / 'outputs' / f'{arm}-{units}-{phase}.raw.gz').open('xb') as target:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
                        stream.write(data)
            runs.append({'arm': arm, 'compute_units': units, 'host_returncode': result['host_returncode'],
                         'ane_requests_per_control': result['ane_requests_per_control'], 'ane_failure_rows': len(result['ane_failures']),
                         'preferred': result['preferred']})
    small = []
    for export_name, run_name, anchor in SMALL:
        for row in reader.json(PROBE / run_name / 'RESULTS.json'):
            small.append({'shape': [1, 32, 1, 64], 'anchor': anchor} |
                         {k: row[k] for k in ('arm', 'compute_units', 'ane_requests_per_control', 'preferred', 'observed',
                                              'reference', 'matches_reference')})
    record = {
        'environment': {'chip': 'Apple M5 Pro', 'macOS': '27.0', 'build': '26A428', 'xcode': '27.0', 'coremltools': manifest['coremltools']},
        'graph': 'y = Q_out(a × Q_1/16(b)); a and b are the two halves of the input channels; signed INT8, zero point 0',
        'input_scale': manifest['input_scale'], 'shape': manifest['shape'], 'output_shape': manifest['output_shape'],
        'inputs': manifest['inputs'] | {'negative': 'the control with every sign bit flipped', 'zero': 'all zeros'},
        'phases': list(PHASES), 'arms': arms, 'runs': runs,
        'small_graph_placement': small,
        'placement_method': 'Core ML compute plan preferred device per operation; successful direct ANE requests counted from the '
                            'host-PID unified log inside each control call window.'}
    (output / 'results.json').write_text(json.dumps(reader.clean(record), indent=2) + '\n')
    (output / 'provenance.json').write_text(json.dumps(reader.clean({
        'importer': 'results/historical/import_qdq_coreml.py', 'sources': sorted(reader.sources),
        'transformations': ['Keep the eight exported graph texts and each run\'s six control outputs, gzip-compressed with a zero mtime.',
                            'Keep per-run ANE request counts, failure-row counts and preferred devices; the logs and compiled models are not included.',
                            'Keep the two 32×64 placement runs as recorded scalars.']}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

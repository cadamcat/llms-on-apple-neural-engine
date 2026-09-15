"""Check the recorded Core ML QDQ multiply outputs, placement and graphs; standard library only, no device."""
import collections
import gzip
import json
import math
from pathlib import Path
import re
import struct

RECORDED = Path(__file__).resolve().parent / 'recorded'
UNITS = {'cpuAndNeuralEngine': 'MLNeuralEngineComputeDevice', 'cpuOnly': 'MLCPUComputeDevice'}
PAIRS = {'original': (1, 1), 'negative': (-1, -1), 'benchmark': (1, .5)}


def require(value, name):
    if not value:
        raise ValueError(name)


def qdq(x, scale):
    """Signed INT8, zero point 0, nearest with ties away from zero; no input here is a tie."""
    code = max(-128, min(127, math.floor(abs(x) / scale + 0.5) * (1 if x >= 0 else -1)))
    return code * scale


def check_graph(graph, s_in, s_out, clamped, shape, name):
    """Follow the persisted probe's dataflow; generated variable names are immaterial."""
    error = 'qdq_coreml_graph:' + name
    statements = re.findall(r'^\s*%(\w+): .+ = (\w+)\((.*)\)$', graph, re.MULTILINE)
    counts = collections.Counter(op for _, op, _ in statements)
    expected = {'slice_by_index': 2, 'quantize': 2, 'dequantize': 2, 'mul': 1, 'identity': 1}
    if clamped:
        expected['clip'] = 1
    require(counts == expected and len({key for key, _, _ in statements}) == len(statements), error)
    nodes = {'%' + key: (op, dict(re.findall(r'(\w+)=(.*?)(?=,\s*\w+=|$)', args)))
             for key, op, args in statements}

    def node(key, op):
        actual, args = nodes[key]
        require(actual == op, error)
        return args

    def qdq(output, scale):
        dq = node(output, 'dequantize')
        q = node(dq['input'], 'quantize')
        require(float(dq['scale']) == float(q['scale']) == scale and
                float(q.get('zero_point', '0')) == float(dq.get('zero_point', '0')) == 0 and
                q['output_dtype'] == '"int8"', error)
        return q['input']

    try:
        input_name = '%' + re.search(r'main\[\w+\]\(%(\w+):', graph)[1]
        output = re.findall(r'} -> \((%\w+)\)', graph)
        require(len(output) == 1, error)
        product = qdq(node(output[0], 'identity')['x'], s_out)
        if clamped:
            clip = node(product, 'clip')
            require(float(clip['alpha']) == -128 * s_out and float(clip['beta']) == 127 * s_out, error)
            product = clip['x']
        multiply = node(product, 'mul')
        operands = [multiply['x'], multiply['y']]
        a = next(key for key in operands if nodes[key][0] == 'slice_by_index')
        b = qdq(next(key for key in operands if nodes[key][0] == 'dequantize'), s_in)
        half = shape[1] // 2
        for key, begin, end in ((a, [0, 0, 0, 0], [shape[0], half, *shape[2:]]),
                                (b, [0, half, 0, 0], shape)):
            part = node(key, 'slice_by_index')
            require(part['x'] == input_name and json.loads(part['begin']) == begin and json.loads(part['end']) == end, error)
    except (KeyError, IndexError, TypeError, ValueError, StopIteration) as exc:
        raise ValueError(error) from exc


def outputs(recorded=RECORDED):
    record = json.loads((recorded / 'results.json').read_text())
    size = 2 * math.prod(record['output_shape'])
    s_in = record['input_scale']
    rows = []
    for arm, spec in record['arms'].items():
        s_out = spec['output_scale']
        require(spec['product_clamp'] == arm.endswith('_clip'), 'qdq_coreml_clamp_flag:' + arm)
        graph = (recorded / 'graphs' / f'{arm}.mil').read_text()
        check_graph(graph, s_in, s_out, spec['product_clamp'], record['shape'], arm)
        reference = {k: qdq(a * qdq(b, s_in), s_out) for k, (a, b) in PAIRS.items()}
        # The Core AI rule: the b branch's integer code dequantized with the output scale.
        substitution = {k: qdq(a * round(b / s_in) * s_out, s_out) for k, (a, b) in PAIRS.items()}
        for run in (r for r in record['runs'] if r['arm'] == arm):
            units = run['compute_units']
            name = f'{arm}-{units}'
            values, raw = {}, {}
            for phase in record['phases']:
                data = gzip.decompress((recorded / 'outputs' / f'{name}-{phase}.raw.gz').read_bytes())
                require(len(data) == size, f'qdq_coreml_output:{name}:{phase}')
                raw[phase] = data
                values[phase] = sorted(set(v[0] for v in struct.iter_unpack('<e', data)))
            require(raw['original'] == raw['repeat'] and raw['benchmark'] == raw['benchmark_repeat'], 'qdq_coreml_repeat:' + name)
            require(raw['zero'] == bytes(size), 'qdq_coreml_zero:' + name)
            require(all(len(values[p]) == 1 for p in PAIRS), 'qdq_coreml_constant_output:' + name)
            observed = {p: values[p][0] for p in PAIRS}
            require(run['host_returncode'] == 0 and run['ane_failure_rows'] == 0 and set(run['preferred'].values()) == {UNITS[units]},
                    'qdq_coreml_placement:' + name)
            requests = 1 if units == 'cpuAndNeuralEngine' else 0
            require(run['ane_requests_per_control'] == [requests] * len(record['phases']), 'qdq_coreml_ane_requests:' + name)
            expected = substitution if units == 'cpuAndNeuralEngine' and not spec['product_clamp'] else reference
            require(observed == expected, 'qdq_coreml_values:' + name)
            rows.append({'arm': arm, 'compute_units': units, 'output_scale': s_out, 'observed': observed, 'reference': reference,
                         'matches_reference': observed == reference})
    for row in record['small_graph_placement']:
        require(row['ane_requests_per_control'] == [0] * len(record['phases']) and set(row['preferred'].values()) == {'MLCPUComputeDevice'}
                and row['matches_reference'], 'qdq_coreml_small_graph:' + row['arm'] + ':' + row['compute_units'])
    return record, rows


if __name__ == '__main__':
    record, rows = outputs()
    for r in rows:
        print(f"{r['arm']:9s} {r['compute_units']:19s} observed {r['observed']}  reference {r['reference']}")
    print(f"{len(rows)} Core ML runs checked; {len(record['small_graph_placement'])} 32×64 runs selected the CPU. No device execution.")

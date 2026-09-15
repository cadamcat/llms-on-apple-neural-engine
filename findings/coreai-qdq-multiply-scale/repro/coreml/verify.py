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


def outputs(recorded=RECORDED):
    record = json.loads((recorded / 'results.json').read_text())
    size = 2 * math.prod(record['output_shape'])
    s_in = record['input_scale']
    rows = []
    for arm, spec in record['arms'].items():
        s_out = spec['output_scale']
        require(spec['product_clamp'] == arm.endswith('_clip'), 'qdq_coreml_clamp_flag:' + arm)
        graph = (recorded / 'graphs' / f'{arm}.mil').read_text()
        ops = collections.Counter(re.findall(r'= (\w+)\(', graph))
        scales = sorted(float(v) for v in re.findall(r'= quantize\(input=%\w+, scale=([0-9.]+)', graph))
        require(ops['quantize'] == ops['dequantize'] == 2 and ops['mul'] == 1 and ops['clip'] == int(spec['product_clamp'])
                and ops['conv'] == 0 and scales == sorted([s_in, s_out]), 'qdq_coreml_graph:' + arm)
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

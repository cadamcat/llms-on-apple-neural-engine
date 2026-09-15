"""Check the recorded QDQ multiply outputs, or a new run's outputs, without Core AI or a device."""
from pathlib import Path
import argparse
import json
import math
import struct
import sys

ROOT = Path(__file__).resolve().parent
ARMS = ['s16', 's16_clip', 's8', 's8_clip', 's4', 's4_clip', 's2', 's2_clip']
VALUES = 16 * 64


def require(condition, name):
    if not condition:
        raise ValueError(name)


def qdq(x, scale):
    """Signed INT8, zero point 0, nearest with ties away from zero; inputs here have no ties."""
    code = max(-128, min(127, math.floor(abs(x) / scale + 0.5) * (1 if x >= 0 else -1)))
    return code * scale


def read_half(path, name):
    data = path.read_bytes()
    require(len(data) == 2 * VALUES, name + '.size')
    values = [v[0] for v in struct.iter_unpack('<e', data)]
    require(all(math.isfinite(v) for v in values), name + '.finite')
    return data, values


def verify(outputs=None):
    record = json.loads((ROOT / 'recorded/results.json').read_text())
    require(sorted(record['arms']) == sorted(ARMS), 'recorded.arms')
    summary = {}
    for arm in ARMS:
        spec = record['arms'][arm]
        require(spec['product_clamp'] == arm.endswith('_clip'), arm + '.clamp_flag')
        s_in, s_out = record['input_scale'], spec['output_scale']
        reference = qdq(1.0 * qdq(1.0, s_in), s_out)
        # The substitution model: the input branch's integer code dequantized with the output scale.
        predicted = round(1.0 / s_in) * s_out
        outputs_for_arm = []
        for index in range(3):
            if outputs is None:
                path = ROOT / 'recorded' / arm / f'control-{index}.raw'
            else:
                path = Path(outputs) / f'{arm}-control-{index}.raw'
            outputs_for_arm.append(read_half(path, f'{arm}.control-{index}'))
        require(outputs_for_arm[0][0] == outputs_for_arm[2][0], arm + '.repeat')
        require(all(v == 0 for v in outputs_for_arm[1][1]), arm + '.zero')
        observed = sorted(set(outputs_for_arm[0][1]))
        row = {'reference': reference, 'observed': observed,
               'matches_reference': observed == [reference],
               'matches_scale_substitution': observed == [predicted]}
        if outputs is None:
            require(spec['ane_requests_per_call'] == [1, 1, 1], arm + '.ane_requests')
            if arm.endswith('_clip'):
                require(row['matches_reference'], arm + '.reference')
            else:
                require(row['matches_scale_substitution'], arm + '.prediction')
        summary[arm] = row
    return {'mode': 'new_outputs' if outputs else 'recorded_evidence', 'arms': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', help='host-output directory written by run.py')
    args = parser.parse_args()
    try:
        result = verify(args.outputs)
    except ValueError as error:
        print(f'FAIL: {error}', file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

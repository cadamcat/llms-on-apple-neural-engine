"""Core ML version of the eight QDQ multiply probes: QDQ(a * QDQ(b, 1/16), output_scale), no weights."""
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np

ARMS = [(d, clip) for d in (16, 8, 4, 2) for clip in (False, True)]


def arm_name(denominator, clip):
    return f's{denominator}' + ('_clip' if clip else '')


def export(out, anchor='none', half=16, positions=64):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.frontend.milproto.load import load
    out.mkdir(parents=True, exist_ok=False)
    control = np.ones((1, 2 * half, 1, positions), np.float16)
    benchmark = control.copy()
    benchmark[:, half:] = 0.5
    control.tofile(out / 'control.raw')
    benchmark.tofile(out / 'benchmark.raw')
    records = {}
    for denominator, clip in ARMS:
        arm = arm_name(denominator, clip)
        folder = out / arm
        folder.mkdir()
        s_in, s_out = np.float16(1 / 16), np.float16(1 / denominator)

        def qdq(x, scale):
            q = mb.quantize(input=x, scale=scale, zero_point=np.int8(0), output_dtype='int8')
            return mb.dequantize(input=q, scale=scale, zero_point=np.int8(0))

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2 * half, 1, positions), dtype=types.fp16)], opset_version=ct.target.iOS18)
        def program(x):
            if anchor in ('in', 'both'):
                # Identity 1x1 convolution: exact in FP16, gives the placement heuristic a reason to choose ANE.
                x = mb.conv(x=x, weight=np.eye(2 * half, dtype=np.float16).reshape(2 * half, 2 * half, 1, 1), name='anchor_in')
            a = mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[1, half, 1, positions])
            b = mb.slice_by_index(x=x, begin=[0, half, 0, 0], end=[1, 2 * half, 1, positions])
            p = mb.mul(x=a, y=qdq(b, s_in))
            if clip:
                p = mb.clip(x=p, alpha=np.float16(-128) * s_out, beta=np.float16(127) * s_out)
            y = qdq(p, s_out)
            if anchor == 'both':
                y = mb.conv(x=y, weight=np.eye(half, dtype=np.float16).reshape(half, half, 1, 1), name='anchor_out')
            return mb.identity(x=y, name='y')

        path = folder / 'model.mlpackage'
        model = ct.convert(program, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS18,
                           compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        model.save(str(path))
        spec = ct.utils.load_spec(str(path))
        blobs = list(path.rglob('weight.bin'))
        persisted = load(spec, spec.specificationVersion, file_weights_dir=str(blobs[0].parent) if blobs else None)
        (folder / 'graph.mil').write_text(str(persisted))
        fn = persisted.functions['main']
        ops = list(fn.operations)
        counts = collections.Counter(o.op_type for o in ops if o.op_type != 'const')
        # Audit the persisted graph, not the builder: both QDQ pairs, their scales and the multiply must survive conversion.
        quantize = [o for o in ops if o.op_type == 'quantize']
        scales = sorted(float(o.scale.val) for o in quantize)
        audit = {'op_counts': dict(counts), 'quantize_scales': scales,
                 'dequantize_follows_quantize_with_same_scale': all(
                     o.input.op.op_type == 'quantize' and np.array_equal(o.scale.val, o.input.op.scale.val)
                     for o in ops if o.op_type == 'dequantize')}
        audit['passed'] = (counts['conv'] == {'none': 0, 'in': 1, 'both': 2}[anchor] and counts['quantize'] == counts['dequantize'] == 2 and counts['mul'] == 1 and
                           counts['clip'] == int(clip) and scales == sorted([float(s_in), float(s_out)]) and
                           audit['dequantize_follows_quantize_with_same_scale'])
        records[arm] = {'output_scale': float(s_out), 'product_clamp': clip, 'model_path': str(path),
                        'output_name': fn.outputs[0].name, 'audit': audit}
    manifest = {'input_scale': 1 / 16, 'anchor': anchor, 'shape': [1, 2 * half, 1, positions], 'output_shape': [1, half, 1, positions], 'arms': records,
                'inputs': {'control': f'all-ones FP16 [1,{2 * half},1,{positions}]; a = first {half} channels, b = last {half}',
                           'benchmark': 'a = 1, b = 0.5'},
                'coremltools': ct.__version__,
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'control.raw', out / 'benchmark.raw', Path(__file__))}}
    (out / 'EXPORT.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--anchor', choices=['none', 'in', 'both'], default='none')
    parser.add_argument('--half', type=int, default=16)
    parser.add_argument('--positions', type=int, default=64)
    args = parser.parse_args()
    result = export(args.out.resolve(), args.anchor, args.half, args.positions)
    print(json.dumps({arm: r['audit'] for arm, r in result['arms'].items()}, indent=1))

"""Core ML export and persisted proto/blob audit. Requires the ``apple`` extra.

The audit deliberately reloads the saved package rather than trusting the
in-memory program: it walks the persisted operands, decodes weights, scales and
zero points independently, and fails on any representation it cannot recognise.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.milproto.load import load
from coremltools.converters.mil.mil import types

from .common import dump, file_hashes

WEIGHT_SCALE = np.float16(.044189453125)
ACTIVATION_SCALE = np.float16(.125)
GROUP = 32


def build(case_id, data, out, info):
    """Export one Core ML case, then audit what was actually written to disk."""
    group = '-group-' in case_id
    split = case_id.endswith('split32')
    fmt = case_id.split('-')[1]
    depth = info['expected_conv_count'] if not group else 1

    weights = np.load(data / ('group_codes.npy' if group else 'weights.npy'), mmap_mode='r')
    scales = np.load(data / 'group_scales.npy') if group else None
    input_scales = np.load(data / 'group_input_scales.npy') if group else None

    @mb.program(input_specs=[mb.TensorSpec(shape=tuple(info['shape']), dtype=types.fp16)],
                opset_version=ct.target.iOS18)
    def prog(x):
        if group:
            zero = np.zeros(64, np.int8)
            q = mb.quantize(input=x, scale=input_scales, zero_point=zero, axis=1,
                            output_dtype='int8')
            z = mb.dequantize(input=q, scale=input_scales, zero_point=zero, axis=1)
            if split:
                parts = []
                for g in range(2):
                    codes = np.array(weights[:, g * GROUP:(g + 1) * GROUP])
                    codes = codes.view(types.nptype_from_builtin(types.int4))
                    w = mb.constexpr_blockwise_shift_scale(
                        data=codes, scale=scales[:, g:g + 1].copy())
                    v = mb.slice_by_index(x=z, begin=[0, g * GROUP, 0, 0],
                                          end=[1, (g + 1) * GROUP, 1, 64])
                    parts.append(mb.conv(x=v, weight=w))
                return mb.add(x=parts[0], y=parts[1], name='y')
            codes = np.array(weights).view(types.nptype_from_builtin(types.int4))
            return mb.conv(x=z, weight=mb.constexpr_blockwise_shift_scale(data=codes,
                                                                         scale=scales),
                           name='y')
        for i in range(depth):
            codes = np.array(weights[i]).reshape(512, 512, 1, 1)
            if fmt == 'w8a8':
                w = mb.constexpr_affine_dequantize(quantized_data=codes, scale=WEIGHT_SCALE,
                                                   zero_point=np.int8(0), axis=0)
            else:
                w = codes.astype(np.float16) * WEIGHT_SCALE
            x = mb.conv(x=x, weight=w, name=f'conv_{i:03d}')
            if fmt == 'w8a8' and i < depth - 1:
                q = mb.quantize(input=x, scale=ACTIVATION_SCALE, zero_point=np.int8(0),
                                output_dtype='int8')
                x = mb.dequantize(input=q, scale=ACTIVATION_SCALE, zero_point=np.int8(0))
        return x

    model = ct.convert(prog, convert_to='mlprogram',
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.iOS18, skip_model_load=True)
    asset = out / 'model.mlpackage'
    model.save(str(asset))
    spec = audit(asset, out, case_id, data, info)
    return {'entrypoint': 'main',
            'input_name': spec.description.input[0].name,
            'output_name': spec.description.output[0].name,
            'model_path': str(asset),
            'representation': ('Core ML direct signed INT4 blockwise' if group
                               else 'Core ML ' + fmt)}


def audit(asset, out, case_id, data, info):
    """Reload the saved package and independently decode what it really contains."""
    group = '-group-' in case_id
    split = case_id.endswith('split32')
    fmt = case_id.split('-')[1]
    depth = info['expected_conv_count'] if not group else 1
    weights = np.load(data / ('group_codes.npy' if group else 'weights.npy'), mmap_mode='r')
    scales = np.load(data / 'group_scales.npy') if group else None
    input_scales = np.load(data / 'group_input_scales.npy') if group else None

    spec = ct.utils.load_spec(str(asset))
    # From here on, only the persisted package is trusted.
    blobs = list(asset.rglob('weight.bin'))
    assert len(blobs) == 1
    persisted = load(spec, spec.specificationVersion, file_weights_dir=str(blobs[0].parent))
    fn = persisted.functions['main']
    ops = list(fn.operations)
    assert (list(fn.inputs) == ['x'] and fn.inputs['x'].shape == tuple(info['shape'])
            and fn.inputs['x'].dtype == types.fp16)
    assert (len(fn.outputs) == 1 and fn.outputs[0].shape == tuple(info['output_shape'])
            and fn.outputs[0].dtype == types.fp16)

    convs = [o for o in ops if o.op_type == 'conv']
    qops = [o for o in ops if o.op_type == 'quantize']
    dqops = [o for o in ops if o.op_type == 'dequantize']
    assert len(convs) == info['expected_conv_count']
    expected_qdq = 1 if group else depth - 1 if fmt == 'w8a8' else 0
    assert len(qops) == len(dqops) == expected_qdq

    for q, dq in zip(qops, dqops):
        assert dq.input is q.outputs[0]
        for op in (q, dq):
            assert op.zero_point is None or not np.any(op.zero_point.val)
            expected = input_scales if group else np.array(.125, np.float16)
            assert np.asarray(op.scale.val).tobytes() == expected.tobytes()
            if group:
                assert int(op.axis.val) == 1
    if group:
        assert qops[0].input is fn.inputs['x']

    decoded = []
    for i, conv in enumerate(convs):
        assert np.array_equal(conv.strides.val, [1, 1])
        assert np.array_equal(conv.dilations.val, [1, 1])
        assert int(conv.groups.val) == 1
        if group:
            wanted_codes = np.array(weights[:, i * GROUP:(i + 1) * GROUP] if split else weights)
            wanted_scale = scales[:, i:i + 1] if split else scales
        else:
            wanted_codes = np.array(weights[i]).reshape(512, 512, 1, 1)
            wanted_scale = np.array(WEIGHT_SCALE)
        weight_op = conv.weight.op

        if group:
            assert weight_op.op_type == 'constexpr_blockwise_shift_scale'
            assert weight_op.data.dtype == types.int4
            codes = np.asarray(weight_op.data.val)
            scale = np.asarray(weight_op.scale.val)
            assert weight_op.offset is None or not np.any(weight_op.offset.val)
            actual = (codes.astype(np.float64)
                      * np.repeat(scale.astype(np.float64), GROUP, axis=1)).astype(np.float16)
            assert actual.tobytes() == weight_op.materialized_val_inference().tobytes()
        elif fmt == 'w8a8':
            assert weight_op.op_type == 'constexpr_affine_dequantize'
            codes = np.asarray(weight_op.quantized_data.val)
            scale = np.asarray(weight_op.scale.val)
            assert codes.dtype == np.int8 and not np.any(weight_op.zero_point.val)
            assert int(weight_op.axis.val) == 0
            actual = (codes.astype(np.float64) * scale.astype(np.float64)).astype(np.float16)
            assert actual.tobytes() == weight_op.materialized_val_inference().tobytes()
        else:
            assert weight_op.op_type == 'const'
            codes, scale = wanted_codes, wanted_scale
            actual = np.asarray(conv.weight.val)

        expected = (wanted_codes.astype(np.float64)
                    * (np.repeat(wanted_scale.astype(np.float64), GROUP, axis=1) if group
                       else float(WEIGHT_SCALE))).astype(np.float16)
        assert np.array_equal(codes, wanted_codes)
        assert scale.tobytes() == wanted_scale.tobytes()
        assert actual.tobytes() == expected.tobytes()

        # Follow the operands, so a correct weight in the wrong place still fails.
        if group and split:
            slicer = conv.x.op
            assert slicer.op_type == 'slice_by_index' and slicer.x is dqops[0].outputs[0]
            assert slicer.begin.val.tolist() == [0, i * GROUP, 0, 0]
            assert slicer.end.val.tolist() == [1, (i + 1) * GROUP, 1, 64]
        elif group:
            assert conv.x is dqops[0].outputs[0]
        elif i == 0:
            assert conv.x is fn.inputs['x']
        elif fmt == 'w8a8':
            assert conv.x is dqops[i - 1].outputs[0]
            assert qops[i - 1].input is convs[i - 1].outputs[0]
        else:
            assert conv.x is convs[i - 1].outputs[0]

        decoded.append({'conv': i, 'shape': list(actual.shape),
                        'sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                        'weight_op': weight_op.op_type})

    if group and split:
        end = fn.outputs[0].op
        assert end.op_type == 'add'
        assert end.x is convs[0].outputs[0] and end.y is convs[1].outputs[0]
    else:
        assert fn.outputs[0] is convs[-1].outputs[0]

    (out / 'persisted-graph.mil').write_text(str(persisted))
    dump(out / 'asset-audit.json', {
        'passed': True,
        'method': 'persisted proto/blob -> version-specific PyMIL operand walk and '
                  'independent weight decode',
        'asset_files': file_hashes(asset),
        'actual_op_counts': dict(Counter(o.op_type for o in ops)),
        'weight_decodes': decoded,
        'checks': ['input/output shape and dtype',
                   'actual convolution/quantize counts',
                   'conv and QDQ SSA connections',
                   'weight values, signed codes, scales, zero points',
                   'group slice and add operands'],
    })
    return spec

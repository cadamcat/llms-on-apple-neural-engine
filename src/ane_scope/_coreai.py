"""Core AI synthetic graphs, with a persisted-bytecode audit pinned to the 0.4.1 API.

Core AI's authoring and bytecode interfaces are private and version specific.
The audit reads the saved bytecode, walks the SSA graph and decodes every
payload independently; an asset it cannot parse fails rather than passing
quietly.  A future package version may need a separately reviewed adapter.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter

import coreai_torch
import numpy as np
import torch
import torch.nn.functional as F
from coreai.authoring import AIProgram
from coreai_torch._compression.custom_layers import (ScaledPalettizeModule,
                                                     WeightDequantizeModule)
from torch import nn

from .common import dump, file_hashes

WEIGHT_SCALE = np.float16(.044189453125)
GROUP = 32
MLIR_DTYPES = {'f16': '<f2', 'si8': 'i1', 'ui8': 'u1', 'si16': '<i2',
               'si32': '<i4', 'ui32': '<u4'}


def qdq(x, scale, zero, axis=0):
    q = torch.ops.coreai.quantize.default(x, scale, torch.int8, zero, None, axis)
    return torch.ops.coreai.dequantize.default(q, scale, zero, None, axis, torch.int8)


class Conv(nn.Module):
    """One convolution whose weight is FP16, INT8 plus scale, or a four-bit LUT."""

    def __init__(self, codes, scales, fmt):
        super().__init__()
        self.fmt = fmt
        c = torch.from_numpy(np.array(codes))
        s = torch.from_numpy(np.array(scales, dtype=np.float16))
        zero = torch.zeros_like(s, dtype=torch.int8)
        if fmt == 'fp16':
            self.register_buffer('w', c.half() * s)
        elif fmt == 'w8a8':
            self.w = WeightDequantizeModule(c, s, zero_point=zero, input_dtype=torch.int8,
                                            output_dtype=torch.float16)
        else:
            self.w = ScaledPalettizeModule(
                (c.to(torch.int16) + 8).to(torch.uint8),
                torch.arange(-8, 8, dtype=torch.int8).reshape(1, 1, 1, 1, 16, 1),
                s, vector_axis=0, zero_point=zero, input_dtype=torch.int8,
                output_dtype=torch.float16)

    def forward(self, x):
        return F.conv2d(x, self.w if self.fmt == 'fp16' else self.w())


class Chain(nn.Module):
    """The depth-N convolution chain, optionally partitioned into K32 pieces."""

    def __init__(self, weights, fmt, split):
        super().__init__()
        self.fmt, self.split = fmt, split
        scale = np.full((1, 1, 1, 1), WEIGHT_SCALE)
        self.layers = nn.ModuleList([
            nn.ModuleList([Conv(w[:, j:j + GROUP].reshape(512, GROUP, 1, 1), scale, fmt)
                           for j in range(0, 512, GROUP)])
            if split else Conv(w.reshape(512, 512, 1, 1), scale, fmt)
            for w in weights])
        self.register_buffer('s', torch.tensor(.125, dtype=torch.float16))
        self.register_buffer('z', torch.tensor(0, dtype=torch.int8))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.split:
                partials = [b(x[:, j * GROUP:(j + 1) * GROUP, :, :])
                            for j, b in enumerate(layer)]
                while len(partials) > 1:                 # balanced adjacent reduction
                    partials = [partials[j] + partials[j + 1]
                                for j in range(0, len(partials), 2)]
                x = partials[0]
            else:
                x = layer(x)
            if self.fmt != 'fp16' and i < len(self.layers) - 1:
                x = qdq(x, self.s, self.z)
        return x


class Group(nn.Module):
    """The K64 grouped-scale probe, native or split into two K32 convolutions."""

    def __init__(self, codes, scales, input_scales, split):
        super().__init__()
        self.split = split
        if split:
            self.parts = nn.ModuleList([
                Conv(codes[:, j * GROUP:(j + 1) * GROUP], scales[:, j:j + 1], 'a8w4')
                for j in range(2)])
        else:
            self.parts = nn.ModuleList([Conv(codes, scales, 'a8w4')])
        self.register_buffer('s', torch.from_numpy(input_scales.copy()))
        self.register_buffer('z', torch.zeros(64, dtype=torch.int8))

    def forward(self, x):
        x = qdq(x, self.s, self.z, 1)
        if self.split:
            return self.parts[0](x[:, :GROUP, :, :]) + self.parts[1](x[:, GROUP:, :, :])
        return self.parts[0](x)


class Multiply(nn.Module):
    """The exact-grid QDQ multiplication probe, in four graph expressions."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.register_buffer('sg', torch.tensor(2. if mode == 'equal' else .5,
                                                dtype=torch.float16))
        self.register_buffer('su', torch.tensor(2., dtype=torch.float16))
        self.register_buffer('z', torch.tensor(0, dtype=torch.int8))

    def apply_qdq(self, x, scale):
        if self.mode == 'explicit_q':
            q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        else:
            q = torch.ops.coreai.quantize.default(x, scale, torch.int8, self.z)
        return torch.ops.coreai.dequantize.default(q, scale, self.z, None, 0, torch.int8)

    def forward(self, x):
        # Branch construction order is the variable under test in reverse_order.
        if self.mode == 'reverse_order':
            u = self.apply_qdq(x[1:2], self.su)
            g = self.apply_qdq(x[0:1], self.sg)
        else:
            g = self.apply_qdq(x[0:1], self.sg)
            u = self.apply_qdq(x[1:2], self.su)
        return g * u


def _parse_bytecode(asset, out):
    """Read the saved bytecode into constants, operations and raw resources."""
    text = str(AIProgram._load_bytecode(asset / 'main.mlirb'))
    body = text.split('\n{-#', 1)[0]
    (out / 'persisted-graph.mlir').write_text(body)
    resources = {name: bytes.fromhex(payload) for name, payload in
                 re.findall(r'([A-Za-z_][A-Za-z0-9_]*): "0x([0-9A-Fa-f]*)"', text)}
    constants, ops = {}, {}
    for line in body.splitlines():
        match = re.match(r'\s*(%\d+) = coreai\.constant (.+) : tensor<([^>]+)>', line)
        if match:
            constants[match[1]] = (match[2], match[3])
            continue
        match = re.match(r'\s*(%\d+) = (coreai\.[a-z_0-9.]+) (.+?) : ', line)
        if match:
            ops[match[1]] = (match[2], re.findall(r'%\w+', match[3]))
    return body, constants, ops, resources


def _reader(constants, resources):
    """Return a function that materialises one constant, four-bit codes included."""

    def value(ssa):
        literal, typ = constants[ssa]
        tokens = typ.split('x')
        dtype = tokens[-1]
        shape = tuple(map(int, tokens[:-1]))
        count = int(np.prod(shape)) if shape else 1
        if literal.startswith('dense_resource'):
            raw = resources[re.fullmatch(r'dense_resource<([^>]+)>', literal)[1]][4:]
            if dtype == 'ui4':
                packed = np.frombuffer(raw, np.uint8)
                array = np.empty(count, np.uint8)
                array[::2] = packed & 15
                array[1::2] = packed >> 4
            else:
                array = np.frombuffer(raw, MLIR_DTYPES[dtype])
                assert array.size == count
        else:
            parsed = ast.literal_eval(literal[len('dense<'):-1])
            if isinstance(parsed, str) and parsed.startswith('0x'):
                array = np.frombuffer(bytes.fromhex(parsed[2:]), MLIR_DTYPES[dtype])
                assert array.size == count
            else:
                array = np.asarray(parsed, dtype=MLIR_DTYPES.get(dtype, 'u1'))
            if array.size == 1:
                array = np.full(count, array.item(), dtype=array.dtype)
        return array.reshape(shape)

    return value


def audit(asset, out, case_id, data, info):
    """Walk the persisted graph and independently decode every weight and scale."""
    body, constants, ops, resources = _parse_bytecode(asset, out)
    value = _reader(constants, resources)

    head = body.splitlines()[1]
    expected_in = 'x'.join(map(str, info['shape'])) + 'xf16'
    expected_out = 'x'.join(map(str, info['output_shape'])) + 'xf16'
    assert f'@{info["entrypoint"]}(' in head
    assert f'%arg0: tensor<{expected_in}>' in head
    assert f'-> (tensor<{expected_out}>' in head
    names = re.findall(r'coreai.name = "([^"]+)"', head)
    assert len(names) == 2
    final = re.search(r'coreai.output (%\d+) :', body)[1]

    counts = Counter(op for op, args in ops.values())
    convs = [(key, args) for key, (op, args) in ops.items() if op == 'coreai.conv2d']
    assert len(convs) == info['expected_conv_count']

    group = '-group-' in case_id
    multiply = '-qdq-' in case_id
    split = 'split32' in case_id
    fmt = case_id.split('-')[1]
    if multiply:
        expected_q = 0 if case_id.endswith('explicit_q') else 2
    elif fmt == 'fp16':
        expected_q = 0
    elif group or split:
        expected_q = 1
    else:
        expected_q = len(convs) - 1
    assert counts['coreai.quantize'] == expected_q
    assert counts['coreai.dequantize'] == (2 if multiply else expected_q)

    qops = [(key, args) for key, (op, args) in ops.items() if op == 'coreai.quantize']
    dqops = [(key, args) for key, (op, args) in ops.items() if op == 'coreai.dequantize']
    for key, args in dqops:
        assert not np.any(value(args[2])) and not np.any(value(args[3]))
        if not (multiply and case_id.endswith('explicit_q')):
            producer, produced_args = ops[args[0]]
            assert producer == 'coreai.quantize'
            assert all(np.array_equal(value(a), value(b))
                       for a, b in zip(args[1:], produced_args[1:]))
        if group:
            expected = np.load(data / 'group_input_scales.npy')
            assert value(args[1]).tobytes() == expected.tobytes()
            assert int(value(args[4])) == 1
        elif not multiply:
            assert float(value(args[1])) == .125 and int(value(args[4])) == 0

    if multiply:
        assert counts['coreai.decomposable.broadcasting_mul'] == 1
        op, args = ops[final]
        assert op == 'coreai.decomposable.broadcasting_mul'
        assert set(args) == {key for key, _ in dqops}
        scales = sorted(float(value(dq_args[1])) for _, dq_args in dqops)
        assert scales == ([2., 2.] if case_id.endswith('-equal') else [.5, 2.])

    weights = (None if multiply else
               np.load(data / ('group_codes.npy' if group else 'weights.npy'), mmap_mode='r'))
    scales = (np.load(data / 'group_scales.npy') if group
              else np.full((1, 1, 1, 1), WEIGHT_SCALE, np.float16))

    def check_slice(key, base, start, width):
        op, args = ops[key]
        assert op == 'coreai.slice' and args[0] == base
        begin = value(args[1]).reshape(-1)
        end = value(args[2]).reshape(-1)
        stride = value(args[3]).reshape(-1)
        assert begin.tolist() == [0, start, 0, 0]
        assert min(int(end[1]), info['shape'][1]) == start + width
        assert np.all(stride == 1)

    per_layer = 2 if group and split else 16 if split else 1
    decodes, layer_ends, current = [], [], []
    for k, (key, args) in enumerate(convs):
        assert np.all(value(args[2]) == 1) and np.all(value(args[3]) == 1)
        assert int(value(args[4])) == 1
        j, i = k % per_layer, k // per_layer
        if group:
            wanted = np.array(weights[:, j * GROUP:(j + 1) * GROUP] if split else weights)
        else:
            wanted = np.array(weights[i, :, j * GROUP:(j + 1) * GROUP] if split
                              else weights[i]).reshape(512, GROUP if split else 512, 1, 1)
        wanted_scale = scales[:, j:j + 1] if group and split else scales

        if fmt == 'fp16':
            actual = value(args[1])
            assert actual.dtype == np.float16
        else:
            op, shift_scale = ops[args[1]]
            assert op == 'coreai.blockwise_shift_scale'
            scale = value(shift_scale[1])
            assert scale.tobytes() == wanted_scale.tobytes()
            assert not np.any(value(shift_scale[2])) and not np.any(value(shift_scale[3]))
            if fmt == 'w8a8':
                codes = value(shift_scale[0])
                assert codes.dtype == np.int8
            else:
                lut_op, lut = ops[shift_scale[0]]
                assert lut_op == 'coreai.lut_to_dense'
                assert constants[lut[0]][1].endswith('xui4')
                palette = value(lut[1])
                assert palette.shape == (1, 1, 1, 1, 16, 1)
                assert np.array_equal(palette.reshape(-1), np.arange(-8, 8, dtype=np.int8))
                codes = palette.reshape(-1)[value(lut[0])]
            assert np.array_equal(codes, wanted)
            expanded = np.repeat(scale, GROUP, axis=1) if group else scale
            actual = (codes.astype(np.float64) * expanded.astype(np.float64)).astype(np.float16)

        target_scale = np.repeat(wanted_scale, GROUP, axis=1) if group else wanted_scale
        target = (wanted.astype(np.float64) * target_scale.astype(np.float64)).astype(np.float16)
        assert actual.tobytes() == target.tobytes()

        if group:
            base = dqops[0][0]
        elif i == 0:
            base = '%arg0'
        elif fmt != 'fp16':
            base = dqops[i - 1][0]
        else:
            base = layer_ends[i - 1]
        if split:
            check_slice(args[0], base, j * GROUP, GROUP)
        else:
            assert args[0] == base

        current.append(key)
        if j == per_layer - 1:
            while len(current) > 1:                      # follow the balanced add tree
                level = []
                for z in range(0, len(current), 2):
                    matches = [k2 for k2, (op, a) in ops.items()
                               if op == 'coreai.decomposable.broadcasting_add'
                               and a == current[z:z + 2]]
                    assert len(matches) == 1
                    level.append(matches[0])
                current = level
            layer_ends.append(current[0])
            current = []
        decodes.append({'conv': k, 'shape': list(actual.shape),
                        'sha256': hashlib.sha256(actual.tobytes()).hexdigest()})

    if not multiply:
        assert final == layer_ends[-1]
        if group:
            assert ops[ops[dqops[0][0]][1][0]][1][0] == '%arg0'
        elif fmt != 'fp16':
            for i, (key, args) in enumerate(qops):
                assert args[0] == layer_ends[i]
        assert counts['coreai.decomposable.broadcasting_add'] == (
            30 if split and not group else 1 if split else 0)

    dump(out / 'asset-audit.json', {
        'passed': True,
        'method': 'persisted Core AI MLIR bytecode, version-specific internal reader, '
                  'SSA walk and independent payload decode',
        'asset_files': file_hashes(asset),
        'actual_op_counts': dict(counts),
        'weight_decodes': decodes,
        'checks': ['input/output signature', 'actual operation counts',
                   'weight operands/codes/scales/zero', 'QDQ SSA and parameters',
                   'chain and split reduction SSA', 'multiply DQ operands/scales'],
    })
    return names


def build(case_id, data, out, info):
    """Author, convert, save and audit one Core AI case."""
    torch.set_num_threads(1)
    if '-group-' in case_id:
        model = Group(np.load(data / 'group_codes.npy'),
                      np.load(data / 'group_scales.npy'),
                      np.load(data / 'group_input_scales.npy'),
                      case_id.endswith('split32'))
    elif '-qdq-' in case_id:
        model = Multiply(case_id.split('-')[-1])
    else:
        depth = int(case_id.split('-')[-1])
        model = Chain(np.load(data / 'weights.npy', mmap_mode='r')[:depth],
                      case_id.split('-')[1], 'split32' in case_id)
    model.eval()

    exported = torch.export.export(model, (torch.zeros(info['shape'], dtype=torch.float16),))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(exported, entrypoint_name=info['entrypoint'])
    program = converter.to_coreai()
    program.optimize()

    asset = out / 'model.aimodel'
    program.save_asset(asset)
    names = audit(asset, out, case_id, data, info)
    palettized = '-a8w4-' in case_id or '-group-' in case_id
    return {'model_path': str(asset), 'input_name': names[0], 'output_name': names[1],
            'representation': 'Core AI ' + ('LUT INT4 plus blockwise scales' if palettized
                                            else case_id.split('-')[1])}

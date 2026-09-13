"""No checkpoint, no weights: QDQ(1 * QDQ(1, 1/16), output_scale)."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import coreai_torch
import coreai_torch._compression.custom_layers  # Registers coreai quantize/dequantize.
from coreai.authoring import AIProgram


class Repro(nn.Module):
    def __init__(self, output_scale, clip):
        super().__init__()
        self.clip = clip
        self.register_buffer('si', torch.tensor(1 / 16, dtype=torch.float16))
        self.register_buffer('so', torch.tensor(output_scale, dtype=torch.float16))
        self.register_buffer('zero', torch.tensor(0, dtype=torch.int8))

    def qdq(self, x, s):
        q = torch.ops.coreai.quantize.default(x, s, torch.int8, self.zero, None, 1)
        return torch.ops.coreai.dequantize.default(q, s, self.zero, None, 1, torch.int8)

    def forward(self, x):
        a, b = x[:, :16], x[:, 16:]
        p = a * self.qdq(b, self.si)
        if self.clip:
            p = p.clamp(min=-128 * self.so, max=127 * self.so)
        return self.qdq(p, self.so)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    np.ones((1, 32, 1, 64), np.float16).tofile(out / 'input.raw')
    np.ones((1, 16, 1, 64), np.float16).tofile(out / 'expected.raw')
    assets = {}
    entries = {}
    for denominator in [16, 8, 4, 2]:
        for clip in [False, True]:
            arm = f's{denominator}' + ('_clip' if clip else '')
            folder = out / arm
            folder.mkdir()
            model = Repro(1 / denominator, clip).eval()
            ep = torch.export.export(model, (torch.ones((1, 32, 1, 64), dtype=torch.float16),)).run_decompositions(coreai_torch.get_decomp_table())
            entry = 'qdq_mul_scale_' + arm
            converter = coreai_torch.TorchConverter()
            converter.add_exported_program(ep, entrypoint_name=entry)
            program = converter.to_coreai()
            program.optimize()
            program.save_asset(folder / 'model.aimodel')
            (folder / 'graph.mlir').write_text(str(AIProgram._load_bytecode(folder / 'model.aimodel/main.mlirb')).split('\n{-#', 1)[0])
            assets[arm] = str(folder / 'model.aimodel')
            entries[arm] = entry
    config = {'assets': assets, 'entries': entries, 'input': str(out / 'input.raw'),
              'expected': {k: str(out / 'expected.raw') for k in assets},
              'pairs': 0, 'warmup': 0, 'repeats': 0, 'order': list(assets), 'probe': True,
              'inputChannels': 32, 'outputChannels': 16}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(out / 'config.json')


if __name__ == '__main__':
    main()

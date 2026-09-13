"""Named quantities from G1-W timings and imported numerical controls."""
from doc_claims import Quantity
from .evidence import BASE, ROOT, _pct, _x
from fractions import Fraction
import importlib.util
import json


def quantities(data):
    out = {}

    def add(name, value, source='timings.json.gz'):
        out['g1w.' + name.replace('_', '-').lower()] = Quantity(str(value), '', '', BASE + '/' + source)

    depth, density, codebook, e4b = (data[k] for k in ('depth', 'density', 'codebook', 'e4b'))
    for arm, value in depth['mean_p50_ms'].items():
        add('depth.' + arm + '.ms', f'{value:.2f}')
    for arm, value in depth['a8_over_w4a16'].items():
        add('depth.' + arm + '.speed', _x(value))
    for arm in ('w4a16', 'a8w4'):
        add('fit.' + arm + '.slope', f"{depth['fit'][arm]['ms_per_layer']:.3f}")
    add('fit.intercept-gap', f"{depth['fit']['a8w4']['ms_intercept'] - depth['fit']['w4a16']['ms_intercept']:.3f}")
    for arm, value in density['process_median_ms'].items():
        add('density.' + arm + '.ms', f'{value:.2f}')
    for arm, value in density['over_sparse'].items():
        add('density.' + arm + '.speed', _x(value))
    for weights, values in codebook.items():
        add('codebook.' + weights + '.fp16-ms', f"{values['fp16_ms']:.2f}")
        for metric in ('a8_over_fp16', 'a8_over_w4a16'):
            add('codebook.' + weights + '.' + metric, _x(values[metric]))
    for kind, value in data['scale'].items():
        add('scale.' + kind + '.speed', _x(value))
    zeros = data['fixtures']['density_weights']
    add('weights.zero-fraction', _pct(data['fixtures']['codebook_full_zero_fraction'], 2), 'evidence.json')
    add('weights.norm-ppm', f"{(zeros['fill_min']['row_norm_min'] / zeros['old']['row_norm_min'] - 1) * 1e6:.1f}", 'evidence.json')
    for arm, value in e4b['median_ms'].items():
        add('e4b.' + arm + '.ms', f'{value:.2f}')
    for arm, value in e4b['speed_vs_w4a16'].items():
        add('e4b.' + arm + '.speed', _x(value, 3))
    coarse = [e4b['speed_vs_w4a16'][a] for a in ('split_coarse_w4', 'split_coarse_a8')]
    add('e4b.coarse-speed-span', f'{min(coarse):.3f}–{max(coarse):.3f}×')
    add('e4b.coarse-calls', e4b['function_calls_per_pipeline']['split_coarse_a8'])
    for layers, key in ((1, 'e4b_single_layer'), (8, 'e4b_eight_layer')):
        for arm, row in data['numerics'][key].items():
            digits = 0 if arm == 'native' else (2 if arm == 'bare' or layers == 1 else 1)
            add(f'e4b.{layers}.{arm}.l2', _pct(row['ordinary'], digits), 'evidence.json')
    work = data['work']
    add('work.stack-Gops', f"{work['e4b_stack_ops'] / 1e9:.1f}")
    add('work.equivalent-layers', f"{work['synthetic_layers_equivalent']:.1f}")
    add('work.e4b-million-weights', f"{work['e4b_weights_per_mlp'] / 1e6:.1f}")
    add('work.synthetic-weights', f"{work['synthetic_weights_per_layer']:,}")
    add('measured-calls', f"{data['measured_calls']:,}")
    unit = data['numerics']['unit_scale_qdq']
    add('qdq.finite-values', f"{unit['finite_values']:,}", 'evidence.json')
    add('qdq.ties-even-mismatches', unit['unit_even_mismatches'], 'evidence.json')
    add('gelu.zero-output', f"{data['numerics']['gelu_zero_input']['gelu_tanh']['unique_values'][0]:.10f}".replace('-', '−'), 'evidence.json')
    repro = ROOT / 'findings/coreai-qdq-multiply-scale/repro'
    spec = importlib.util.spec_from_file_location('qdq_claim_evidence', repro / 'verify.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    outputs = verifier.verify()['arms']
    record = json.loads((repro / 'recorded/results.json').read_text())
    for arm, row in outputs.items():
        source = 'findings/coreai-qdq-multiply-scale/repro/recorded/' + arm + '/control-0.raw'
        for field in ('observed', 'reference'):
            value = row[field][0] if field == 'observed' else row[field]
            out[f'g1w.probe.{arm.replace("_", "-")}.{field}'] = Quantity(f'{value:g}', '', '', source)
        out[f'g1w.probe.{arm.replace("_", "-")}.scale'] = Quantity(
            str(Fraction(record['arms'][arm]['output_scale'])), '', '',
            'findings/coreai-qdq-multiply-scale/repro/recorded/results.json')
    return out

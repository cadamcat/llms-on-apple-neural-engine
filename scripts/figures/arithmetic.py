"""Arithmetic figures: the three questions behind "accuracy", and the QDQ controls.

The scale-control figure reads its numbers from the bundled smoke records, so
the table cannot drift from the evidence; the conceptual figure carries no
measured value at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from .canvas import Canvas, T_BODY, T_SMALL

LEFT, FULL = 40.0, 820.0


def levels():
    title = 'Three questions behind one word: accuracy'
    desc = ('Runtime compatibility compares a device output with a frozen arithmetic reference. '
            'Quantization deviation compares the quantized pipeline with an FP16 baseline. Model '
            'quality needs a language or task evaluation. Evidence at one level cannot stand in '
            'for the next.')
    c = Canvas(title, desc, 470)
    c.header(title, 'different references answer different questions, and none of them substitutes for the next')

    rows = [
        ('01 · Runtime compatibility', 'teal',
         'Does the declared quantized computation match what the runtime actually did?',
         'Observed directly in the synthetic fixtures; the scope is exactly the frozen reference.',
         'Device output', 'Frozen arithmetic reference'),
        ('02 · Quantization deviation', 'blue',
         'How far did quantization move the result away from the float baseline?',
         'A close reference match and a large FP16 difference can hold at the same time.',
         'Quantized output', 'FP16 baseline'),
        ('03 · Model or task quality', 'amber',
         'What happens to NLL, KL or any downstream task metric?',
         'Not established by these model-free chains. It needs a separate experiment.',
         'Quantized model', 'Language or task evaluation'),
    ]
    for i, (heading, color, question, boundary, left, right) in enumerate(rows):
        y = 88.0 + i * 106
        c.card(LEFT, y, FULL, 92, heading, [], color)
        c.text(LEFT + 16, y + 44, question, T_BODY, 'ink')
        c.text(LEFT + 16, y + 65, boundary, T_SMALL, 'muted')
        c.tint(520, y + 28, 148, 30, color, 6, stroke=False)
        c.text(594, y + 47, left, T_SMALL, color, 'sb', anchor='middle')
        c.text(680, y + 47, 'vs', T_SMALL, 'muted', anchor='middle')
        c.tint(692, y + 28, 168, 30, 'slate', 6, stroke=False)
        c.text(776, y + 47, right, T_SMALL, 'ink', anchor='middle')

    c.tint(LEFT, 406, FULL, 46, 'blue')
    c.text(58, 426, 'Choosing RNE or RZA changes the comparison, never the output that was recorded.',
           T_BODY, 'blue', 'sb')
    c.text(58, 443, 'Freeze a new hypothesis and its held-out inputs before it is used in a new device experiment.',
           T_SMALL, 'muted')
    return title, desc, c.finish()


def qdq_controls(repo: Path):
    source = 'results/fresh/smoke.json'
    runs = json.loads((repo / source).read_text())['runs']
    variants = [('equal', '2, 2', 'Standard QDQ'),
                ('unequal', '0.5, 2', 'Standard QDQ'),
                ('reverse_order', '0.5, 2', 'Reversed branch construction'),
                ('explicit_q', '0.5, 2', 'Explicit quantize expression')]
    values = []
    for variant, scales, expression in variants:
        i, run = next((i, r) for i, r in enumerate(runs) if r['case_id'] == 'coreai-qdq-' + variant)
        j, record = next((j, r) for j, r in enumerate(run['numerical']['comparisons'])
                         if r['phase'] == 'control_original')
        values.append({'case': run['case_id'], 'expression': expression, 'scales_label': scales,
                       'relative_L2': record['relative_L2'], 'max_abs': record['max_abs'],
                       'every_control_has_ANE': run['placement']['every_control_has_ANE'],
                       'source_pointer': f'/runs/{i}/numerical/comparisons/{j}',
                       'placement_pointer': f'/runs/{i}/placement/every_control_has_ANE'})

    title = 'Exact-grid controls separate a scale error from a rounding error'
    desc = ('For the exact inputs g equals 2 and u equals 8 the target product is 16, with no '
            'midpoint tie and no saturation anywhere. Equal-scale and explicit-quantize controls '
            'have zero error; unequal scales give a relative L2 of 0.75 and reversed construction '
            'gives 3. All four cases show ANE participation in every control, which does not make '
            'them numerically correct.')
    c = Canvas(title, desc, 434)
    c.header(title,
             'one multiplication, four ways of writing the same graph',
             'Core AI · Apple M5 Pro · fresh smoke suite · coreai-torch 0.4.1')

    c.tint(LEFT, 88, 250, 78, 'blue')
    c.text(58, 110, 'g = 2  →  QDQ(scale 0.5)', T_BODY, 'blue', 'sb')
    c.text(58, 130, 'u = 8  →  QDQ(scale 2)', T_BODY, 'blue', 'sb')
    c.text(58, 152, 'Both quantized codes are exactly 4.', T_SMALL, 'muted')
    c.arrow(298, 127, 322, 127)
    c.tint(332, 88, 188, 78, 'teal')
    c.text(426, 112, 'Expected product', T_SMALL, 'teal', 'sb', anchor='middle')
    c.text(426, 144, '2 × 8 = 16', 21, 'teal', 'b', anchor='middle')
    c.tint(542, 88, 318, 78, 'slate')
    c.text(560, 110, 'No midpoint ties, no saturation.', T_BODY, 'ink', 'sb')
    c.text(560, 130, 'RNE and RZA produce identical codes here.', T_SMALL, 'muted')
    c.text(560, 150, 'Equal-scale case: scales 2, 2 → codes 1, 4.', T_SMALL, 'muted')

    columns = [(58, 'Graph expression', 'start'), (392, 'Scales (g, u)', 'start'),
               (560, 'Relative L2', 'end'), (664, 'Max |error|', 'end'),
               (860, 'ANE in all four controls', 'end')]
    for x, label, anchor in columns:
        c.text(x, 202, label, T_SMALL, 'muted', 'sb', anchor=anchor)
    c.line(LEFT, 212, 860, 212, 'rule', 1)

    for k, value in enumerate(values):
        y = 222.0 + k * 36
        exact = value['relative_L2'] == 0
        c.tint(LEFT, y, FULL, 30, 'teal' if exact else 'amber', 6, stroke=False)
        cells = [value['expression'], value['scales_label'], f"{value['relative_L2']:g}",
                 f"{value['max_abs']:g}", 'yes' if value['every_control_has_ANE'] else 'no']
        for (x, _, anchor), cell in zip(columns, cells):
            emphasis = 'b' if anchor == 'end' and cell not in ('yes', 'no') else ''
            color = 'ink' if not emphasis else ('teal' if exact else 'amber')
            c.text(x, y + 20, cell, T_BODY, color, emphasis, anchor=anchor,
                   cls='num' if anchor == 'end' else '')

    c.text(LEFT, 396, 'Reversing the order changes how the two branches are built, not the multiplication they target.',
           T_BODY, 'ink', 'sb')
    c.footnote(414, 'Errors are shown for the original input; the ANE column covers all four '
                    'controls, original, zero, negative and repeat.')
    return title, desc, values, source, c.finish()


def dot_bracket(repo: Path):
    """The binary16 grid around one exact dot product, and what the ANE returned."""
    source = 'results/historical/dot-localization.json'
    leaf = json.loads((repo / source).read_text())['leaf']
    exact = leaf['exact_dot']
    prediction = leaf['model_prediction']       # correctly rounded, either tie rule
    actual = leaf['ane_actual']
    upper = prediction + (prediction - actual)  # the other end of the bracket
    ulp = upper - prediction

    title = 'Final rounding of the exact dot cannot explain this output'
    desc = ('The binary16 grid around one exact 32-term dot product. The exact value lies '
            'between two representable numbers, so any correctly rounded result must be one '
            'of those two. The recorded device output is a third value, one unit in the last '
            'place beyond the lower one. This excludes final rounding alone, not intermediate product or accumulation effects.')
    c = Canvas(title, desc, 322)
    c.header(title,
             'one 32-term dot product from a real gate · the binary16 grid, zoomed to four ULP',
             'Qwen8B first gate · output channel 11793, token 22 · input channels 2912–2943')

    x0, x1, axis = 120.0, 800.0, 162.0
    place = lambda v: x0 + (v - (prediction - 2 * ulp)) / (4 * ulp) * (x1 - x0)
    left, right = place(prediction), place(upper)

    # The bracket: every correctly rounded answer lands on one of its two ends.
    c.text((left + right) / 2, 128, 'any correctly rounded result is one of these two',
           T_SMALL, 'teal', 'sb', anchor='middle')
    c.tint(left, 140, right - left, 44, 'teal', 6, stroke=False)

    c.line(x0, axis, x1, axis, 'rule', 1)
    for step in (-2, -1, 0, 1, 2):
        x = place(prediction + step * ulp)
        c.line(x, axis - 7, x, axis + 7, 'muted', 1)
    c.text(x1, axis + 26, 'representable binary16 values, 1 ULP apart', T_SMALL, 'muted',
           anchor='end')

    for value, color, label in ((actual, 'red', '✗  what the ANE returned'),
                                (prediction, 'teal', '✓  correctly rounded')):
        x = place(value)
        c.dot(x, axis, 7, color)
        c.line(x, axis + 9, x, axis + 34, color, 1)
        c.text(x, axis + 50, label, T_BODY, color, 'b', anchor='middle')
        c.text(x, axis + 66, repr(value), T_SMALL, 'muted', anchor='middle', cls='num')

    # Exact and its rounding differ by 0.04 ULP, so they are one point at this zoom.
    offset = (exact - prediction) / ulp
    c.text(40, 268, f'The exact dot product is {exact:.17g}, which sits {offset:.2f} ULP '
                    f'above the lower end — indistinguishable from it here, and nowhere '
                    f'near a midpoint.', T_SMALL, 'ink')
    c.footnote(292, 'Final rounding alone is ruled out; intermediate products and accumulation remain unobserved. '
                    'Check the 32 published terms.')
    return title, desc, source, c.finish()


def generate(repo: Path, out: Path) -> list[dict]:
    """Write both arithmetic figures and return their metadata."""
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    title, description, content = levels()
    (out / 'arithmetic-levels.svg').write_text(content, encoding='utf-8')
    metadata = [{'filename': 'arithmetic-levels.svg', 'title': title, 'description': description,
                 'sources': ['articles/03-arithmetic-compatibility.md', 'docs/METHODS.md'],
                 'semantic_checks': ['Three distinct comparisons',
                                     'A conceptual distinction, not a measured causal path',
                                     'Changing a reference never changes a recorded output']}]

    title, description, values, source, content = qdq_controls(repo)
    (out / 'qdq-scale-controls.svg').write_text(content, encoding='utf-8')
    metadata.append({'filename': 'qdq-scale-controls.svg', 'title': title, 'description': description,
                     'sources': [source, 'docs/METHODS.md',
                                 'articles/03-arithmetic-compatibility.md'],
                     'values': values,
                     'semantic_checks': ['Exact grid, so no midpoint tie explains the error',
                                         'Error column uses the original input',
                                         'ANE column covers all four controls',
                                         'ANE participation does not imply a correct output']})

    title, description, source, content = dot_bracket(repo)
    (out / 'dot-outside-bracket.svg').write_text(content, encoding='utf-8')
    metadata.append({'filename': 'dot-outside-bracket.svg', 'title': title,
                     'description': description,
                     'sources': [source, 'findings/fp16-dot-residual/README.md'],
                     'semantic_checks': [
                         'Exact value lies strictly between two binary16 values',
                         'The model prediction is the correctly rounded one',
                         'The recorded device output is neither',
                         'The exact value is not at a midpoint',
                         'Recomputable from published scalars without a device']})
    return metadata

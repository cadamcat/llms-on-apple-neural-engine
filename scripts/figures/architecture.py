"""Explanatory diagrams for the method, the split ablation and the CLI workflow.

These describe documented procedure rather than measured values, so they carry
no numbers that are not also stated in the sources listed with each figure.
"""

from __future__ import annotations

from pathlib import Path

from .canvas import Canvas, T_BODY, T_SMALL

LEFT, RIGHT, FULL = 40.0, 860.0, 820.0
COL_W = 390.0
COL2_X = 470.0


def evidence_pipeline():
    title = 'From a frozen workload to an admitted measurement'
    desc = ('Freeze references, audit the persisted asset, then check numerical controls, '
            'resource limits and actual ANE participation. Only after all of that is a '
            'synchronous call timed. A failed admission is preserved as a finding with no '
            'benchmark timing.')
    c = Canvas(title, desc, 700)
    c.header(title, 'each layer answers a different question, and none of them substitutes for the next')

    c.card(LEFT, 88, FULL, 72, '1 · Freeze the workload and its references',
           ['Fix shapes, weights, inputs, scales and rounding rules before any output is observed.',
            'Keep separate references wherever graph reduction or QDQ boundaries differ.'])
    c.arrow(450, 160, 450, 176)
    c.card(LEFT, 176, FULL, 72, '2 · Export, reload and audit the persisted asset',
           ['Follow graph operands; decode weights, scales and zero points independently.',
            'Record source and asset hashes. An unrecognised representation fails the audit.'])

    c.path('M450 248 v14 H171 M450 262 H729', 'muted', 1.4)
    for x in (171, 450, 729):
        c.arrow(x, 262, x, 282)
    for x, heading, lines in (
        (LEFT, 'Numerical controls',
         ['Original · zero · negative', 'Repeated original is byte-identical',
          'Frozen reference plus hashes', 'Finite and stable outputs']),
        (319, 'Resource boundaries',
         ['Guard the owned process tree', 'Memory, disk and deadlines',
          'Serial suite behind a host lock', 'Every check is recorded']),
        (598, 'Actual device evidence',
         ['ANE events in all four windows', 'Core ML: convolutions prefer ANE',
          'Core AI: loaded function and', 'requested specialization']),
    ):
        c.card(x, 282, 262, 104, heading, lines, 'teal')

    c.path('M171 386 v14 H729 v-14 M450 386 v14', 'muted', 1.4)
    c.arrow(300, 400, 300, 422)
    c.arrow(700, 400, 700, 422)
    c.card(LEFT, 422, 500, 86, '3 · Admit the synchronous timed call',
           ['Only once the required evidence has passed.',
            '10 warmups, then 30 measured calls per process.',
            'Three fresh processes per arm; report p50 and p95.'], 'teal')
    c.card(570, 422, 290, 86, 'A failure stays a finding',
           ['Wrong outputs, CPU choices and', 'errors are all preserved.',
            'They are never given a benchmark time.'], 'red')
    c.arrow(450, 508, 450, 530)
    c.card(LEFT, 530, FULL, 72, '4 · Preserve the raw records, verify, then select metrics',
           ['Retain controls, output hashes, duration rows and source identity.',
            'Copying, checking, hashing, writing and log capture all sit outside the timer.'])

    c.tint(LEFT, 626, FULL, 46, 'amber')
    c.text(58, 646, 'ANE participation is not exclusive placement, and neither is a physical INT8 instruction count.',
           T_BODY, 'amber', 'sb')
    c.text(58, 663, 'A synthetic graph rate says nothing on its own about model quality, token throughput or energy.',
           T_SMALL, 'muted')
    return title, desc, c.finish()


def split_structure():
    title = 'Same K512 weights, different graph boundaries'
    desc = ('A two-layer Core AI A8W4 ablation. The wide path runs one K512 convolution per '
            'layer; the split path runs sixteen contiguous K32 convolutions and a balanced '
            'fifteen-add FP16 tree per layer. One A8 QDQ separates the layers. The counts '
            'describe graph operations, not host calls.')
    c = Canvas(title, desc, 726)
    c.header(title, 'Core AI A8W4 · input [1, 512, 64, 64] · two layers · identical initial weight values')

    c.card(LEFT, 86, COL_W, 72, 'Wide path',
           ['W is 512 × 512 and every K512 input is read at once.',
            'One convolution produces the FP16 layer output.'], 'teal')
    c.card(COL2_X, 86, COL_W, 72, 'Split path',
           ['The same W, partitioned along the input dimension K.',
            'Sixteen contiguous slices, each of width K32.'], 'amber')
    c.arrow(235, 158, 235, 178)
    c.arrow(665, 158, 665, 178)

    c.card(LEFT, 178, COL_W, 86, 'One K512 convolution',
           ['One dot product per output and position.', 'A single FP16 boundary at the layer output.',
            'No partial-sum addition tree at all.'], 'teal')
    c.card(COL2_X, 178, COL_W, 86, 'Sixteen K32 convolutions',
           ['Each one produces all 512 output channels.', 'An FP16 boundary on every partial result.',
            'The same synthetic weight scale in each piece.'], 'amber')
    c.arrow(665, 264, 665, 284)
    c.card(COL2_X, 284, COL_W, 72, 'Balanced FP16 addition tree',
           ['Sixteen partials reduce 16 → 8 → 4 → 2 → 1.',
            'Fifteen adds, with an FP16 boundary at each level.'], 'amber')
    c.path('M235 264 v112', 'muted', 1.4, arrow=True)
    c.arrow(665, 356, 665, 376)

    for x, color in ((LEFT, 'teal'), (COL2_X, 'amber')):
        c.card(x, 376, COL_W, 72, 'One inter-layer A8 QDQ',
               ['Q8 rounding to nearest, ties away from zero.',
                'Scale 0.125, then dequantize back to FP16.'], color)
    c.arrow(235, 448, 235, 468)
    c.arrow(665, 448, 665, 468)

    c.card(LEFT, 468, COL_W, 72, 'Repeat for layer 2',
           ['Two-layer total: 2 convolutions and 1 QDQ.',
            'FP16 output, with no final QDQ.'], 'teal')
    c.card(COL2_X, 468, COL_W, 72, 'Repeat for layer 2',
           ['Two-layer total: 32 convolutions and 30 adds.',
            'One QDQ, FP16 output, no final QDQ.'], 'amber')

    c.text(LEFT, 574, 'One host prediction executes either complete two-layer graph.', T_BODY, 'ink', 'sb')
    c.text(LEFT, 591, 'The two paths therefore need separate frozen references: split partial and '
                      'reduction FP16 midpoint ties use RZA.', T_SMALL, 'muted')
    c.card(LEFT, 610, FULL, 72, 'A separate fixture: the K64 compatibility probe',
           ['64 output channels, two K32 groups with distinct scales, identity input and an input A8 QDQ.',
            'Its native-versus-split comparison is a different experiment from the K512 ablation above.'],
           'blue')
    return title, desc, c.finish()


def reproduction_workflow():
    title = 'Reproduce a run, then verify and select its evidence'
    desc = ('The public CLI prepares a locked environment, inspects it, runs one suite serially '
            'behind a resource guard into a new output directory, verifies hashes and source '
            'identity, then reports a selection for review. Fresh and historical evidence stay '
            'in separate trees.')
    c = Canvas(title, desc, 700)
    c.header(title, 'no model download · a new output directory for every run · every stage is separable')

    steps = [
        (88, 72, '1 · Prepare the locked environment', 'uv sync --locked --extra apple',
         ['Add --offline against a populated cache. Record host, OS, SDK, dependencies and source.'], 'blue'),
        (176, 72, '2 · Inspect before running', 'ane-scope doctor  ·  ane-scope list',
         ['Confirm what the environment offers, then pick smoke, compatibility, throughput or split.'], 'blue'),
        (264, 90, '3 · Execute through the guard', 'ane-scope run --suite smoke --output runs/my-smoke',
         ['Serial execution behind a host lock, with sampled RSS, pressure, disk, swap and time limits.',
          'Export → asset audit → controls → admitted timing, and timing only for performance suites.'], 'teal'),
        (370, 72, '4 · Verify the run and its source identity', 'ane-scope verify runs/my-smoke',
         ['Recheck hashes, references, placement windows and statistics; read source_matches_run.'], 'teal'),
        (458, 72, '5 · Report, inspect, then copy', 'ane-scope report runs/my-smoke',
         ['A report plus public-results.json. Inspect the selection before it reaches results/fresh/.'], 'blue'),
    ]
    for y, h, heading, command, lines, color in steps:
        c.card(LEFT, y, FULL, h, heading, [], color)
        c.text(LEFT + 16, y + 41, command, T_BODY, 'ink', 'sb', cls='mono')
        for i, line in enumerate(lines):
            c.text(LEFT + 16, y + 60 + i * 16, line, T_SMALL, 'muted')
        if y < 458:
            c.arrow(450, y + h, 450, y + h + 16)

    c.arrow(450, 530, 450, 550)
    c.card(LEFT, 550, COL_W, 72, 'Fresh evidence',
           ['Selected records live in results/fresh/.',
            'Raw artifacts stay in the ignored runs/ tree.'], 'teal')
    c.card(COL2_X, 550, COL_W, 72, 'Historical imports stay apart',
           ['Existing evidence lives in results/historical/.',
            'Original provenance and hashes are preserved.'], 'amber')
    c.footnote(654, 'Offline tests and table recalculation check arithmetic and identity. '
                    'They are not a fresh device run.')
    return title, desc, c.finish()


FIGURES = [
    ('evidence-pipeline.svg', evidence_pipeline,
     ['docs/METHODS.md', 'articles/01-measuring-ane-performance.md'],
     ['Frozen reference before observation',
      'All admission evidence precedes timing',
      'Failed admission has no timing',
      'Device participation does not imply physical INT8']),
    ('split-structure.svg', split_structure,
     ['docs/METHODS.md', 'articles/02-group-quantization-and-split.md',
      'src/ane_scope/_coreai.py', 'src/ane_scope/references/prepare.py'],
     ['Per-layer 16 K32 conv and 15 FP16 adds',
      'Two-layer 32 conv and 30 adds',
      'Exactly one inter-layer QDQ',
      'Separate frozen references per path',
      'K64 compatibility separate from K512 performance']),
    ('reproduction-workflow.svg', reproduction_workflow,
     ['docs/REPRODUCING.md', 'docs/PROVENANCE.md'],
     ['Guarded CLI only',
      'New run directory for every suite',
      'Verify source identity',
      'Inspect before copying a fresh selection',
      'Historical and fresh evidence stay separate']),
]


def generate(repo: Path, out: Path) -> list[dict]:
    """Write the diagrams and return their source and semantic metadata."""
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = []
    for filename, draw, sources, checks in FIGURES:
        for source in sources:
            if not (repo / source).is_file():
                raise FileNotFoundError(source)
        title, description, content = draw()
        (out / filename).write_text(content, encoding='utf-8')
        metadata.append({'filename': filename, 'title': title, 'description': description,
                         'sources': sources, 'semantic_checks': checks})
    return metadata

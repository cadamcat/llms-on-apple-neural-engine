"""Import the one-layer FP16/W4 load-time placement pair into the finding's recorded evidence; no device."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('import_g4a', HERE / 'import_g4a.py')
g4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4a)

PLACEMENT = 'workplans/g6-night-20260913/w4-placement'
POINTER = re.compile(r'0x[0-9a-f]{8,}')
KEEP = ('Start of mlir<anec> Validation', 'End of mlir<anec> Validation', '[proxy compileModel:...] returned',
        '[proxy loadModel:...] returned', 'ANE compilation failed', 'preferredDevice=')


def log_record(reader, path):
    counts = dict(ane_direct_request_success_rows=0, metal_shader_compile_rows=0)
    lines = []
    for line in reader.raw(path).decode(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        message = row.get('eventMessage', '')
        counts['ane_direct_request_success_rows'] += 'ANEProgramProcessRequestDirect() status=0x0' in message
        counts['metal_shader_compile_rows'] += 'Metal Compiling Shader' in message
        if any(k in message for k in KEEP):
            text = POINTER.sub('0x…', message)
            text = re.sub(r'^/AppleInternal/\S*?/Sources/', '', text)
            lines.append({'subsystem': row.get('subsystem'), 'message': text})
    return dict(counts, key_lines=lines)


def extract(workspace, output):
    reader = g4a.Reader(workspace)
    writer = g4a.Writer(reader, output)
    base = Path(PLACEMENT)
    pair = {}
    for name in ('fp16', 'w4'):
        export = reader.json(base / f'assets/l1-c1280-{name}/G6-Q1-EXPORT.json')
        probe = reader.json(base / f'l1-c1280-{name}-r1/PROBE.json')
        if probe['status'] != 'closed':
            raise ValueError('probe_not_closed:' + name)
        results = {}
        for query, result in probe['results'].items():
            results[query] = {k: result.get(k) for k in ('passed', 'error', 'ids', 'graphs', 'final_KV', 'same_ids_as_first_query',
                                                         'same_history_vs_first_query')}
            results[query]['host_returncode'] = result['host_closed']['returncode'] if result.get('host_closed') else None
        pair[name] = {'export': {k: export[k] for k in ('layers', 'capacities', 'max_context_length', 'compression', 'queries',
                                                       'trace_query', 'environment', 'source_revision')} | {'functions': sorted(export['functions'])},
                      'same_host': probe['same_host'], 'queries_in_order': probe['queries'], 'input': probe['input'],
                      'log_from_before_host_start': log_record(reader, base / f'{name}-load-log.ndjson'), 'results': results}
    writer.dump('observations.json', {
        'pair': pair,
        'procedure': 'Fresh bundles from the same exporter, one host per bundle; the unified log was streamed by process name from '
                     'three seconds before the host started until three seconds after it closed. Each host ran two greedy decode '
                     'steps with query 8, then query 4, capturing logits.',
        'scope': 'One M5 Pro, macOS 27.0 (26A428), coreai-models 7304c47, coreai-torch 0.4.2. Log lines are filtered to compile, '
                 'load, validation and delegate messages; pointer values and build paths are removed.'})
    products = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(output).iterdir())}
    writer.dump('provenance.json', {'importer': 'results/historical/import_palettized_placement.py', 'sources': reader.sources,
                                    'products': products, 'transformations': [
        'Keep each bundle export record and probe query result; drop logits paths and host argv.',
        'Count direct ANE request successes and Metal shader compilations in each process-name log stream.',
        'Keep validation, compile, load, ANE-compilation-failure and delegate-option lines with pointers and build paths removed.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

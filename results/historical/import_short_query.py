"""Import the decode-query-width diagnosis (Q1/Q2/Q4/Q8 on ANE) into the finding's recorded evidence; no device."""
import argparse
import difflib
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('import_g4a', HERE / 'import_g4a.py')
g4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4a)

DIAGNOSIS = 'workplans/g6-night-20260913/q1-diagnosis'
G4_TIERS = 'workplans/g4-cplus256-20260912'
ENGINE = 'swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAIStaticShapeEngine.swift'
HOST = 'swift/Sources/Tools/g3-flow-host/G3Host.swift'
# name, asset, what the case varies
CASES = [
    ('c1280-n1024-r1', f'{G4_TIERS}/tiers/qwen3-4b-fp16-ane-g4-c1280', '36 layers, G4 A tier; 1,024-token input; fresh host per query'),
    ('l1-c768-q1-8-64-t8-r2', f'{DIAGNOSIS}/assets/l1-c768-q1-8-64-t8', '1 layer; functions for queries 1, 8 and 64'),
    ('l1-c768-q1-64-t8-r1', f'{DIAGNOSIS}/assets/l1-c768-q1-64-t8', '1 layer; no query-8 function in the bundle'),
    ('l1-c768-q12483-t8-q8-q4', f'{DIAGNOSIS}/assets/l1-c768-q1-2-4-8-64-t8', '1 layer; functions for queries 1, 2, 4, 8 and 64'),
    ('l1-c768-q12483-t8-q8-q2', f'{DIAGNOSIS}/assets/l1-c768-q1-2-4-8-64-t8', '1 layer; functions for queries 1, 2, 4, 8 and 64'),
]
STATUS = re.compile(r'(ANE_ProgramSendRequest:\d+ status=[0-9a-fx]+|ANEProgramProcessRequestDirect\(\) Failed with status=0x[0-9a-f]+ : statusType=0x[0-9a-f]+)')
INFERENCE = re.compile(r'Code=(-?\d+) "ANE inference operation failed due to unknown error\. Model: 0x[0-9a-f]+, model UUID: [0-9A-F-]+, inferenceCount: (\d+), failureCount: (\d+)\.')


def log_record(reader, path):
    ok, statuses, functions = 0, [], []
    for line in reader.raw(path).decode(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        message = json.loads(line).get('eventMessage', '')
        ok += 'ANEProgramProcessRequestDirect() status=0x0' in message
        statuses += STATUS.findall(message)
        if message.startswith("Loading function '"):
            functions.append(message.split("'")[1])
    return {'ane_direct_request_success_rows': ok, 'failure_status_lines': statuses, 'loaded_functions': functions}


def extract(workspace, output):
    reader = g4a.Reader(workspace)
    writer = g4a.Writer(reader, output)
    diagnosis = Path(DIAGNOSIS)
    cases = []
    for name, asset, varies in CASES:
        root = diagnosis / name
        probe = reader.json(root / 'PROBE.json')
        if probe['status'] != 'closed':
            raise ValueError('probe_not_closed:' + name)
        asset_path = Path(asset)
        record = reader.json(asset_path / 'G6-Q1-EXPORT.json') if (reader.workspace / asset_path / 'G6-Q1-EXPORT.json').exists() else None
        if record is None:
            tier = reader.json(asset_path / 'TIER.json')
            signatures, layers, trace = tier['functions'], 36, 8
            environment = None
        else:
            signatures, layers, trace = record['functions'], record['layers'], record['trace_query']
            environment = record['environment']
        functions = sorted(signatures)
        decode_signatures = {f: signatures[f] for f in functions if f.startswith(f"extend_{probe['capacity']}_")}
        queries = {}
        for query, result in probe['results'].items():
            entry = {k: result.get(k) for k in ('passed', 'error', 'ids', 'graphs', 'final_KV')}
            entry['host_returncode'] = result['host_closed']['returncode']
            if 'same_history_vs_first_query' in result:
                entry['same_ids_as_first_query'] = result['same_ids_as_first_query']
                entry['same_history_vs_first_query'] = result['same_history_vs_first_query']
            entry['log'] = log_record(reader, root / f'q{query}-log.ndjson')
            stderr = reader.raw(root / f'q{query}/stderr.log').decode(errors='replace')
            match = INFERENCE.search(stderr)
            entry['mpsgraph_assertion'] = None if match is None else {
                'error_code': int(match.group(1)), 'inference_count': int(match.group(2)), 'failure_count': int(match.group(3))}
            queries[query] = entry
        cases.append({'case': name, 'varies': varies, 'asset': {'layers': layers, 'trace_query': trace, 'functions': functions,
                      'decode_signatures': decode_signatures,
                      'capacity': probe['capacity']}, 'input': probe['input'], 'input_tokens': probe['input_tokens'],
                      'queries_in_order': probe['queries'], 'decode_steps_requested': probe['outputs'] - 1,
                      'host': 'diagnostic host, query allow-list [1, 2, 4, 8]' if probe.get('host') else 'G4 A host, query allow-list [1, 8]',
                      'environment': environment, 'results': queries})
    trace_log = reader.raw(diagnosis / 'assets/l1-c768-q1-8-64-t1.log').decode(errors='replace')
    violation = [l.strip() for l in trace_log.splitlines() if 'specialized it to be a constant (1)' in l]
    if not violation:
        raise ValueError('trace_query_1_violation_missing')
    changes = {}
    for relative in (HOST, ENGINE):
        new = reader.raw(diagnosis / 'host-src' / relative).decode().splitlines()
        old = reader.raw(Path(G4_TIERS) / 'vendor/coreai-models' / relative).decode().splitlines()
        changes[relative] = [l for l in difflib.unified_diff(old, new, lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    writer.dump('observations.json', {
        'cases': cases,
        'trace_query_1_export': {'command_options': '--queries 1 8 64 --trace-query 1 --layers 1 --capacities 768',
                                 'torch_export_error': violation[0]},
        'host_query_allow_list_diff': changes,
        'upstream_default_query_lengths': [8, 16, 64],
        'scope': 'One M5 Pro, macOS 27.0 (26A428), coreai-models 7304c47, coreai-torch 0.4.2. Unified-log rows are filtered to each host PID after the host reported ready; only counts, status lines and loaded function names are kept.'})
    writer.dump('provenance.json', {'importer': 'results/historical/import_short_query.py', 'sources': sorted(reader.sources),
                                    'transformations': [
        'Keep each probe query result, host return code and same-history comparison; drop logits paths and the host argv.',
        'Reduce each host unified log to direct ANE request successes, failure status lines and loaded function names.',
        'Parse the MPSGraph assertion error code, inference count and failure count from each host stderr.',
        'Keep the torch.export constraint violation line from the query-1 trace export log.',
        'Keep the two-line query allow-list diff between the G4 A host sources and the diagnostic host sources.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.workspace, args.output)

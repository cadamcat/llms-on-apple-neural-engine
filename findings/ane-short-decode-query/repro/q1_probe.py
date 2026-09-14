"""G6 Q1 diagnosis: one capacity, fresh host per query, greedy decode with captured logits.

Adapted from g4-recovery b_alignment_probe.py, whose forced IDs plus captured logits
violate the host request contract (it never reached the device). Q8 runs first; Q1 is
compared under the same greedy history. The target PID's unified log is kept so an ANE
request failure can be located.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'workplans/g4-cplus256-20260912'))
sys.path.insert(0, str(ROOT / 'workplans/g4-ready-20260913'))
from flow_host import Host, OwnedProcess
from launch import acquire_run_lock
from g4 import save

REFERENCES = {'reading_1024': 'reading_1024.npy', 'quality_short': 'quality_short.npy'}


def run(out, capacity, input_name, queries, steps, asset=None, host_binary=None, truncated=False, same_host=False, reference_dir=None):
    import numpy as np
    from g3_4b_validate import compare, read_logits
    config = json.loads((ROOT / 'workplans/g4-ready-20260913/CONFIG.json').read_text())
    config.update(host=str(host_binary) if host_binary else config['hosts']['ane'], capacity=capacity,
                  assets={'ane': str(asset) if asset else config['ane_assets'][str(capacity)]})
    inputs = json.loads(Path(config['inputs']).read_text())
    reference = Path(reference_dir or ROOT / 'workplans/g3-night-20260912-energy-r5/reference') / REFERENCES[input_name]
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', host=config['host'], asset=str(asset) if asset else None, truncated=truncated, same_host=same_host, reference=str(reference), capacity=capacity, input=input_name, input_tokens=len(inputs[input_name]),
                  queries=list(queries), outputs=steps, formal_benchmark=False, results={})
    baseline = None; shared = None
    try:
        for query in queries:
            entry = {'passed': False}; report['results'][str(query)] = entry
            host = logger = stream = None
            last = query == queries[-1]
            try:
                if shared:
                    host, logger, stream = shared
                else:
                    tag = 'switch' if same_host else f'q{query}'
                    host = Host('ane', out / tag, config, inputs)
                    stream = (out / f'{tag}-log.ndjson').open('wb')
                    logger = OwnedProcess(['/usr/bin/log', 'stream', '--predicate',
                        f'processIdentifier == {host.proc.target_pid}', '--level', 'debug', '--style', 'ndjson'],
                        out / f'{tag}-log-OWNED.json', stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    time.sleep(1)
                    if same_host: shared = (host, logger, stream)
                identity = f'probe-q{query}'
                # Quality mode exports a 36-layer KV summary; truncated assets use full mode with exported tokens.
                row, stats = host.request(identity, input_name, mode='full' if truncated else 'quality',
                    outputs=steps, capture_steps=list(range(steps)), export_tokens=True,
                    graph_capacity=capacity, decode_query=query)
                arrays = [read_logits(host.out / 'host' / identity / f'logits-{i:04d}.f16') for i in range(steps)]
                entry.update(protocol=stats, final_KV=row['final_KV_count'], ids=row['generated_ids'],
                             graphs=sorted({g.get('graph') for g in row['graphs'] if g.get('graph')}))
                entry['first_output_reference'] = (compare(np.load(reference), arrays[0], config['quality'])
                    if not truncated else {'passed': True, 'skipped': 'truncated diagnostic asset'})
                if query == queries[0]:
                    baseline = (row['generated_ids'], arrays)
                else:
                    entry['same_ids_as_first_query'] = row['generated_ids'] == baseline[0]
                    entry['same_history_vs_first_query'] = [compare(a, b, config['quality'])
                                                            for a, b in zip(baseline[1], arrays, strict=True)]
                entry['passed'] = entry['first_output_reference']['passed'] and (
                    query == queries[0] or (entry['same_ids_as_first_query'] and
                                            all(m['passed'] for m in entry['same_history_vs_first_query'])))
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}:{exc}'; last = True
            finally:
                if last or not same_host:
                    if logger:
                        time.sleep(.5); logger.terminate(); logger.wait(timeout=14)
                    if stream: stream.close()
                    if host:
                        host.close(); entry['host_closed'] = json.loads((host.out / 'CLOSED.json').read_text())
                    shared = None
                save(out / 'PROBE.json', report)
            if 'error' in entry: break
    finally:
        report.update(status='closed', passed=all(r['passed'] for r in report['results'].values())
                      and len(report['results']) == len(queries), closed_ns=time.time_ns())
        save(out / 'PROBE.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--capacity', type=int, default=1280)
    p.add_argument('--input', default='reading_1024', choices=sorted(REFERENCES))
    p.add_argument('--queries', type=int, nargs='+', default=[8, 1])
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--asset', type=Path, help='diagnostic bundle; skips the full-model reference')
    p.add_argument('--host', type=Path, help='diagnostic host binary')
    p.add_argument('--tier-asset', type=Path, help='full-model bundle; keeps the independent reference')
    p.add_argument('--same-host', action='store_true', help='switch queries inside one host session')
    p.add_argument('--reference-dir', type=Path, help='directory with <input>.npy first-output references')
    a = p.parse_args()
    with acquire_run_lock():
        if a.asset and a.tier_asset: p.error('--asset and --tier-asset are exclusive')
        run(a.out, a.capacity, a.input, a.queries, a.steps, a.asset or a.tier_asset, a.host, truncated=a.asset is not None, same_host=a.same_host, reference_dir=a.reference_dir)

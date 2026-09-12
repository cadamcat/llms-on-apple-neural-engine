"""Recompute G3 curves and energy from the portable request and power fields."""
from collections import Counter
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from .protocol import require, validate_request
from .power import decode_power, energy_window, response

ROOT = Path(__file__).resolve().parents[2]
BASE = 'results/historical/g3-qwen3-4b'


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def model_work(structure):
    """Parameter and byte counts behind the implied rates."""
    config = structure['config']
    counts = {name: math.prod(value['shape']) for name, value in structure['tensors'].items()}
    embedding = counts['model.embed_tokens.weight']
    projection = sum(n for name, n in counts.items()
                     if 'norm' not in name and name != 'model.embed_tokens.weight')
    total = sum(counts.values())
    require(config['tie_word_embeddings'] and 'lm_head.weight' not in counts, 'g3_tied_output_head')
    require(2 * total == structure['source_total_bytes'], 'g3_weight_bytes')
    return {'projection_parameters': projection, 'embedding_parameters': embedding,
            'total_parameters': total, 'weight_bytes': 2 * total,
            'flops_per_token': 2 * projection,
            'attention_flops_per_token_per_context':
                4 * config['num_attention_heads'] * config['head_dim'] * config['num_hidden_layers'],
            'kv_bytes_per_token': 2 * 2 * config['num_hidden_layers'] *
                config['num_key_value_heads'] * config['head_dim']}


def implied(work, pairs):
    """Useful-work rates: prefill FLOP/s over projection weights, decode bytes read per step."""
    rows = []
    for pair in pairs:
        n = pair['context']
        for arm in ('ane', 'gpu'):
            rate = pair[arm]['rate']
            if pair['mode'] == 'prefill':
                rows.append({'mode': 'prefill', 'context': n, 'arm': arm,
                             'tera_flops_per_second': rate * work['flops_per_token'] / 1e12,
                             'attention_tera_flops_per_second': rate * n / 2 *
                                 work['attention_flops_per_token_per_context'] / 1e12})
            else:
                steps = pair[arm]['tokens']
                # Existing KV before each forward: n, n+1, ..., n+steps-1.
                # Current-token writes and fixed-graph padding are outside this model.
                kv_positions = steps * n + steps * (steps - 1) // 2
                block_bytes = steps * work['weight_bytes'] + kv_positions * work['kv_bytes_per_token']
                step = block_bytes / steps
                rows.append({'mode': 'decode', 'context': n, 'arm': arm,
                             'mean_prior_KV': kv_positions / steps,
                             'block_bytes': block_bytes, 'step_bytes': step,
                             'giga_bytes_per_second': rate * step / 1e9})
    return rows


def synthetic_reference(repo):
    """Median source-equivalent rate of the same-machine Core AI FP16 chain."""
    runs = json.loads((Path(repo) / 'results/fresh/throughput.json').read_text())['runs']
    return statistics.median(r['source_ops'] / (r['p50_ms'] * 1e9) for r in runs
                             if r['case_id'] == 'coreai-fp16-128' and r['p50_ms'])


def derive(bundle=None):
    bundle = Path(bundle) if bundle is not None else ROOT / BASE
    provenance = json.loads((bundle / 'provenance.json').read_text())
    for name, digest in provenance['products'].items():
        require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest,
                'g3_product_identity:' + name)
    protocol = json.loads((bundle / 'protocol.json').read_text())
    structure = json.loads((bundle / 'model-structure.json').read_text())
    inputs = json.loads((bundle / 'inputs.json').read_text())
    identity = json.loads((bundle / 'input-identity.json').read_text())
    lengths = protocol['lengths']
    require(set(inputs) == {f'reading_{n}' for n in lengths} | {'quality_short'}, 'g3_inputs')
    for name, record in identity['inputs'].items():
        ids = inputs[name]
        packed = b''.join(i.to_bytes(4, 'little', signed=True) for i in ids)
        require(len(ids) == record['tokens'] and
                hashlib.sha256(packed).hexdigest() == record['sha256_int32_le'], 'g3_input_identity:' + name)
    for run, config in protocol['runs'].items():
        require(all(config[k] == protocol[k] for k in config), 'g3_model_identity:' + run)
    requests = {}
    carry = {}
    coverage = []
    for item in read_rows(bundle / 'requests.jsonl.gz'):
        row, command = item['result'], item['command']
        run, host = item['run'], (item['run'], item['host'])
        key = (run, row['id'])
        require(key not in requests, 'g3_duplicate_request')
        require(run in protocol['runs'] and row['arm'] in ('ane', 'gpu'), 'g3_request_run_arm')
        count = carry[host] if command['input'] == 'carry' else len(inputs[command['input']])
        stats = validate_request(row, command, count)
        require(row['input'] == command['input'], 'g3_request_input')
        carry[host] = count + row['output_tokens']
        rebuilt = dict(item, **stats)
        requests[key] = rebuilt
        if item['stage'] == 'coverage':
            require(run == 'r4' and row['mode'] == 'full' and row['decode_steps'] ==
                    protocol['coverage_decode_steps'], 'g3_coverage_work')
            coverage.append({'arm': row['arm'], 'context': count, 'run': run,
                             'prefill_rate': stats['prefill_tokens_per_second'],
                             'decode_rate': stats['decode_tokens_per_second'],
                             'prefill_seconds': stats['prefill_seconds'],
                             'request_id': row['id']})
    matrix = {(arm, n) for arm in ('ane', 'gpu') for n in lengths}
    require(len(coverage) == len(matrix) and {(r['arm'], r['context']) for r in coverage} == matrix,
            'g3_coverage_inventory')
    power = []
    previous = None
    for index, row in enumerate(read_rows(bundle / 'power.jsonl.gz')):
        require(row['index'] == index, 'g3_power_sample_inventory')
        receipt = row['receipt']
        require(receipt['monotonic_before_ns'] <= receipt['monotonic_after_ns'] and
                (previous is None or 0 < receipt['monotonic_after_ns'] - previous <= 3e9), 'g3_power_receipt')
        previous = receipt['monotonic_after_ns']
        decoded = decode_power(row['plist_fields'].encode() + b'\0', receipt)
        require(decoded == row['recorded_decoded'] and not decoded['issues'], 'g3_power_decode')
        power.append({'receipt': receipt, 'decoded': decoded})
    audit = json.loads((bundle / 'capture-status.json').read_text())
    require(audit['integrity_passed'] and audit['capture_completed'] and audit['cleanup_passed'] and
            not audit['clock_issues'] and len(power) == audit['streams']['power']['samples'], 'g3_capture_integrity')
    blocks = []
    seen = set()
    for block in json.loads((bundle / 'blocks.json').read_text()):
        run, arm, n, mode = block['run'], block['arm'], block['context'], block['mode']
        key = (run, arm, n, mode)
        require(key not in seen and block['capture'] == 'r5' and run in ('r5', 'r6'), 'g3_block_identity')
        seen.add(key)
        count = (protocol['stage_decode_steps'] if mode == 'decode' else
                 protocol['supplement_prefill_count'] if run == 'r6' else protocol['stage_prefill_counts'][str(n)])
        require(block['requested_count'] == count and block['completed_count'] == count, 'g3_block_work_count')
        names = block['request_ids']
        require(len(names) == len(set(names)) == (count if mode == 'prefill' else 1), 'g3_block_requests')
        items = [requests[(run, name)] for name in names]
        rows = [r['result'] for r in items]
        for item in items:
            row = item['result']
            require(row['arm'] == arm and row['mode'] == mode and item['group'] == block['group'], 'g3_block_request_identity')
            require(not row['forced'] and item['command']['forced_ids'] is None, 'g3_decode_policy')
            if mode == 'prefill':
                require(row['input'] == f'reading_{n}' and row['input_tokens'] == n and
                        row['output_tokens'] == 1 and row['initial_processed'] == 0, 'g3_prefill_work')
            else:
                require(row['input'] == 'carry' and row['initial_processed'] == n and
                        row['decode_steps'] == count, 'g3_decode_KV_work')
        tokens = n * count if mode == 'prefill' else count
        start, end = min(r['request_start_ns'] for r in rows), max(r['request_end_ns'] for r in rows)
        windows = {str(lag): energy_window(power, start, end, lag) for lag in protocol['sensitivity_lag_ns']}
        primary = windows[str(protocol['primary_extra_lag_ns'])]
        observed = response(power, start, end, arm)
        seconds = (end - start) / 1e9
        graph_counts = Counter(g['graph'] for r in rows for g in r['graphs'] if g['event'] == 'graph_end')
        kv = []
        if mode == 'decode':
            row = rows[0]
            begin = start
            for offset in range(0, row['decode_steps'], 256):
                chunk = row['token_times'][offset:offset+256]
                finish = chunk[-1]['end_ns']
                kv.append({'start_KV': n+offset, 'end_KV': n+offset+len(chunk),
                           'rate': len(chunk)/((finish-begin)/1e9)})
                begin = finish
        blocks.append(dict(block, tokens=tokens, start_ns=start, end_ns=end, seconds=seconds,
            rate=tokens/seconds, mean_W={k: v['estimate_J']/seconds for k,v in primary['domains'].items()},
            J_per_token={k: v['estimate_J']/tokens for k,v in primary['domains'].items()},
            energy=primary['domains'], sensitivity=windows, response=observed,
            graphs=dict(graph_counts), kv=kv,
            admitted=observed['passed'] and all(v['integrity_passed'] for v in windows.values())))
    expected = {('r5', arm, n, mode) for arm, n in matrix for mode in ('prefill', 'decode')}
    expected |= {('r6', arm, 500, 'prefill') for arm in ('ane', 'gpu')}
    require(seen == expected, 'g3_block_inventory')
    preferred = [b for b in blocks if b['run'] == (protocol['preferred_500_prefill_run']
                 if b['context'] == 500 and b['mode'] == 'prefill' else 'r5')]
    require(all(b['admitted'] for b in preferred), 'g3_preferred_energy_admission')
    pairs = []
    for n in lengths:
        for mode in ('prefill', 'decode'):
            pair = {b['arm']: b for b in preferred if b['context'] == n and b['mode'] == mode}
            require(set(pair) == {'ane', 'gpu'} and pair['ane']['tokens'] == pair['gpu']['tokens'], 'g3_same_work')
            a, g = pair['ane'], pair['gpu']
            ae, ge = a['energy']['components'], g['energy']['components']
            pairs.append({'context': n, 'mode': mode, 'ane': a, 'gpu': g,
                'speed_share': a['rate']/g['rate'], 'energy_ratio': a['J_per_token']['components']/g['J_per_token']['components'],
                'lower_ratio': ae['lower_J']/ge['upper_J'], 'upper_ratio': ae['upper_J']/ge['lower_J']})
    work = model_work(structure)
    return {'protocol': protocol, 'coverage': coverage, 'blocks': blocks, 'pairs': pairs,
            'work': work, 'implied': implied(work, pairs),
            'request_count': len(requests), 'power_samples': len(power)}


def measurements(data):
    lines = ['## G3: complete Qwen3-4B FP16', '',
        'Warmed stage blocks on one M5 Pro. Rate, mean power and energy use the same block. '
        'Energy is CPU + GPU + ANE software component energy, without idle subtraction or recovery. '
        '[Scope](SCOPE.md#g3-complete-model-observations) · [Source bundle](../'+BASE+'/).', '',
        '| Stage | Input / initial KV | GPU token/s | ANE token/s | GPU / ANE speed | GPU W | ANE W | GPU J/token | ANE J/token | ANE / GPU J/token [timing bounds] |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in data['pairs']:
        a,g = p['ane'],p['gpu']
        lines.append(f"| {p['mode']} | {p['context']:,} | {g['rate']:.3f} | {a['rate']:.3f} | {1/p['speed_share']:.2f}× | "
            f"{g['mean_W']['components']:.4f} | {a['mean_W']['components']:.4f} | {g['J_per_token']['components']:.7f} | "
            f"{a['J_per_token']['components']:.7f} | {p['energy_ratio']:.5f} [{p['lower_ratio']:.5f}, {p['upper_ratio']:.5f}] |")
    lines += ['', '### Work and component energy', '',
        '| Run | Stage | Context | Arm | Work tokens | Seconds | CPU J/token | GPU J/token | ANE J/token | Response accepted |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---|']
    for b in data['blocks']:
        e=b['J_per_token']
        lines.append(f"| {b['run']} | {b['mode']} | {b['context']:,} | {b['arm']} | {b['tokens']:,} | {b['seconds']:.6f} | "
            f"{e['cpu']:.7f} | {e['gpu']:.7f} | {e['ane']:.7f} | {b['admitted']} |")
    lines += ['', 'Both r5 short 500-token prefill blocks are superseded by r6 in the primary pair. '
        'The original GPU block has too few interior samples for the response rule.', '',
        '### Single coverage requests (r4)', '',
        'One full request per context and engine, including prefill and 256 decode steps. '
        'These are separate from the warmed stage blocks, not cold-start measurements.', '',
        '| Input tokens | Arm | Prefill token/s | Decode token/s | Prefill + first token (s) |',
        '|---:|---|---:|---:|---:|']
    for r in sorted(data['coverage'], key=lambda r:(r['context'],r['arm'])):
        lines.append(f"| {r['context']:,} | {r['arm']} | {r['prefill_rate']:.3f} | {r['decode_rate']:.3f} | {r['prefill_seconds']:.6f} |")
    work = data['work']
    lines += ['', '### Implied compute and weight-read bandwidth', '',
        f"Derived from the rates above and [model-structure.json](../{BASE}/model-structure.json). "
        f"Prefill counts two FLOPs per projection weight per token ({work['projection_parameters']:,} weights) "
        f"and lists attention separately. Decode assumes every FP16 weight ({work['weight_bytes']:,} bytes) "
        f"and the existing KV cache are read once per step, summing the growing KV over the whole block. "
        f"Current-token writes are excluded. These are useful-work rates for these implementations: "
        f"they exclude padding inside a fixed graph, and no device counter or DRAM traffic was measured.", '',
        '| Stage | Input / initial KV | GPU | ANE | Unit | Attention adds (GPU / ANE) |',
        '|---|---:|---:|---:|---|---:|']
    for row in [r for r in data['implied'] if r['arm'] == 'gpu']:
        other = next(r for r in data['implied'] if r['arm'] == 'ane' and
                     r['mode'] == row['mode'] and r['context'] == row['context'])
        if row['mode'] == 'prefill':
            lines.append(f"| prefill | {row['context']:,} | {row['tera_flops_per_second']:.2f} | "
                f"{other['tera_flops_per_second']:.2f} | TFLOP/s over projection weights | "
                f"{row['attention_tera_flops_per_second']:.2f} / {other['attention_tera_flops_per_second']:.2f} |")
        else:
            lines.append(f"| decode | {row['context']:,} | {row['giga_bytes_per_second']:.1f} | "
                f"{other['giga_bytes_per_second']:.1f} | GB/s over {row['step_bytes']:,.0f} mean bytes per step | \u2014 |")
    return '\n'.join(lines)+'\n'

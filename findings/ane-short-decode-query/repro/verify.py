"""Check the recorded decode-query-width outcomes; standard library only, no device."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDED = HERE / 'recorded'


def require(value, name):
    if not value:
        raise ValueError(name)


def outcomes(recorded=RECORDED, scripts=HERE):
    provenance = json.loads((recorded / 'provenance.json').read_text())
    for name, digest in provenance['products'].items():
        require(hashlib.sha256((recorded / name).read_bytes()).hexdigest() == digest, 'short_query_product_identity:' + name)
    data = json.loads((recorded / 'observations.json').read_text())
    for name, digest in data['scripts_sha256_at_import'].items():
        require(hashlib.sha256((scripts / name).read_bytes()).hexdigest() == digest, 'short_query_script_identity:' + name)
    rows = []
    for case in data['cases']:
        functions = set(case['asset']['functions'])
        capacity = case['asset']['capacity']
        for query, result in case['results'].items():
            q = int(query)
            require(f'extend_{capacity}_{q}' in functions, f"short_query_function_present:{case['case']}:{q}")
            inputs = dict(map(tuple, case['asset']['decode_signatures'][f'extend_{capacity}_{q}']['inputs']))
            require(inputs['transformer_input'] == f'NDArray (Float16, 1 × {q} × 1 × 2560)' and
                    inputs['position_ids'] == f'NDArray (UInt16, 1 × {q})' and
                    inputs['causal_mask'] == f'NDArray (Float16, 1 × {capacity} × 1 × {q})', f"short_query_signature:{case['case']}:{q}")
            log, assertion = result['log'], result['mpsgraph_assertion']
            if q in (1, 2):
                # The first request of this width fails in the ANE driver; the host aborts in MPSGraph.
                require(not result['passed'] and result['host_returncode'] != 0 and
                        'ANE_ProgramSendRequest:1029 status=e00002c2' in log['failure_status_lines'] and
                        assertion is not None and assertion['error_code'] == -19 and assertion['failure_count'] == 0 and
                        assertion['inference_count'] > 0 and log['ane_direct_request_success_rows'] >= assertion['inference_count'],
                        f"short_query_failure:{case['case']}:{q}")
                outcome = 'fails'
            else:
                require(result['passed'] and result['host_returncode'] == 0 and not log['failure_status_lines'] and
                        log['ane_direct_request_success_rows'] > 0 and assertion is None, f"short_query_success:{case['case']}:{q}")
                if 'same_history_vs_first_query' in result:
                    require(result['same_ids_as_first_query'] and all(
                        m['passed'] and m['max_abs_error_diagnostic'] == 0 for m in result['same_history_vs_first_query']),
                        f"short_query_same_logits:{case['case']}:{q}")
                outcome = 'runs'
            rows.append({'case': case['case'], 'layers': case['asset']['layers'], 'query': q, 'outcome': outcome,
                         'ane_requests_before_failure': assertion['inference_count'] if assertion else None,
                         'bundle_queries': sorted(int(f.split('_')[-1]) for f in functions if f.startswith(f'extend_{capacity}_'))})
    require({r['query'] for r in rows if r['outcome'] == 'fails'} == {1, 2} and
            {r['query'] for r in rows if r['outcome'] == 'runs'} == {4, 8}, 'short_query_outcome_inventory')
    require(any(r['layers'] == 36 and r['outcome'] == 'fails' for r in rows) and
            any(r['outcome'] == 'fails' and 8 not in r['bundle_queries'] for r in rows), 'short_query_isolation_cases')
    require('specialized it to be a constant (1)' in data['trace_query_1_export']['torch_export_error'], 'short_query_trace_export')
    return data, rows


if __name__ == '__main__':
    data, rows = outcomes()
    for r in rows:
        extra = f", {r['ane_requests_before_failure']} ANE requests before the failure" if r['ane_requests_before_failure'] else ''
        print(f"{r['case']}: {r['layers']} layer(s), bundle queries {r['bundle_queries']}, query {r['query']} {r['outcome']}{extra}")
    print('Query 1 trace export: torch.export specializes seq_len to a constant. No device execution.')

"""Check the recorded FP16/W4 load-time placement pair; standard library only, no device."""
import json
from pathlib import Path

RECORDED = Path(__file__).resolve().parent / 'recorded'


def require(value, name):
    if not value:
        raise ValueError(name)


def placement(recorded=RECORDED):
    pair = json.loads((recorded / 'observations.json').read_text())['pair']
    fp16, w4 = pair['fp16'], pair['w4']
    # Same exporter, shapes and functions; only the compression differs.
    for key in ('layers', 'capacities', 'max_context_length', 'queries', 'trace_query', 'environment', 'source_revision', 'functions'):
        require(fp16['export'][key] == w4['export'][key], 'palettized_matched_export:' + key)
    require(fp16['export']['compression'] == 'none' and w4['export']['compression'] == '4bit_weight_palettized_group32',
            'palettized_compression')
    rows = {}
    for name, entry in pair.items():
        log = entry['log_from_before_host_start']
        messages = [line['message'] for line in log['key_lines']]
        results = entry['results']
        # One host served both widths; its exit is recorded with the last width.
        last = str(entry['queries_in_order'][-1])
        require(all(r['passed'] and r['host_returncode'] in (0, None) for r in results.values()) and results[last]['host_returncode'] == 0 and
                results['4']['same_ids_as_first_query'] and
                all(m['passed'] and m['max_abs_error_diagnostic'] == 0 for m in results['4']['same_history_vs_first_query']),
                'palettized_probe:' + name)
        require(any('preferredDevice=NeuralEngine' in m for m in messages) and
                any(m == 'End of mlir<anec> Validation' for m in messages), 'palettized_ane_preferred_and_validated:' + name)
        compiled = [m for m in messages if m.startswith('[proxy compileModel:...] returned success=')]
        require(len(compiled) == 1, 'palettized_one_compile:' + name)
        rows[name] = {'compile_success': compiled[0].startswith('[proxy compileModel:...] returned success=1'),
                      'ane_compilation_failed': any('ANE compilation failed' in m for m in messages),
                      'ane_requests': log['ane_direct_request_success_rows'], 'metal_shader_compiles': log['metal_shader_compile_rows']}
    require(rows['fp16']['compile_success'] and not rows['fp16']['ane_compilation_failed'] and rows['fp16']['ane_requests'] > 0 and
            rows['fp16']['metal_shader_compiles'] == 0, 'palettized_fp16_on_ane')
    require(not rows['w4']['compile_success'] and rows['w4']['ane_compilation_failed'] and rows['w4']['ane_requests'] == 0 and
            rows['w4']['metal_shader_compiles'] > 0, 'palettized_w4_not_on_ane')
    return rows


if __name__ == '__main__':
    for name, row in placement().items():
        print(f"{name}: ANE compile {'succeeded' if row['compile_success'] else 'failed'}, {row['ane_requests']} ANE requests, "
              f"{row['metal_shader_compiles']} Metal shader compilations; both decode widths returned the same logits")
    print('No device execution.')

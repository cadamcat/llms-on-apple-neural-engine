"""Export Qwen3-4B iOS bundles for the G6 query-width work (truncated for diagnosis, or full).

Same frozen exporter and shape override as g4-cplus256 prepare.py, with the layer count,
capacity, query set and trace query length as parameters. Diagnostic assets only.
"""
import argparse, importlib.metadata, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'workplans/g4-cplus256-20260912'
sys.path.insert(0, str(BASE))
from prepare import SOURCE, VENDOR, save


def shape_configs(original, cfg, capacities, queries):
    # As prepare.shape_configs: the exporter's 256-cache entries per query, resized to each capacity.
    result = original(cfg, max(capacities))
    for variants in result.values():
        if not any('_' in k for k in variants):
            continue
        templates = {q: variants[f'"256_{q}"'] for q in queries}
        variants.clear()
        for capacity in capacities:
            for q in queries:
                entry = {}
                for key, shape in templates[q].items():
                    shape = list(shape)
                    if key == 'causal_mask': shape[1] = capacity
                    elif key in ('key_cache', 'value_cache'): shape[4] = capacity
                    entry[key] = tuple(shape)
                variants[f'"{capacity}_{q}"'] = entry
    return result


def export(out, layers, capacities, queries, trace_query, prompt_queries, max_context, compression='none'):
    from coreai.authoring import AIProgram
    from coreai_models.export.pipeline import ExportConfig, export_model
    from coreai_models.export.metadata import _METADATA
    from coreai_models.models.base import BaseForCausalLMForiOS
    import inspect, torch
    if Path(inspect.getfile(BaseForCausalLMForiOS)).resolve() != VENDOR / 'python/src/coreai_models/models/base.py':
        raise RuntimeError('exporter_source_path')
    out = Path(out)
    if out.exists(): raise RuntimeError('fresh_asset_directory_required')
    import hashlib
    identity = json.loads((SOURCE / 'source.json').read_text())
    for row in identity['files']:
        path = SOURCE / row['name']
        with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise RuntimeError('source_identity:' + row['name'])
    torch.set_num_threads(4)
    BaseForCausalLMForiOS.IOS_STATIC_QUERY_LENS = tuple(queries)
    BaseForCausalLMForiOS.IOS_QUERY_LEN = trace_query
    original = BaseForCausalLMForiOS.export_static_shape_configs
    BaseForCausalLMForiOS.export_static_shape_configs = classmethod(
        lambda cls, cfg, _capacity: shape_configs(original, cfg, capacities, queries))
    original_set = AIProgram.set_static_shape_config
    def set_shapes(self, entrypoint, configs):
        if entrypoint == 'prompt_opt':
            configs = {k: v for k, v in configs.items() if int(k.strip('"').split('_')[1]) in prompt_queries}
        return original_set(self, entrypoint, configs)
    AIProgram.set_static_shape_config = set_shapes
    _METADATA[str(SOURCE)] = _METADATA['Qwen/Qwen3-4B']
    bundle = export_model(ExportConfig(hf_model_id=str(SOURCE), variant='iOS',
        max_context_length=max_context or max(capacities), compute_precision='float16', compression=compression,
        disable_embedding_quantization=True, output_dir=str(out.parent),
        output_name=out.name, overwrite=False, num_layers=layers))
    if Path(bundle).resolve() != out.resolve(): raise RuntimeError('export_destination')
    from coreai.authoring import AIModelAsset
    asset = AIModelAsset.load(out / (out.name + '.aimodel'))
    summary = asset.summary(include_statistics=False)
    functions = {n: {'inputs': summary.function_inputs(n), 'outputs': summary.function_outputs(n)}
                 for n in summary.function_names}
    save(out / 'G6-Q1-EXPORT.json', {'source_revision': identity['revision'], 'source_verified': True, 'layers': layers, 'capacities': list(capacities), 'max_context_length': max_context or max(capacities), 'compression': compression, 'queries': list(queries),
        'trace_query': trace_query, 'prompt_queries': list(prompt_queries), 'functions': functions,
        'environment': {p: importlib.metadata.version(p) for p in ('coreai-core', 'coreai-torch', 'torch', 'transformers')}})
    print(json.dumps({'bundle': str(out), 'functions': sorted(functions)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', type=int, default=1, help='0 exports every layer')
    p.add_argument('--capacities', type=int, nargs='+', default=[768])
    p.add_argument('--max-context', type=int, help='KV alignment capacity; default the largest capacity')
    p.add_argument('--queries', type=int, nargs='+', default=[1, 8, 64])
    p.add_argument('--trace-query', type=int, default=8)
    p.add_argument('--prompt-queries', type=int, nargs='+', default=[64])
    p.add_argument('--compression', default='none')
    a = p.parse_args()
    export(a.out, a.layers or None, a.capacities, a.queries, a.trace_query, a.prompt_queries, a.max_context, a.compression)

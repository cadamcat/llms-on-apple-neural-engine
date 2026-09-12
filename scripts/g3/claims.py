"""Named document quantities derived from the imported G3 records."""
import json
from pathlib import Path
from .evidence import BASE, derive, synthetic_reference


def lengths_short(data):
    """Largest context before the ANE path changes graph, from the protocol lengths."""
    return data['protocol']['lengths'][2]


def quantities(data, root):
    from doc_claims import Quantity
    out = {}
    out['g3.synthetic-fp16'] = Quantity(f'{synthetic_reference(root):.1f}',
        'T source-equivalent ops/s', 'T 源图等效 ops/s', 'results/fresh/throughput.json')
    def add(key, value, en='', zh='', source='blocks.json'):
        out['g3.'+key] = Quantity(str(value), en, zh, BASE+'/'+source)
    for p in data['pairs']:
        stem=f"{p['mode']}.{p['context']}."
        for arm in ('ane','gpu'):
            b=p[arm]
            add(stem+arm+'-rate',f"{b['rate']:,.1f}",'token/s','token/s','requests.jsonl.gz')
            add(stem+arm+'-power',f"{b['mean_W']['components']:.2f}",'W','W','power.jsonl.gz')
            add(stem+arm+'-energy',f"{b['J_per_token']['components']:.5f}",'J/token','J/token','power.jsonl.gz')
        add(stem+'share',f"{100*p['speed_share']:.1f}%",source='requests.jsonl.gz')
        add(stem+'ratio',f"{p['energy_ratio']:.3f}",source='power.jsonl.gz')
        add(stem+'energy-x',f"{p['energy_ratio']:.2f}×",source='power.jsonl.gz')
        add(stem+'gpu-faster',f"{1/p['speed_share']:.2f}×",source='requests.jsonl.gz')
    for r in data['implied']:
        stem=f"{r['mode']}.{r['context']}.{r['arm']}-"
        if r['mode']=='prefill':
            add(stem+'tflops',f"{r['tera_flops_per_second']:.1f}",'TFLOP/s','TFLOP/s','model-structure.json')
        else:
            add(stem+'bandwidth',f"{r['giga_bytes_per_second']:.0f}",'GB/s','GB/s','model-structure.json')
    add('projection-parameters',f"{data['work']['projection_parameters']/1e9:.2f}",'billion','billion','model-structure.json')
    work=data['work']
    largest=max(data['protocol']['ANE_shapes']['contexts'])
    add('projection-flops',f"{work['flops_per_token']/1e9:.2f}",'GFLOP/token','GFLOP/token','model-structure.json')
    add('attention-flops-largest-graph',
        f"{work['attention_flops_per_token_per_context']*largest/1e9:.1f}",'GFLOP/token','GFLOP/token','model-structure.json')
    rates={(p['mode'],p['context']):p for p in data['pairs']}
    add('prefill-drop',f"{rates[('prefill',2048)]['ane']['rate']/rates[('prefill',4096)]['ane']['rate']:.1f}×",
        source='requests.jsonl.gz')
    add('projection-parameters-cn',f"{data['work']['projection_parameters']/1e8:.1f}",'hundred million','亿','model-structure.json')
    add('weight-bytes',f"{data['work']['weight_bytes']/1e9:.2f}",'GB','GB','model-structure.json')
    prefill=[p for p in data['pairs'] if p['mode']=='prefill']
    decode=[p for p in data['pairs'] if p['mode']=='decode']
    def span(values, places):
        return f'{min(values):.{places}f}–{max(values):.{places}f}'
    add('short-prefill-energy-x',span([p['energy_ratio'] for p in prefill[:3]],2)+'×',source='power.jsonl.gz')
    add('short-prefill-energy-share',span([100*p['energy_ratio'] for p in prefill[:3]],0)+'%',source='power.jsonl.gz')
    add('long-prefill-energy-ratio',span([p['energy_ratio'] for p in prefill[3:]],2)+'×',source='power.jsonl.gz')
    add('long-decode-energy-ratio',span([p['energy_ratio'] for p in decode[2:]],2)+'×',source='power.jsonl.gz')
    short_tflops=[r['tera_flops_per_second'] for r in data['implied']
                  if r['mode']=='prefill' and r['arm']=='ane' and r['context']<=lengths_short(data)]
    add('short-prefill-ane-tflops',span(short_tflops,1),'TFLOP/s','TFLOP/s','model-structure.json')
    add('short-prefill-speed-share',span([100*p['speed_share'] for p in prefill[:3]],1)+'%',source='requests.jsonl.gz')
    add('long-prefill-speed-share',span([100*p['speed_share'] for p in prefill[3:]],1)+'%',source='requests.jsonl.gz')
    add('coverage-steps',data['protocol']['coverage_decode_steps'],source='protocol.json')
    add('decode-steps',f"{data['protocol']['stage_decode_steps']:,}",source='protocol.json')
    add('supplement-count',data['protocol']['supplement_prefill_count'],source='protocol.json')
    add('contexts',' / '.join(f'{n:,}' for n in data['protocol']['lengths']),source='protocol.json')
    add('context-count',len(data['protocol']['lengths']),source='protocol.json')
    for n in data['protocol']['lengths']:
        add(f'n.{n}',str(n) if n<1000 else f'{n//1024}K',source='protocol.json')
    label=lambda n: str(n) if n<1000 else f'{n//1024}K'
    lengths=data['protocol']['lengths']
    add('short-contexts',label(lengths[0])+'–'+label(lengths[2]),source='protocol.json')
    add('long-contexts',label(lengths[3])+'–'+label(lengths[-1]),source='protocol.json')
    add('decode-long-contexts',label(lengths[2])+'–'+label(lengths[-1]),source='protocol.json')
    add('primary-lag',data['protocol']['primary_extra_lag_ns']//10**9,'s','秒',source='protocol.json')
    add('graph-contexts',' / '.join(str(n) if n<1000 else f'{n//1024}K'
                                for n in data['protocol']['ANE_shapes']['contexts']),source='protocol.json')
    add('largest-graph',label(max(data['protocol']['ANE_shapes']['contexts'])),source='protocol.json')
    add('capacity',f"{data['protocol']['capacity']:,}",source='protocol.json')
    add('old-short-gpu-seconds',f"{next(b['seconds'] for b in data['blocks'] if b['run']=='r5' and b['context']==500 and b['arm']=='gpu' and b['mode']=='prefill'):.2f}",'s','秒',source='requests.jsonl.gz')
    for r in data['coverage']:
        add(f"coverage.{r['context']}.{r['arm']}-prefill-rate",f"{r['prefill_rate']:,.1f}",'token/s','token/s','requests.jsonl.gz')
    return out


def check(root, data=None):
    from doc_claims import check_document
    root=Path(root)
    catalog=quantities(data if data is not None else derive(root/BASE), root)
    required=json.loads((root/'scripts/g3/claims.json').read_text())
    errors=[]
    for name, locations in required.items():
        language='zh' if name.endswith('zh-CN.md') or '/zh/' in name else 'en'
        # Other claim families in a README are checked by their own registry.
        import re
        body=(root/name).read_text()
        body=re.sub(r'<!-- claim:(?!g3\.)[^>]+-->.*?<!-- /claim -->','',body,flags=re.DOTALL)
        errors.extend(check_document(body,catalog,language,required=locations,name=name))
    return errors

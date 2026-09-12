"""Static, model-free experiment catalog and lazy backend dispatch."""
from pathlib import Path
from .common import dump, read

CHAINS = [f'{r}-{f}-{d}' for d in (2,128) for r,fs in [('coreml',('fp16','w8a8')),('coreai',('fp16','w8a8','a8w4'))] for f in fs]
GROUPS = [f'{r}-group-{k}' for r in ('coreml','coreai') for k in ('native64','split32')]
MULTIPLY = [f'coreai-qdq-{k}' for k in ('equal','unequal','reverse_order','explicit_q')]
SPLIT = 'coreai-a8w4-split32-2'
CASES = CHAINS + [SPLIT] + GROUPS + MULTIPLY

def describe(case_id):
    if case_id not in CASES:raise ValueError(f'unknown case {case_id}')
    group='-group-' in case_id;mul='-qdq-' in case_id;split=case_id==SPLIT
    depth=int(case_id.rsplit('-',1)[1]) if not (group or mul) else None
    info={'case_id':case_id,'runtime':case_id.split('-')[0], 'shape':[2,64,1,16] if mul else [1,64,1,64] if group else [1,512,64,64],
          'output_shape':[1,64,1,16] if mul else [1,64,1,64] if group else [1,512,64,64],
          'input_file':'mul_input.raw' if mul else 'group_input.raw' if group else 'input.raw',
          'reference_file':'reference-mul.npy' if mul else 'reference-group.npy' if group else 'reference-a8w4-split32-2.npy' if split else f'reference-{"fp16" if "-fp16-" in case_id else "w8a8"}-{depth}.npy',
          'reference_repeat_columns':1 if group or mul else 256,'negative_parity':'even' if mul else 'odd',
          'l2_limit':0 if group or mul or split else .01 if depth==2 else .05,
          'expected_conv_count':0 if mul else (2 if case_id.endswith('split32') else 1) if group else 32 if split else depth,
          'benchmark_eligible':not(group or mul),'source_ops':None if group or mul else 2*depth*512**2*4096,
          'entrypoint':'scope_'+case_id.replace('-','_'),'input_name':None,'output_name':None}
    return info

def export_case(case_id:str,data:Path,out:Path)->dict:
    data=Path(data);out=Path(out);info=describe(case_id)
    if out.exists():raise FileExistsError(out)
    if not (data/info['reference_file']).exists():raise ValueError('data profile lacks requested reference')
    from .common import sha
    for name,digest in read(data/'manifest.json')['files'].items():
        if sha(data/name)!=digest:raise ValueError('fixture identity mismatch: '+name)
    if info['runtime']=='coreml':
        from ._coreml import build
    else:
        from ._coreai import build
    out.mkdir(parents=True);details=build(case_id,data,out,info);info.update(details)
    if not read(out/'asset-audit.json')['passed']:raise ValueError('persisted asset audit failed')
    dump(out/'export.json',info);return info

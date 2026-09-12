"""Core AI synthetic graphs; persisted bytecode audit is pinned to the 0.4.1 API."""
import ast,hashlib,re
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import coreai_torch
from coreai.authoring import AIProgram
from coreai_torch._compression.custom_layers import ScaledPalettizeModule,WeightDequantizeModule
from .common import dump,file_hashes
S=np.float16(.044189453125)

def qdq(x,s,z,axis=0):
    q=torch.ops.coreai.quantize.default(x,s,torch.int8,z,None,axis)
    return torch.ops.coreai.dequantize.default(q,s,z,None,axis,torch.int8)

class Conv(nn.Module):
    def __init__(self,codes,scales,fmt):
        super().__init__();self.fmt=fmt
        c=torch.from_numpy(np.array(codes));s=torch.from_numpy(np.array(scales,dtype=np.float16))
        if fmt=='fp16':self.register_buffer('w',c.half()*s)
        elif fmt=='w8a8':self.w=WeightDequantizeModule(c,s,zero_point=torch.zeros_like(s,dtype=torch.int8),input_dtype=torch.int8,output_dtype=torch.float16)
        else:self.w=ScaledPalettizeModule((c.to(torch.int16)+8).to(torch.uint8),torch.arange(-8,8,dtype=torch.int8).reshape(1,1,1,1,16,1),s,vector_axis=0,zero_point=torch.zeros_like(s,dtype=torch.int8),input_dtype=torch.int8,output_dtype=torch.float16)
    def forward(self,x):return F.conv2d(x,self.w if self.fmt=='fp16' else self.w())

class Chain(nn.Module):
    def __init__(self,weights,fmt,split):
        super().__init__();self.fmt=fmt;self.split=split
        self.layers=nn.ModuleList([nn.ModuleList([Conv(w[:,j:j+32].reshape(512,32,1,1),np.full((1,1,1,1),S),fmt) for j in range(0,512,32)]) if split else Conv(w.reshape(512,512,1,1),np.full((1,1,1,1),S),fmt) for w in weights])
        self.register_buffer('s',torch.tensor(.125,dtype=torch.float16));self.register_buffer('z',torch.tensor(0,dtype=torch.int8))
    def forward(self,x):
        for i,layer in enumerate(self.layers):
            if self.split:
                ys=[b(x[:,j*32:(j+1)*32,:,:]) for j,b in enumerate(layer)]
                while len(ys)>1:ys=[ys[j]+ys[j+1] for j in range(0,len(ys),2)]
                x=ys[0]
            else:x=layer(x)
            if self.fmt!='fp16' and i<len(self.layers)-1:x=qdq(x,self.s,self.z)
        return x

class Group(nn.Module):
    def __init__(self,codes,scales,si,split):
        super().__init__();self.split=split
        self.parts=nn.ModuleList([Conv(codes[:,j*32:(j+1)*32],scales[:,j:j+1],'a8w4') for j in range(2)]) if split else nn.ModuleList([Conv(codes,scales,'a8w4')])
        self.register_buffer('s',torch.from_numpy(si.copy()));self.register_buffer('z',torch.zeros(64,dtype=torch.int8))
    def forward(self,x):
        x=qdq(x,self.s,self.z,1)
        return self.parts[0](x[:,:32,:,:])+self.parts[1](x[:,32:,:,:]) if self.split else self.parts[0](x)

class Multiply(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.register_buffer('sg',torch.tensor(2. if mode=='equal' else .5,dtype=torch.float16));self.register_buffer('su',torch.tensor(2.,dtype=torch.float16));self.register_buffer('z',torch.tensor(0,dtype=torch.int8))
    def apply_qdq(self,x,s):
        q=torch.round(x/s).clamp(-128,127).to(torch.int8) if self.mode=='explicit_q' else torch.ops.coreai.quantize.default(x,s,torch.int8,self.z)
        return torch.ops.coreai.dequantize.default(q,s,self.z,None,0,torch.int8)
    def forward(self,x):
        if self.mode=='reverse_order':u=self.apply_qdq(x[1:2],self.su);g=self.apply_qdq(x[0:1],self.sg)
        else:g=self.apply_qdq(x[0:1],self.sg);u=self.apply_qdq(x[1:2],self.su)
        return g*u

def audit(asset,out,case_id,data,info):
    text=str(AIProgram._load_bytecode(asset/'main.mlirb'));body=text.split('\n{-#',1)[0]
    (out/'persisted-graph.mlir').write_text(body)
    constants={};ops={};resources={n:bytes.fromhex(h) for n,h in re.findall(r'([A-Za-z_][A-Za-z0-9_]*): "0x([0-9A-Fa-f]*)"',text)}
    for line in body.splitlines():
        m=re.match(r'\s*(%\d+) = coreai\.constant (.+) : tensor<([^>]+)>',line)
        if m:constants[m[1]]=(m[2],m[3]);continue
        m=re.match(r'\s*(%\d+) = (coreai\.[a-z_0-9.]+) (.+?) : ',line)
        if m:ops[m[1]]=(m[2],re.findall(r'%\w+',m[3]))
    def value(ssa):
        literal,typ=constants[ssa];tokens=typ.split('x');dtype=tokens[-1];shape=tuple(map(int,tokens[:-1]));n=int(np.prod(shape)) if shape else 1
        dtypes={'f16':'<f2','si8':'i1','ui8':'u1','si16':'<i2','si32':'<i4','ui32':'<u4'}
        if literal.startswith('dense_resource'):
            raw=resources[re.fullmatch(r'dense_resource<([^>]+)>',literal)[1]][4:]
            if dtype=='ui4':
                packed=np.frombuffer(raw,np.uint8);a=np.empty(n,np.uint8);a[::2]=packed&15;a[1::2]=packed>>4
            else:a=np.frombuffer(raw,dtypes[dtype]);assert a.size==n
        else:
            v=ast.literal_eval(literal[len('dense<'):-1])
            if isinstance(v,str) and v.startswith('0x'):
                a=np.frombuffer(bytes.fromhex(v[2:]),dtypes[dtype]);assert a.size==n
            else:a=np.asarray(v,dtype=dtypes.get(dtype,'u1'))
            if a.size==1:a=np.full(n,a.item(),dtype=a.dtype)
        return a.reshape(shape)
    head=body.splitlines()[1];expected_in='x'.join(map(str,info['shape']))+'xf16';expected_out='x'.join(map(str,info['output_shape']))+'xf16'
    assert f'@{info["entrypoint"]}(' in head and f'%arg0: tensor<{expected_in}>' in head and f'-> (tensor<{expected_out}>' in head
    names=re.findall(r'coreai.name = "([^"]+)"',head);assert len(names)==2
    final=re.search(r'coreai.output (%\d+) :',body)[1]
    counts=Counter(op for op,args in ops.values());convs=[(key,args) for key,(op,args) in ops.items() if op=='coreai.conv2d']
    assert len(convs)==info['expected_conv_count']
    group='-group-' in case_id;mul='-qdq-' in case_id;split='split32' in case_id;fmt=case_id.split('-')[1]
    expectedq=2 if mul and not case_id.endswith('explicit_q') else 0 if mul or fmt=='fp16' else 1 if group or split else len(convs)-1
    assert counts['coreai.quantize']==expectedq and counts['coreai.dequantize']==(2 if mul else expectedq)
    qops=[(key,args) for key,(op,args) in ops.items() if op=='coreai.quantize'];dqops=[(key,args) for key,(op,args) in ops.items() if op=='coreai.dequantize']
    for key,args in dqops:
        assert not np.any(value(args[2])) and not np.any(value(args[3]))
        if not (mul and case_id.endswith('explicit_q')):
            qop,qa=ops[args[0]];assert qop=='coreai.quantize'
            assert all(np.array_equal(value(a),value(b)) for a,b in zip(args[1:],qa[1:]))
        if group:assert value(args[1]).tobytes()==np.load(data/'group_input_scales.npy').tobytes() and int(value(args[4]))==1
        elif not mul:assert float(value(args[1]))==.125 and int(value(args[4]))==0
    if mul:
        assert counts['coreai.decomposable.broadcasting_mul']==1
        op,args=ops[final];assert op=='coreai.decomposable.broadcasting_mul' and set(args)=={k for k,a in dqops}
        scales=sorted(float(value(a[1])) for k,a in dqops)
        assert scales==([2.,2.] if case_id.endswith('-equal') else [.5,2.])
    weights=None if mul else np.load(data/('group_codes.npy' if group else 'weights.npy'),mmap_mode='r');scales=np.load(data/'group_scales.npy') if group else np.full((1,1,1,1),S,np.float16)
    decodes=[];layer_ends=[]
    def check_slice(key,base,start,width):
        op,a=ops[key];assert op=='coreai.slice' and a[0]==base
        begin=value(a[1]).reshape(-1);end=value(a[2]).reshape(-1);stride=value(a[3]).reshape(-1)
        assert begin.tolist()==[0,start,0,0] and min(int(end[1]),info['shape'][1])==start+width and np.all(stride==1)
    perlayer=2 if group and split else 16 if split else 1
    current=[]
    for k,(key,a) in enumerate(convs):
        assert np.all(value(a[2])==1) and np.all(value(a[3])==1) and int(value(a[4]))==1
        j=k%perlayer;i=k//perlayer
        wanted=np.array(weights[:,j*32:(j+1)*32] if split else weights) if group else np.array(weights[i,:,j*32:(j+1)*32] if split else weights[i]).reshape(512,32 if split else 512,1,1)
        s=scales[:,j:j+1] if group and split else scales
        if fmt=='fp16':actual=value(a[1]);assert actual.dtype==np.float16
        else:
            op,b=ops[a[1]];assert op=='coreai.blockwise_shift_scale'
            scale=value(b[1]);assert scale.tobytes()==s.tobytes() and not np.any(value(b[2])) and not np.any(value(b[3]))
            if fmt=='w8a8':q=value(b[0]);assert q.dtype==np.int8
            else:
                lop,l=ops[b[0]];assert lop=='coreai.lut_to_dense' and constants[l[0]][1].endswith('xui4')
                palette=value(l[1]);assert palette.shape==(1,1,1,1,16,1) and np.array_equal(palette.reshape(-1),np.arange(-8,8,dtype=np.int8))
                q=palette.reshape(-1)[value(l[0])]
            assert np.array_equal(q,wanted)
            expanded=np.repeat(scale,32,axis=1) if group else scale
            actual=(q.astype(np.float64)*expanded.astype(np.float64)).astype(np.float16)
        target=(wanted.astype(np.float64)*(np.repeat(s,32,axis=1) if group else s).astype(np.float64)).astype(np.float16)
        assert actual.tobytes()==target.tobytes()
        base=dqops[0][0] if group else '%arg0' if i==0 else dqops[i-1][0] if fmt!='fp16' else layer_ends[i-1]
        if split:check_slice(a[0],base,j*32,32)
        else:assert a[0]==base
        current.append(key)
        if j==perlayer-1:
            while len(current)>1:
                nextlevel=[]
                for z in range(0,len(current),2):
                    matches=[key for key,(op,a) in ops.items() if op=='coreai.decomposable.broadcasting_add' and a==current[z:z+2]]
                    assert len(matches)==1;nextlevel.append(matches[0])
                current=nextlevel
            layer_ends.append(current[0]);current=[]
        decodes.append({'conv':k,'shape':list(actual.shape),'sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
    if not mul:
        assert final==layer_ends[-1]
        if group:assert ops[ops[dqops[0][0]][1][0]][1][0]=='%arg0'
        elif fmt!='fp16':
            for i,(key,a) in enumerate(qops):assert a[0]==layer_ends[i]
        assert counts['coreai.decomposable.broadcasting_add']==(30 if split and not group else 1 if split else 0)
    dump(out/'asset-audit.json',{'passed':True,'method':'persisted Core AI MLIR bytecode, version-specific internal reader, SSA walk and independent payload decode','asset_files':file_hashes(asset),'actual_op_counts':dict(counts),'weight_decodes':decodes,'checks':['input/output signature','actual operation counts','weight operands/codes/scales/zero','QDQ SSA and parameters','chain and split reduction SSA','multiply DQ operands/scales']})
    return names

def build(case_id,data,out,info):
    torch.set_num_threads(1)
    if '-group-' in case_id:model=Group(np.load(data/'group_codes.npy'),np.load(data/'group_scales.npy'),np.load(data/'group_input_scales.npy'),case_id.endswith('split32'))
    elif '-qdq-' in case_id:model=Multiply(case_id.split('-')[-1])
    else:model=Chain(np.load(data/'weights.npy',mmap_mode='r')[:int(case_id.split('-')[-1])],case_id.split('-')[1],'split32' in case_id)
    model.eval();ep=torch.export.export(model,(torch.zeros(info['shape'],dtype=torch.float16),)).run_decompositions(coreai_torch.get_decomp_table())
    converter=coreai_torch.TorchConverter();converter.add_exported_program(ep,entrypoint_name=info['entrypoint']);program=converter.to_coreai();program.optimize()
    asset=out/'model.aimodel';program.save_asset(asset);names=audit(asset,out,case_id,data,info)
    return {'model_path':str(asset),'input_name':names[0],'output_name':names[1],'representation':'Core AI '+('LUT INT4 plus blockwise scales' if '-a8w4-' in case_id or '-group-' in case_id else case_id.split('-')[1])}

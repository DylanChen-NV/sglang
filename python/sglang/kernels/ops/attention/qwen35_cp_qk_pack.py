"""CP output addressing for SGLang's Q/K norm, RoPE and gate kernel.

The arithmetic mirrors fused_qk_rmsnorm_rope_gate. Only destination addresses
change; K programs also copy their corresponding V head as raw BF16 bits.
"""
import torch
import triton
import triton.language as tl
import functools
import hashlib
from pathlib import Path
from .fused_qk_rmsnorm_rope_gate import _ENABLE_PDL


@functools.lru_cache(maxsize=1)
def _check_reference():
    source=Path(__file__).with_name('fused_qk_rmsnorm_rope_gate.py')
    if hashlib.sha256(source.read_bytes()).hexdigest()!='d1972dffb4da33fab4410567f32359527bc5c32794c8d8015ffc39b9b9d80cc1':
        raise RuntimeError('CP Q/K arithmetic reference changed; revalidate packet fusion')


@triton.jit
def _norm_pack(QG,K,V,QA,KH,VH,Send,Gate,QW,KW,Rope,Positions,
               qgs,ks,vs,ropes,start,prefix,R:tl.constexpr,ENABLE_PDL:tl.constexpr):
    token,head=tl.program_id(0),tl.program_id(1)
    is_k=head>=16
    local_head=tl.where(is_k,head-16,head)
    if is_k:
        base=K+token*ks+local_head*256
        weight=KW
        own=local_head==R
        output=tl.where(own,KH+(prefix+start+token)*256,Send+token*2560+2048)
    else:
        base=QG+token*qgs+local_head*512
        weight=QW
        own=(local_head//8)==R
        output=tl.where(own,QA+(start+token)*2048+(local_head%8)*256,
                       Send+token*2560+(local_head%8)*256)
    offsets=tl.arange(0,256)
    x=tl.load(base+offsets).to(tl.float32)
    w=tl.load(weight+offsets).to(tl.float32)
    var=tl.sum(x*x,axis=0)/256
    inv_rms=tl.rsqrt(var+1e-6)
    normalized=(x*inv_rms*(w+1.0)).to(tl.bfloat16).to(tl.float32)
    tl.store(output+offsets,normalized,offsets>=64)
    rot=tl.arange(0,32)
    x1=tl.load(base+rot).to(tl.float32)
    x2=tl.load(base+32+rot).to(tl.float32)
    w1=tl.load(weight+rot).to(tl.float32)
    w2=tl.load(weight+32+rot).to(tl.float32)
    x1=(x1*inv_rms*(w1+1.0)).to(tl.bfloat16).to(tl.float32)
    x2=(x2*inv_rms*(w2+1.0)).to(tl.bfloat16).to(tl.float32)
    pos=tl.load(Positions+token).to(tl.int64)
    cos=tl.load(Rope+pos*ropes+rot).to(tl.float32)
    sin=tl.load(Rope+pos*ropes+32+rot).to(tl.float32)
    tl.store(output+rot,x1*cos-x2*sin)
    tl.store(output+32+rot,x2*cos+x1*sin)
    if is_k:
        # Integer pointers preserve even NaN payloads and signed zero in V.
        value=tl.load(V+token*vs+local_head*256+offsets)
        dest=tl.where(own,VH+(prefix+start+token)*256,Send+token*2560+2304)
        tl.store(dest.to(tl.pointer_type(tl.int16))+offsets,value)
    else:
        gate=tl.load(base+256+offsets)
        tl.store(Gate+token*4096+local_head*256+offsets,gate)
    if ENABLE_PDL:tl.extra.cuda.gdc_launch_dependents()


def norm_pack(qg,k,v,qw,kw,rope,positions,qa,kh,vh,send,rank,start,prefix=0):
    _check_reference()
    n=len(qg)
    gate=torch.empty(n,16,256,dtype=qg.dtype,device=qg.device)
    if n:
        _norm_pack[(n,18)](qg,k,v.view(torch.int16),qa,kh,vh,send,gate,qw,kw,rope,positions,
                          qg.stride(0),k.stride(0),v.stride(0),rope.stride(0),start,prefix,
                          rank,_ENABLE_PDL,num_warps=4)
    return gate

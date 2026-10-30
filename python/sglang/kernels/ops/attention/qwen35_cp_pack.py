"""Exact BF16 bit copies around CP attention's existing send/recv calls."""
import triton
import triton.language as tl
import torch


@triton.jit
def _pack(Q,K,V,QA,KH,VH,Send,N,qs,ks,vs,start,prefix,
          R:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    row,col=i//2560,i%2560
    mq=(i<N*2560)&(col<2048)
    mk=(i<N*2560)&(col>=2048)&(col<2304)
    mv=(i<N*2560)&(col>=2304)
    q=tl.load(Q+row*qs+R*2048+col,mq,other=0)
    k=tl.load(K+row*ks+R*256+col-2048,mk,other=0)
    v=tl.load(V+row*vs+R*256+col-2304,mv,other=0)
    tl.store(QA+(start+row)*2048+col,q,mq)
    tl.store(KH+(prefix+start+row)*256+col-2048,k,mk)
    tl.store(VH+(prefix+start+row)*256+col-2304,v,mv)
    q=tl.load(Q+row*qs+(1-R)*2048+col,mq,other=0)
    k=tl.load(K+row*ks+(1-R)*256+col-2048,mk,other=0)
    v=tl.load(V+row*vs+(1-R)*256+col-2304,mv,other=0)
    tl.store(Send+i,tl.where(col<2048,q,tl.where(col<2304,k,v)),i<N*2560)


@triton.jit
def _unpack(Recv,QA,KH,VH,N,start,prefix,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    row,col=i//2560,i%2560
    value=tl.load(Recv+i,i<N*2560,other=0)
    tl.store(QA+(start+row)*2048+col,value,(i<N*2560)&(col<2048))
    tl.store(KH+(prefix+start+row)*256+col-2048,value,(i<N*2560)&(col>=2048)&(col<2304))
    tl.store(VH+(prefix+start+row)*256+col-2304,value,(i<N*2560)&(col>=2304))


@triton.jit
def _combine(Local,Peer,Out,N,R:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    row,col=i//4096,i%4096
    own=(col//2048)==R
    index=row*2048+col%2048
    a=tl.load(Local+index,(i<N*4096)&own,other=0)
    b=tl.load(Peer+index,(i<N*4096)&~own,other=0)
    tl.store(Out+i,tl.where(own,a,b),i<N*4096)


def pack(q,k,v,qa,kh,vh,send,rank,start,prefix=0):
    n=len(send)
    if n:
        _pack[(triton.cdiv(n*2560,4096),)](
            q.view(torch.int16),k.view(torch.int16),v.view(torch.int16),
            qa.view(torch.int16),kh.view(torch.int16),vh.view(torch.int16),send.view(torch.int16),
            n,q.stride(0),k.stride(0),v.stride(0),start,prefix,rank,4096,num_warps=4)


def unpack(recv,qa,kh,vh,start,prefix=0):
    n=len(recv)
    if n:
        _unpack[(triton.cdiv(n*2560,4096),)](
            recv.view(torch.int16),qa.view(torch.int16),kh.view(torch.int16),vh.view(torch.int16),
            n,start,prefix,4096,num_warps=4)


def combine(local,peer,out,rank):
    n=len(out)
    if n:
        _combine[(triton.cdiv(n*4096,4096),)](
            local.view(torch.int16),peer.view(torch.int16),out.view(torch.int16),
            n,rank,4096,num_warps=4)
    return out

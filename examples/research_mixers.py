"""Sourced SUBSTITUTE ideas, not recovered EvoScientist top-5.

Source: ShinjuKim/ai4scienceframework/outputs/
run_20260814_150104_Imagining_that_you_are_a_Deep_/final_report.md
and 03_step3_HypothesisGenerator.md in the read-only laboratory archive.
Causal mixer-only interpretations below are newly implemented for this harness.
No historical performance/novelty claims are adopted. Frozen training is unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from proxy_tasks.common_set.mixers import CausalSelfAttention, CausalLinearAttention, LinearScan


class AdaptiveLocalGlobal(nn.Module):
    """Interpretation: causal depthwise k=7 convolution gated with full causal attention."""
    def __init__(self,d_model,num_heads,l_max):
        super().__init__();self.attention=CausalSelfAttention(d_model,num_heads)
        self.conv=nn.Conv1d(d_model,d_model,7,padding=6,groups=d_model,bias=False)
        self.gate=nn.Linear(d_model,1)
    def forward(self,x):
        local=self.conv(x.transpose(1,2))[...,:x.shape[1]].transpose(1,2)
        g=self.gate(x).sigmoid()
        return g*self.attention(x)+(1-g)*local


class TemporalSpecialization(nn.Module):
    """Interpretation: weight-shared paths at token, trailing local, and prefix scales.
    Shared weights preserve the frozen parameter budget; no training curriculum.
    """
    def __init__(self,d_model,num_heads,l_max):
        super().__init__();self.attention=CausalSelfAttention(d_model,num_heads)
    def forward(self,x):
        prefix=x.cumsum(1)/torch.arange(1,x.shape[1]+1,device=x.device)[None,:,None]
        shifted=F.pad(x[:,:-4],(0,0,4,0)) if x.shape[1]>4 else torch.zeros_like(x)
        local=x-shifted
        return (self.attention(x)+self.attention(prefix)+self.attention(local))/3


class SSMLinearHybrid(nn.Module):
    """Interpretation: stable diagonal recurrent memory feeds causal linear attention."""
    def __init__(self,d_model,num_heads,l_max):
        super().__init__();self.attention=CausalLinearAttention(d_model,num_heads)
        self.decay=nn.Parameter(torch.full((d_model,),2.0))
    def forward(self,x):
        a=self.decay.sigmoid()[None,None,:].expand_as(x)
        h=LinearScan.apply(a,(1-a)*x)
        return self.attention(h)


class DynamicCausalGraph(CausalSelfAttention):
    """Interpretation: learned bounded q/k define dense soft directed prefix edges.
    This is a causal soft-graph proxy, not a reproduction of a compositional GNN.
    """
    def __init__(self,d_model,num_heads,l_max):super().__init__(d_model,num_heads)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=(z.view(b,t,self.num_heads,self.head_dim).transpose(1,2) for z in self.qkv(x).chunk(3,-1))
        q,k=self.rope(q,k)
        y=F.scaled_dot_product_attention(q.tanh(),k.tanh(),v,is_causal=True,dropout_p=0)
        return self.output(y.transpose(1,2).reshape(b,t,d))


class RelationalPrefixMemory(CausalSelfAttention):
    """Interpretation: four fixed relation slots pool values causally by prefix sums.
    Slot routing uses learned q/k. No global pooling of unseen future tokens.
    """
    def __init__(self,d_model,num_heads,l_max):
        super().__init__(d_model,num_heads)
        slot=torch.sin(torch.arange(1,5).float()[:,None]*torch.arange(1,self.head_dim+1).float()[None,:])
        self.register_buffer('slots',F.normalize(slot,dim=-1))
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=(z.view(b,t,self.num_heads,self.head_dim).transpose(1,2) for z in self.qkv(x).chunk(3,-1))
        weights=(k@self.slots.T).softmax(-1)
        memory=(weights[...,None]*v[:,:,:,None,:]).cumsum(2)
        memory=memory/weights.cumsum(2)[...,None].clamp_min(1e-6)
        read=(q@self.slots.T).softmax(-1)
        y=(read[...,None]*memory).sum(-2)
        return self.output(y.transpose(1,2).reshape(b,t,d))


class TokenwiseControl(nn.Module):
    """Negative control: learned tokenwise map, exactly Transformer mixer parameter count.
    It cannot transfer information across token positions; this is not literal identity.
    """
    def __init__(self,d_model,num_heads,l_max):
        super().__init__();self.up=nn.Linear(d_model,2*d_model,bias=False);self.down=nn.Linear(2*d_model,d_model,bias=False)
    def forward(self,x):return self.down(F.silu(self.up(x)))


class IdentityControl(TokenwiseControl):
    """Literal identity output, with matched registered parameters (inactive by design).
    These parameters are unused/zero-gradient; report explicitly flags this control.
    """
    def forward(self,x):return x+0*self.down(self.up(x))


class FutureLeakControl(CausalSelfAttention):
    def __init__(self,d_model,num_heads,l_max):super().__init__(d_model,num_heads)
    def forward(self,x):return super().forward(x)+x.mean(1,keepdim=True)

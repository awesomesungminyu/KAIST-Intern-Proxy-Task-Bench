"""Tier 2 extensions using the archived Genomics and Bach task definitions."""
import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F

from proxy_tasks.final16.base import ArrayCls, Part


class Candidate:
    def __init__(self, spec, data):
        self.spec = spec
        self.data = Path(data)
        self.name = spec['id']
        self.l_max = spec['length']
        self.batch_size = spec['batch']
        self.steps = spec['steps']
        self.score_name = spec['metric']
        self.direction = 'max' if spec['direction']=='maximize' else 'min'

    def describe(self):
        return self.spec

    def arrays(self, device, keys):
        with np.load(self.data/'arrays.npz', allow_pickle=False) as archive:
            return [torch.from_numpy(archive[k]).to(device) for k in keys]

class Classification(Candidate, ArrayCls):
    def __init__(self,spec,data):
        super().__init__(spec,data)
        self.in_kind = 'tok'
        self.in_dim = spec['vocab']
        self.out_dim = spec['classes']
        self.pad_id = 0
        self.eval_bs = 64

    def load(self,device):
        self.xtr,self.ytr,self.xva,self.yva = self.arrays(device,('x_train','y_train','x_val','y_val'))

    @torch.no_grad()
    def evaluate(self, model):
        return {'score': float(self._acc(model,self.xva,self.yva))}

class Chorales(Candidate, Part):
    in_kind,readout,in_dim,out_dim = 'vec','all',128,128

    def load(self,device):
        self.tr,self.va = self.arrays(device,('roll_train','roll_val'))
        with np.load(self.data/'arrays.npz',allow_pickle=False) as a:
            lt,lv=a['lengths_train'].tolist(),a['lengths_val'].tolist()
        def indices(lengths,train):
            offset=0
            windows=[]
            for length in lengths:
                # Train on full contexts where available. Short chorales and
                # validation tails carry an explicit loss mask.
                starts=range(max(1,length-self.l_max)) if train else range(0,length-1,self.l_max)
                windows.extend((offset+j, min(self.l_max,length-1-j)) for j in starts)
                offset+=length
            return torch.tensor(windows,dtype=torch.long,device=device)
        self.tridx,self.vaidx=indices(lt,True),indices(lv,False)
        self.offset=torch.arange(self.l_max+1,device=device)

    def window(self,roll,index):
        start,length=index[:,0:1],index[:,1:2]
        frames=roll[start+torch.minimum(self.offset[None,:],length)].float()
        return {'x':frames[:,:-1],'y':frames[:,1:],'valid':self.offset[None,:-1]<length}

    def sample(self,g):
        i=torch.randint(len(self.tridx),(self.batch_size,),generator=g,device=self.tr.device)
        return self.window(self.tr,self.tridx[i])

    def frame_nll(self,out,batch):
        return (F.binary_cross_entropy_with_logits(out,batch['y'],reduction='none').sum(-1)*batch['valid']).sum()

    def loss(self,out,batch):
        return self.frame_nll(out,batch)/batch['valid'].sum()

    @torch.no_grad()
    def evaluate(self,model):
        loss,count=0.,0
        for inds in self.vaidx.split(64):
            b=self.window(self.va,inds)
            loss+=self.frame_nll(model(b['x']),b).item()
            count+=int(b['valid'].sum())
        return {'score':loss/count}

ADAPTERS = {"promoters": Classification, "bach": Chorales}

"""Executable frozen inference: callback-based view purchases, no test labels."""
import time
import numpy as np
import torch
from .routing import execute_request


class FrozenCascadeService:
    def __init__(self,model,shape=(26,8,4),rank='margin',veto='none',geometry='standard',cheap_weight=.3):
        if model.training:raise ValueError('Expected eval mode')
        from .preprocess import transform_for
        self.model=model;self.device=next(model.parameters()).device;self.shape=tuple(shape)
        self.rank=rank;self.veto=veto;self.weight=cheap_weight
        self.low=transform_for(224,256)
        self.high=transform_for(288,320 if geometry=='tight' else 288)
        self.refine=transform_for(320,352 if geometry in ('tight','hybrid') else 320)
        if geometry not in ('standard','tight','hybrid'):raise ValueError(geometry)
    @torch.no_grad()
    def __call__(self,images):
        if self.model.training or not 0<len(images)<=64:raise ValueError('One eval request of 1..64 images')
        start=time.perf_counter();calls=[];store={}
        def forward(xs):
            return self.model(xs.to(self.device)).float()
        z0=forward(torch.stack([self.low(im) for im in images]));p0=z0.softmax(1).cpu().numpy()
        def one(ids):
            x=torch.stack([self.high(images[i]) for i in ids]).to(self.device);z=forward(x)
            store.update(ids=ids,x=x,z=z);calls.append(('high288',len(ids)))
            return z.softmax(1).cpu().numpy()
        def flip(ids):
            remap={int(i):j for j,i in enumerate(store['ids'])};local=[remap[int(i)] for i in ids]
            zf=forward(torch.flip(store['x'][local],dims=[3]));calls.append(('flip288',len(ids)))
            # Exact intended cache-writer action; reuse high logits already purchased.
            return (.5*(store['z'][local]+zf)).softmax(1).cpu().numpy()
        def refine(ids):
            z=forward(torch.stack([self.refine(images[i]) for i in ids]));calls.append(('refine320',len(ids)))
            return z.softmax(1).cpu().numpy()
        final,trace=execute_request(p0,one,flip,refine,self.shape,self.rank,self.veto,self.weight)
        return {'probabilities':final,'predictions':final.argmax(1),'calls':calls,
          'selected':{k:trace[k].tolist() for k in ('s1','s2','s3')},
          'elapsed_ms':1000*(time.perf_counter()-start),
          'scope':'Includes transforms, transfers, selection, model and fusion; excludes decode and prior request queue. No CUDA Graph claim.'}


def service_from_gate(model,decision):
    return FrozenCascadeService(model,shape=(int(decision['k']),0,0),cheap_weight=float(decision['cheap_weight']))

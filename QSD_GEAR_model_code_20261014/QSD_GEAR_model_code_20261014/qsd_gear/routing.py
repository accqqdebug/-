"""Label-free frozen GEAR request selection and probability fusion."""

import numpy as np

from .ops import margin, js

def priority(current,cheap,view,rank):
    score=margin(current)
    if rank=='margin':return score
    if rank=='changed_first':return score-2*(current.argmax(1)!=cheap.argmax(1))
    if rank=='disagree_first':return score-2*(view.argmax(1)!=cheap.argmax(1))
    if rank=='js':return score-.25*js(cheap,view)
    raise ValueError(rank)

def execute_request(p0,one,flip,refine,shape=(26,8,4),rank='margin',veto='none',cheap_weight=.3):
    """Only selected view probabilities affect output. All decisions are label-free.

    pf follows EXISTING cache semantics: softmax of mean(original,flip) logits.
    For k2=0,k3>0, stage3 selects inside S1 and refines stage1 directly.
    """
    n=len(p0);take=lambda k:int(np.floor(k*n/64+.5));k1,k2,k3=shape
    assert n<=64 and k1<=64 and k2<=k1 and k3<=(k2 if k2 else k1)
    pred=p0.copy();s1=np.argsort(margin(p0),kind='stable')[:take(k1)]
    s2=s3=np.empty(0,dtype=int);stage1=pred.copy();stage2=pred.copy()
    view1=one(s1) if len(s1) else p0[s1]
    pred[s1]=cheap_weight*p0[s1]+(1-cheap_weight)*view1;stage1=pred.copy()
    view1full=np.zeros_like(p0);view1full[s1]=view1
    view2full=np.zeros_like(p0)
    if k2:
        local=np.argsort(priority(pred[s1],p0[s1],view1,rank),kind='stable')[:take(k2)]
        s2=s1[local];view2=flip(s2) if len(s2) else p0[s2];view2full[s2]=view2
        pred[s2]=.3*pred[s2]+.7*(.2*p0[s2]+.8*view2)
        changed=stage1[s2].argmax(1)!=p0[s2].argmax(1)
        if veto=='top1':reject=changed&(view2.argmax(1)==p0[s2].argmax(1))
        elif veto=='pairwise':
            r=np.arange(len(s2));reject=changed&(view2[r,p0[s2].argmax(1)]>view2[r,stage1[s2].argmax(1)])
        elif veto=='none':reject=np.zeros(len(s2),bool)
        else:raise ValueError(veto)
        pred[s2[reject]]=p0[s2[reject]]
    stage2=pred.copy()
    if k3:
        parent=s2 if k2 else s1;last=view2full[parent] if k2 else view1full[parent]
        local=np.argsort(priority(pred[parent],p0[parent],last,rank),kind='stable')[:take(k3)]
        s3=parent[local]
        if len(s3):pred[s3]=.7*pred[s3]+.3*refine(s3)
    return pred,{'s1':s1,'s2':s2,'s3':s3,'stage1':stage1,'stage2':stage2}

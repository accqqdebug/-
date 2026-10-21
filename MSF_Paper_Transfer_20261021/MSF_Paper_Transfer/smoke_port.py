import os
import sys
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'deps'), str(ROOT / 'msf-paper-port')]
import torch
torch.set_num_threads(2)
from yolox.models.paper_transfer import PaperTransfer
from yolox.models.paper_components import fcbr, kcl

results = {'device': 'cpu', 'purpose': 'synthetic integration checks, NOT performance evaluation', 'checks': []}
def record(name, value):
    assert value, name
    results['checks'].append(name)

torch.manual_seed(1)
r, c, g = [torch.randn(2, 2, 8, 8, 8, requires_grad=True) for _ in range(3)]
loss = fcbr(r, c, g)
loss.backward()
record('FCBR finite, both real detection branches receive gradients; reference direct path detached',
       torch.isfinite(loss).item() and r.grad is None and c.grad.abs().sum().item() > 0 and g.grad.abs().sum().item() > 0)
p = torch.randn(1, 2, 7, 4, 4, requires_grad=True)
record('T=1 KCL zero', kcl([p]).item() == 0)
for dataset, channels in [('gen1', 2), ('coco', 3)]:
    for variant in ['baseline', 'paper', 'paper_binary']:
        start = time.time()
        model = PaperTransfer(dataset, variant, time_steps=2, in_channels=channels)
        model.eval()
        x = torch.randn(1, 2, channels, 64, 64)
        with torch.no_grad():
            pred = model(x)
            pred2 = model(x)
        record(dataset + '/' + variant + ' prediction shape and finite',
               pred.shape == (1, 84, 7 if dataset == 'gen1' else 85) and torch.isfinite(pred).all().item())
        record(dataset + '/' + variant + ' no cross-sample recurrent leakage', torch.equal(pred, pred2))
        if variant != 'baseline':
            model.train()
            model.epoch = 10
            # Real YOLOX assignment + loss + backward, not isolated dummy auxiliary loss.
            target = torch.zeros(1, 2, 5)
            target[0, 0] = torch.tensor([0, 30, 30, 18, 18])
            losses = model(x, target)
            losses['total_loss'].backward()
            record(dataset + '/' + variant + ' full detection+aux backward finite',
                   torch.isfinite(losses['total_loss']).item() and all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
            decay_grads = [p.grad for n, p in model.named_parameters() if 'decay_w' in n and p.grad is not None]
            record(dataset + '/' + variant + ' learned decay connected', len(decay_grads) > 0)
        print(dataset, variant, tuple(pred.shape), 'seconds', round(time.time() - start, 2), flush=True)
        del model
results['status'] = 'passed'
(ROOT / 'port_cpu_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))

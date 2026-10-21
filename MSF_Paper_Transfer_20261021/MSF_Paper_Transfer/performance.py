"""Matched MSF performance pilot. Standard GEN1 histograms or genuine COCO2017."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'deps'), str(ROOT / 'msf-paper-port')]
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from yolox.models.paper_transfer import PaperTransfer
from yolox.utils import postprocess


class DetectionData(Dataset):
    def __init__(self, root, dataset, split, size=320, steps=5):
        self.root, self.dataset, self.split = Path(root), dataset, split
        self.size, self.steps = size, steps
        if dataset == 'gen1':
            if not 1 <= steps <= 10:
                raise ValueError('GEN1 source has ten temporal bins: require 1<=T<=10')
            self.entries = (self.root / 'images' / (split + '.txt')).read_text().splitlines()
            self.cat_ids = [0, 1]
            self.annotations = None
        else:
            annotation = self.root / 'annotations' / ('instances_' + split + '2017.json')
            self.annotations = json.loads(annotation.read_text())
            self.entries = self.annotations['images']
            self.cat_ids = sorted(c['id'] for c in self.annotations['categories'])
            self.labels = {}
            for a in self.annotations['annotations']:
                if a.get('iscrowd', 0):
                    continue
                x, y, w, h = a['bbox']
                if w > 0 and h > 0:
                    self.labels.setdefault(a['image_id'], []).append([self.cat_ids.index(a['category_id']), x+w/2, y+h/2, w, h])

    def __len__(self):
        return len(self.entries)

    def source(self, index):
        if self.dataset == 'gen1':
            path = self.root / 'images' / self.entries[index]
            hist = np.load(str(path))
            if hist.shape != (20, 240, 304):
                raise ValueError('unexpected GEN1 representation: ' + str(hist.shape))
            assignment = np.minimum(np.arange(10) * self.steps // 10, self.steps - 1)
            x = np.stack([np.stack([hist[:10][assignment == t].sum(0), hist[10:][assignment == t].sum(0)]) for t in range(self.steps)]).astype(np.float32)
            # Fixed histogram scale, shared across all arms; no per-image fitted transform.
            x /= 255.
            label = self.root / 'labels' / Path(self.entries[index]).with_suffix('.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            rows = np.loadtxt(str(label), dtype=np.float32).reshape(-1, 5) if label.stat().st_size else np.empty((0, 5), np.float32)
            h, w = hist.shape[-2:]
            rows = rows.copy()
            rows[:, 1:] *= np.array([w, h, w, h], np.float32)
        else:
            entry = self.entries[index]
            path = self.root / (self.split + '2017') / entry['file_name']
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape[:2]
            x = np.repeat(image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None], self.steps, axis=0) / 255.
            rows = np.array(self.labels.get(entry['id'], []), np.float32).reshape(-1, 5)
        return x, rows, h, w

    def __getitem__(self, index):
        x, rows, h, w = self.source(index)
        ratio = min(self.size / h, self.size / w)
        nh, nw = round(h*ratio), round(w*ratio)
        canvas = np.zeros((len(x), x.shape[1], self.size, self.size), dtype=np.float32)
        for t in range(len(x)):
            canvas[t, :, :nh, :nw] = cv2.resize(x[t].transpose(1, 2, 0), (nw, nh)).transpose(2, 0, 1)
        targets = rows.copy()
        targets[:, 1:] *= ratio
        return torch.from_numpy(canvas), torch.from_numpy(targets), index, ratio

    def ground_truth(self):
        if self.dataset == 'coco':
            return self.annotations
        images, annotations = [], []
        for i in range(len(self)):
            # Ground truth needs labels and native sizes, not costly image reloading.
            label = self.root / 'labels' / Path(self.entries[i]).with_suffix('.txt')
            rows = np.loadtxt(str(label), dtype=np.float32).reshape(-1, 5) if label.stat().st_size else np.empty((0, 5), np.float32)
            images.append({'id': i, 'width': 304, 'height': 240})
            for cls, xc, yc, w, h in rows:
                w, h, xc, yc = float(w*304), float(h*240), float(xc*304), float(yc*240)
                annotations.append({'id': len(annotations)+1, 'image_id': i, 'category_id': int(cls), 'bbox': [xc-w/2, yc-h/2, w, h], 'area': w*h, 'iscrowd': 0})
        return {'images': images, 'annotations': annotations, 'categories': [{'id': i, 'name': str(i)} for i in range(2)], 'info': {}}


def collate(batch):
    x, rows, ids, ratios = zip(*batch)
    target = torch.zeros(len(rows), max(1, max(map(len, rows))), 5)
    for i, row in enumerate(rows):
        target[i, :len(row)] = row
    return torch.stack(x), target, ids, ratios


@torch.no_grad()
def evaluate(model, data, batch, workers, device, out):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    predictions = []
    for x, _, ids, ratios in DataLoader(data, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate):
        output = postprocess(model(x.to(device)), len(data.cat_ids), .001, .65, class_agnostic=False)
        for i, scale, pred in zip(ids, ratios, output):
            if pred is None:
                continue
            image_id = i if data.dataset == 'gen1' else data.entries[i]['id']
            pred = pred.cpu().numpy()
            pred = pred[np.argsort(-(pred[:, 4] * pred[:, 5]))[:100]]
            for row in pred:
                x1, y1, x2, y2 = row[:4] / scale
                predictions.append({'image_id': image_id, 'category_id': data.cat_ids[int(row[6])],
                                    'bbox': [float(x1), float(y1), float(max(x2-x1, 0)), float(max(y2-y1, 0))], 'score': float(row[4]*row[5])})
    out.write_text(json.dumps(predictions))
    gt = COCO()
    gt.dataset = data.ground_truth()
    gt.createIndex()
    if predictions:
        det = gt.loadRes(predictions)
    else:
        det = COCO()
        det.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'], 'annotations': []}
        det.createIndex()
    evaluator = COCOeval(gt, det, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(zip(['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large'], map(float, evaluator.stats)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['gen1','coco'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--variant', choices=['baseline','paper','paper_binary'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--lr', type=float, default=.00125)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-batches', type=int, default=0, help='debug only; nonzero disqualifies performance claims')
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    if args.device.startswith('cuda') and os.environ.get('MSF_CUDA_MEMORY_FRACTION'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['MSF_CUDA_MEMORY_FRACTION']))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['protocol'] = 'matched pilot, NOT author-published recipe reproduction; no external pretrained weights; no augmentation'
    config['gen1_evaluator'] = 'COCOeval on existing standard split labels, NOT official Prophesee evaluator'
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    train = DetectionData(args.data, args.dataset, 'train', args.size, args.steps)
    val = DetectionData(args.data, args.dataset, 'val', args.size, args.steps)
    split_hash = hashlib.sha256(json.dumps(train.entries, sort_keys=True).encode()).hexdigest()
    (out / 'data_manifest.json').write_text(json.dumps({'train': len(train), 'val': len(val), 'train_index_sha256': split_hash}, indent=2))
    model = PaperTransfer(args.dataset, args.variant, args.steps, 2 if args.dataset == 'gen1' else 3).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=5e-4)
    # FP32 first: do not confound a new neuron/attention port with AMP instability.
    start, best = 0, -1.
    last = out / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location=args.device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch']+1, state['best']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if 'cuda_rng' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start, args.epochs):
        model.train()
        model.epoch = epoch + 1
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train, batch_size=args.batch, shuffle=True, generator=generator, num_workers=args.workers, collate_fn=collate, drop_last=False)
        lr = args.lr * min((epoch+1)/5., 1.) * (.05 + .95 * .5*(1+math.cos(math.pi*epoch/args.epochs)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        total, batches, begin = 0., 0, time.time()
        for i, (x, target, _, _) in enumerate(loader):
            losses = model(x.to(args.device), target.to(args.device))
            loss = losses['total_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            (loss / args.accumulate).backward()
            if (i+1) % args.accumulate == 0 or i+1 == len(loader) or (args.max_batches and i+1 >= args.max_batches):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            batches += 1
            if i % 100 == 0:
                print(json.dumps({'epoch': epoch+1, 'batch': i, 'loss': loss.item(), 'lr': lr}), flush=True)
            if args.max_batches and batches >= args.max_batches:
                break
        row = {'epoch': epoch+1, 'loss': total/batches, 'seconds': time.time()-begin}
        if not args.max_batches and ((epoch+1) % args.eval_every == 0 or epoch+1 == args.epochs):
            row.update(evaluate(model, val, args.batch, args.workers, args.device, out / 'val_predictions.json'))
        improved = row.get('AP', -1) > best
        if improved:
            best = row['AP']
        state = {'epoch': epoch, 'best': best, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'config': config}
        torch.save(state, out / 'last.tmp.pt')
        os.replace(str(out / 'last.tmp.pt'), str(last))
        if improved:
            torch.save(state, out / 'best.pt')
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    (out / 'completed.json').write_text(json.dumps({'epochs': args.epochs, 'best_val_AP': best, 'debug_only': bool(args.max_batches)}, indent=2))


if __name__ == '__main__':
    main()

"""Evaluate a performance.py checkpoint on GEN1 or COCO using the same code path."""
import argparse
import json
from pathlib import Path
from performance import torch, DetectionData, PaperTransfer, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='val', choices=['val','test'])
    parser.add_argument('--out', default='evaluation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    # Only load trusted project checkpoints: torch checkpoints use pickle.
    try:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(args.checkpoint, map_location='cpu')
    config = state['config']
    if config['dataset'] == 'coco' and args.split == 'test':
        parser.error('COCO test-dev has no public instances ground truth; use val here')
    model = PaperTransfer(config['dataset'], config['variant'], config['steps'], 2 if config['dataset']=='gen1' else 3)
    model.load_state_dict(state['model'])
    model.to(args.device)
    data = DetectionData(args.data, config['dataset'], args.split, config['size'], config['steps'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, data, args.batch, args.workers, args.device, out/'predictions.json')
    (out/'metrics.json').write_text(json.dumps({'metrics':metrics,'split':args.split,'checkpoint':args.checkpoint,'training_config':config},indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

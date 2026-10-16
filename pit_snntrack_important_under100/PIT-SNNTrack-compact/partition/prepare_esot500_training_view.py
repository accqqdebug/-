"""Build SDTrack-compatible views from ESOT500-L train split only."""
import argparse
import json
from pathlib import Path

import numpy as np
from dv import AedatFile


def aligned_boxes(aedat, annotation, records):
    with AedatFile(str(aedat)) as recording:
        origin = int(next(iter(recording['events'].numpy()))['timestamp'][0])
    labels = np.loadtxt(annotation)
    times = np.rint(labels[:, 0] * 1e6).astype(np.int64)
    ends = np.asarray([r['t_end_us'] - origin for r in records])
    right = np.clip(np.searchsorted(times, ends), 0, len(times)-1)
    left = np.clip(right-1, 0, len(times)-1)
    nearest = np.where(np.abs(times[left]-ends) <= np.abs(times[right]-ends), left, right)
    valid = (ends >= times[0]) & (ends <= times[-1]) & (np.abs(times[nearest]-ends) <= 1500)
    boxes = labels[nearest, 1:5].copy()
    boxes[~valid] = 0
    return boxes, int(valid.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--processed-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    names = sorted((args.data_root/'train.txt').read_text().split())
    validation = set(names[::10])
    manifest = {'train': [n for n in names if n not in validation],
                'val': [n for n in names if n in validation], 'sequences': {}}
    for name in names:
        metadata = args.processed_root/name/'uniform/inter4_stack_3008/partition_metadata.jsonl'
        records = [json.loads(line) for line in metadata.read_text().splitlines()]
        boxes, valid = aligned_boxes(args.data_root/'aedat4'/f'{name}.aedat4',
                                     args.data_root/'anno_t'/f'{name}.txt', records)
        assert valid >= 2, name
        split = 'val' if name in validation else 'train'
        manifest['sequences'][name] = {'split': split, 'frames': len(records), 'valid_frames': valid}
        for policy in ('uniform', 'jitter', 'fixed_count'):
            target = args.output_root/policy/split/name
            target.mkdir(parents=True, exist_ok=True)
            np.savetxt(target/'groundtruth_rect.txt', boxes, delimiter=',', fmt='%.5f')
            link = target/'inter4_stack_3008'
            source = args.processed_root/name/policy/'inter4_stack_3008'
            assert source.exists(), source
            if not link.exists(): link.symlink_to(source, target_is_directory=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'train_sequences':len(manifest['train']),
                      'val_sequences':len(manifest['val']),
                      'valid_frames':sum(r['valid_frames'] for r in manifest['sequences'].values())}))


if __name__ == '__main__': main()

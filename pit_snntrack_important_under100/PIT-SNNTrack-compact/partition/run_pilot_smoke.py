"""Integration check for SDTrack T4D1 on unlabelled raw-event examples.

The initial box is user supplied. Results are never benchmark AUC/PR evidence.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.test.evaluation.tracker import reset_LIF_layers
from lib.test.parameter.SDTrack import parameters
from lib.test.tracker.SDTrack import SDTrack


def read_stack(folder, index):
    images = []
    for bin_index in range(1, 5):
        path = folder / f"{index:04d}_{bin_index}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(images)


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--init-box", type=float, nargs=4, default=[100, 80, 80, 80])
    args = parser.parse_args()

    params = parameters("SDTrack-tiny-fe108-T4D1")
    params.debug = 0
    uniform = args.data / "uniform" / "inter4_stack_3008"
    template = read_stack(uniform, 1)
    results = {}
    for policy in ("uniform", "jitter", "uniform_repeat"):
        tracker = SDTrack(params, "eotb")
        tracker.initialize(template, {"init_bbox": args.init_box})
        boxes = [args.init_box]
        source_policy = "uniform" if policy == "uniform_repeat" else policy
        folder = args.data / source_policy / "inter4_stack_3008"
        for frame in range(2, args.frames + 1):
            box = tracker.track(read_stack(folder, frame))["target_bbox"]
            reset_LIF_layers(tracker.network)
            boxes.append(box)
        results[policy] = boxes
        del tracker
        torch.cuda.empty_cache()
    overlaps = [iou_xywh(a, b) for a, b in zip(results["uniform"], results["jitter"])]
    repeat_error = np.max(np.abs(np.asarray(results["uniform"]) - np.asarray(results["uniform_repeat"])))
    result = {
        "status": "unlabelled_integration_check",
        "init_box": args.init_box,
        "frames": args.frames,
        "mean_pairwise_iou": float(np.mean(overlaps[1:])),
        "minimum_pairwise_iou": float(np.min(overlaps[1:])),
        "uniform_repeat_max_box_difference": float(repeat_error),
        "boxes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "boxes"}))


if __name__ == "__main__":
    main()

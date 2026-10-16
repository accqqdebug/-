"""Render controlled four-bin event inputs for the SDTrack T4D1 tracker.

For each APS frame interval, every policy uses exactly the same events and
start/end timestamps. The output names match SDTrack's inter4_stack_3008 reader.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from dv import AedatFile


def event_images(events, height, width, amount):
    valid = (
        (events["x"] > 0) & (events["x"] < width)
        & (events["y"] > 0) & (events["y"] < height)
    )
    events = events[valid]
    flat = events["y"].astype(np.int64) * width + events["x"]
    positive = events["polarity"].astype(bool)
    pos = np.bincount(flat[positive], minlength=height * width)
    neg = np.bincount(flat[~positive], minlength=height * width)
    pos = np.minimum(pos * amount, 255).astype(np.uint8).reshape(height, width)
    neg = np.minimum(neg * amount, 255).astype(np.uint8).reshape(height, width)
    return pos, neg


def boundaries(policy, start, end, timestamps, jitter_strength=None, rng=None, polarities=None, jitter_cut="all"):
    if policy == "uniform":
        fractions = np.array([0, 0.25, 0.5, 0.75, 1.0])
        return np.rint(start + (end - start) * fractions).astype(np.int64)
    if policy == "jitter":
        if jitter_strength is None:
            fractions = np.array([0, 0.15, 0.48, 0.82, 1.0])
        else:
            if rng is None:
                raise ValueError("Random jitter requires a seeded RNG")
            # Strength is the maximum displacement as a fraction of one
            # uniform bin width; <=0.5 keeps all four bins ordered.
            fractions = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
            perturbations = rng.uniform(-jitter_strength / 4,
                                        jitter_strength / 4, size=3)
            if jitter_cut == "all":
                fractions[1:4] += perturbations
            else:
                cut = int(jitter_cut)
                if cut not in (1, 2, 3):
                    raise ValueError("jitter_cut must be all, 1, 2, or 3")
                fractions[cut] += perturbations[cut - 1]
        return np.rint(start + (end - start) * fractions).astype(np.int64)
    if policy == "fixed_count":
        if len(timestamps) < 4:
            return boundaries("uniform", start, end, timestamps)
        # A timestamp at a boundary is assigned to the later bin. Duplicate
        # timestamps may yield empty bins, which is recorded in metadata.
        cuts = timestamps[np.minimum(
            (np.arange(1, 4) * len(timestamps) // 4), len(timestamps) - 1
        )]
        return np.array([start, *cuts, end], dtype=np.int64)
    if policy == "polarity_balanced":
        # Pick monotone timestamp cuts that make cumulative positive and
        # negative event fractions as close as possible to 1/4, 1/2, 3/4.
        # Group duplicate timestamps first so equal-time events are never split.
        ts = np.asarray(timestamps, dtype=np.int64)
        if polarities is None:
            raise ValueError("Polarity-balanced cuts require event polarities")
        pol = np.asarray(polarities, dtype=bool)
        unique_ts, inverse = np.unique(ts, return_inverse=True)
        groups = len(unique_ts)
        if groups < 4:
            return boundaries("uniform", start, end, timestamps)
        pos_group = np.bincount(inverse[pol], minlength=groups)
        neg_group = np.bincount(inverse[~pol], minlength=groups)
        pos_cum = np.concatenate(([0], np.cumsum(pos_group, dtype=np.int64)))
        neg_cum = np.concatenate(([0], np.cumsum(neg_group, dtype=np.int64)))
        pos_total = max(int(pos_cum[-1]), 1)
        neg_total = max(int(neg_cum[-1]), 1)
        back = [None]
        previous = np.full(groups + 1, np.inf, dtype=np.float64)
        first = np.arange(1, groups - 2)
        first_cost = ((pos_cum[first] / pos_total - 0.25) ** 2
                      + (neg_cum[first] / neg_total - 0.25) ** 2)
        previous[first] = first_cost
        back.append(np.full(groups + 1, -1, dtype=np.int64))
        for cut in (2, 3):
            prefix_value = np.minimum.accumulate(previous)
            idx = np.arange(groups + 1)
            new_min = np.r_[True, previous[1:] < prefix_value[:-1]]
            prefix_index = np.maximum.accumulate(np.where(new_min, idx, 0))
            current = np.full(groups + 1, np.inf, dtype=np.float64)
            parent = np.full(groups + 1, -1, dtype=np.int64)
            js = np.arange(cut, groups - (4 - cut) + 1)
            parents = prefix_index[js - 1]
            reachable = np.isfinite(prefix_value[js - 1])
            js, parents = js[reachable], parents[reachable]
            target = cut / 4.0
            cost = ((pos_cum[js] / pos_total - target) ** 2
                    + (neg_cum[js] / neg_total - target) ** 2)
            current[js] = prefix_value[js - 1] + cost
            parent[js] = parents
            previous = current
            back.append(parent)
        candidates = np.arange(3, groups)
        last = int(candidates[np.argmin(previous[candidates])])
        cuts_idx = [last]
        for cut in (3, 2):
            cuts_idx.append(int(back[cut][cuts_idx[-1]]))
        cuts_idx.reverse()
        return np.array([start, *(int(unique_ts[j]) for j in cuts_idx), end],
                        dtype=np.int64)
    raise ValueError(policy)


def render(aedat_path, output_dir, policy, max_frames, amount, hidden_amount,
           hidden_decay, jitter_strength=None, seed=None, jitter_cut="all"):
    with AedatFile(str(aedat_path)) as recording:
        height, width = recording["events"].size
        packets = [packet for packet in recording["events"].numpy()]
    with AedatFile(str(aedat_path)) as recording:
        frame_times = np.array(
            [frame.timestamp_start_of_frame for frame in recording["frames"]],
            dtype=np.int64,
        )
    if len(frame_times) < 2:
        raise ValueError("AEDAT4 needs at least two APS frames")
    if not packets:
        raise ValueError("AEDAT4 has no events")
    events = np.concatenate(packets)
    timestamps = events["timestamp"]
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("Event timestamps are not sorted")
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden = np.zeros((height, width), dtype=np.float32)
    last_pos = np.zeros((height, width), dtype=np.uint8)
    last_neg = np.zeros((height, width), dtype=np.uint8)
    first_bin = True
    rendered = 0
    rng = np.random.default_rng(seed) if seed is not None else None
    with (output_dir / "partition_metadata.jsonl").open("w", encoding="utf-8") as metadata:
        end_index = min(len(frame_times) - 1, max_frames) if max_frames else len(frame_times) - 1
        for frame_index in range(1, end_index + 1):
            start = int(frame_times[frame_index - 1])
            end = int(frame_times[frame_index])
            if end <= start:
                raise ValueError("APS frame timestamps must increase")
            lo, hi = np.searchsorted(timestamps, [start, end], side="left")
            frame_events = events[lo:hi]
            edges = boundaries(policy, start, end, frame_events["timestamp"],
                               jitter_strength=jitter_strength, rng=rng,
                               polarities=frame_events["polarity"],
                               jitter_cut=jitter_cut)
            if np.any(np.diff(edges) < 0):
                raise ValueError("Partition boundaries must be monotone")
            indices = np.searchsorted(frame_events["timestamp"], edges, side="left")
            counts = []
            polarity_counts = []
            for bin_index in range(4):
                event_bin = frame_events[indices[bin_index]:indices[bin_index + 1]]
                pos, neg = event_images(event_bin, height, width, amount)
                if first_bin:
                    first_bin = False
                else:
                    hidden = hidden * hidden_decay
                    hidden += hidden_amount * ((last_pos == 0) & (pos != 0))
                    hidden += hidden_amount * ((last_neg == 0) & (neg != 0))
                    np.clip(hidden, 0, 255, out=hidden)
                image = np.stack((pos, neg, hidden.astype(np.uint8)), axis=-1)
                image_path = output_dir / f"{frame_index:04d}_{bin_index + 1}.png"
                if not cv2.imwrite(str(image_path), image):
                    raise OSError(f"Could not write {image_path}")
                last_pos, last_neg = pos, neg
                counts.append(len(event_bin))
                polarity = event_bin["polarity"].astype(bool)
                polarity_counts.append([int(np.count_nonzero(polarity)),
                                        int(np.count_nonzero(~polarity))])
            if sum(counts) != len(frame_events):
                raise AssertionError("Partition lost or duplicated events")
            metadata.write(json.dumps({
                "frame": frame_index, "policy": policy, "t_start_us": start,
                "t_end_us": end, "boundaries_us": edges.tolist(),
                "event_counts": counts, "polarity_counts_pos_neg": polarity_counts,
                "total_events": len(frame_events),
                "jitter_strength": jitter_strength, "jitter_seed": seed,
                "jitter_cut": jitter_cut,
            }) + "\n")
            rendered += 1
    print(json.dumps({"policy": policy, "frames": rendered,
                      "output": str(output_dir), "sensor": [height, width]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aedat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["uniform", "jitter", "fixed_count", "polarity_balanced"], required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--amount", type=int, default=30)
    parser.add_argument("--hidden-amount", type=float, default=30)
    parser.add_argument("--hidden-decay", type=float, default=0.8)
    parser.add_argument("--jitter-strength", type=float, default=None,
                        help="maximum boundary displacement as a fraction of one uniform bin width (0..0.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for independent jitter boundaries")
    parser.add_argument("--jitter-cut", choices=["all", "1", "2", "3"], default="all",
                        help="perturb all three cuts or only one internal cut")
    args = parser.parse_args()
    if args.jitter_strength is not None:
        if args.policy != "jitter":
            parser.error("--jitter-strength is valid only with --policy jitter")
        if not 0 <= args.jitter_strength <= 0.5:
            parser.error("--jitter-strength must be in [0, 0.5]")
        if args.seed is None:
            parser.error("--seed is required with --jitter-strength")
    render(args.aedat, args.output, args.policy, args.max_frames, args.amount,
           args.hidden_amount, args.hidden_decay, args.jitter_strength, args.seed,
           args.jitter_cut)


if __name__ == "__main__":
    main()

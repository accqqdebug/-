"""Paired partition sensitivity pilot on timestamped ESOT500-L annotations."""
import argparse
import json
import hashlib
import cv2
from pathlib import Path

import numpy as np
import torch
from dv import AedatFile

from partition.run_pilot_smoke import read_stack, iou_xywh
from lib.test.evaluation.tracker import reset_LIF_layers
from lib.test.parameter.SDTrack import parameters
from lib.test.tracker.SDTrack import SDTrack
from lib.models.SDTrack.SDTrack_tiny_LIF_T4D1_model import LIF


def nearest_labels(aedat_path, label_path, records):
    with AedatFile(str(aedat_path)) as recording:
        origin_us = int(next(iter(recording['events'].numpy()))['timestamp'][0])
    annotations = np.loadtxt(label_path)
    times_us = np.rint(annotations[:, 0] * 1e6).astype(np.int64)
    end_us = np.asarray([r['t_end_us'] - origin_us for r in records])
    right = np.searchsorted(times_us, end_us)
    right = np.clip(right, 0, len(times_us) - 1)
    left = np.clip(right - 1, 0, len(times_us) - 1)
    closest = np.where(np.abs(times_us[left] - end_us) <= np.abs(times_us[right] - end_us), left, right)
    gap = np.abs(times_us[closest] - end_us)
    valid = (end_us >= times_us[0]) & (end_us <= times_us[-1]) & (gap <= 1500)
    return annotations[closest, 1:5], gap, valid, times_us[closest]


def set_partition_timing(network, record, timing_order='identity'):
    durations = np.diff(np.asarray(record['boundaries_us'], dtype=np.float64))
    if timing_order == 'reverse':
        durations = durations[::-1].copy()
    reference = (record['t_end_us'] - record['t_start_us']) / 4.0
    device = next(network.parameters()).device
    for module in network.modules():
        if isinstance(module, LIF):
            module.set_time_deltas(durations, reference, device)


def read_search_stack(folder, uniform_folder, frame, history_mode="native", event_mode="native"):
    stack = read_stack(folder, frame)
    if folder != uniform_folder and (history_mode == "uniform" or event_mode == "uniform"):
        # Renderer writes BGR=(positive, negative, history); read_stack returns RGB.
        reference = read_stack(uniform_folder, frame)
        if history_mode == "uniform":
            stack[..., 0] = reference[..., 0]
        if event_mode == "uniform":
            stack[..., 1:] = reference[..., 1:]
    return stack


def success_auc(overlaps):
    return float(np.mean([np.mean(overlaps >= t) for t in np.linspace(0, 1, 21)]))


def evaluate(data_root, processed_root, name, max_frames, checkpoint=None, physical_time=False, oracle_reset=False, policies=("uniform", "jitter", "fixed_count"), timing_order="identity", history_mode="native", event_mode="native", pulse_policy=None, pulse_frame=None, pulse_start_index=None, pulse_length=None, pulse_schedule=None, input_delay_steps=0, oracle_reset_interval=1):
    base = processed_root / name
    metadata = {}
    required_metadata = tuple(dict.fromkeys(('uniform',) + tuple(policies)))
    for policy in required_metadata:
        path = base / policy / 'inter4_stack_3008' / 'partition_metadata.jsonl'
        metadata[policy] = [json.loads(line) for line in path.read_text().splitlines()]
    uniform_folder = base / 'uniform' / 'inter4_stack_3008'
    records = metadata['uniform']
    for policy in policies:
        if policy == 'uniform':
            continue
        assert len(metadata[policy]) == len(records)
        assert all((a['frame'], a['t_start_us'], a['t_end_us'], a['total_events']) ==
                   (b['frame'], b['t_start_us'], b['t_end_us'], b['total_events'])
                   for a, b in zip(records, metadata[policy]))
    boxes, gap, valid, label_times_us = nearest_labels(data_root / 'aedat4' / f'{name}.aedat4',
                                                       data_root / 'anno_t' / f'{name}.txt', records)
    ids = np.flatnonzero(valid)
    if max_frames:
        ids = ids[:max_frames]
    if len(ids) < 2:
        raise ValueError(f'{name}: fewer than two aligned frames')
    start_frame = int(records[ids[0]]['frame'])
    template = read_stack(base / 'uniform' / 'inter4_stack_3008', start_frame)
    initial_box = boxes[ids[0]].tolist()
    predictions = {}
    scores = {}
    for policy in policies:
        params = parameters('SDTrack-tiny-fe108-T4D1')
        params.debug = 0
        if checkpoint is not None:
            params.checkpoint = str(checkpoint)
        tracker = SDTrack(params, 'eotb')
        if physical_time:
            set_partition_timing(tracker.network, records[ids[0]], timing_order)
        folder = base / policy / 'inter4_stack_3008'
        if oracle_reset:
            # Diagnostic only: oracle GT at the previous labelled frame is used to
            # reset tracking state; all policies share the uniform previous-frame template.
            pred = [initial_box]
            template_folder = base / 'uniform' / 'inter4_stack_3008'
            for step, idx in enumerate(ids[1:], start=1):
                previous = ids[step - 1]
                reset_LIF_layers(tracker.network)
                input_idx = ids[min(len(ids)-1, max(0, step - input_delay_steps))]
                if step == 1 or (step - 1) % oracle_reset_interval == 0:
                    prior_template = read_stack(template_folder, int(records[previous]['frame']))
                    reset_LIF_layers(tracker.network)
                    tracker.initialize(prior_template, {'init_bbox': boxes[previous].tolist()})
                frame = int(records[input_idx]['frame'])
                pred.append(tracker.track(read_search_stack(folder, uniform_folder, frame, history_mode, event_mode))['target_bbox'])
                reset_LIF_layers(tracker.network)
        else:
            tracker.initialize(template, {'init_bbox': initial_box})
            pred = [initial_box]
            for step, idx in enumerate(ids[1:], start=1):
                input_idx = ids[min(len(ids)-1, max(0, step - input_delay_steps))]
                frame = int(records[input_idx]['frame'])
                if physical_time:
                    set_partition_timing(tracker.network, metadata[policy][idx], timing_order)
                search_folder = folder
                if pulse_policy is not None:
                    if pulse_schedule is not None:
                        offset = step - pulse_start_index
                        if pulse_schedule == 'clustered20':
                            active = 7 <= offset < 27
                        else:
                            active = any(a <= offset < b for a,b in ((0,5),(10,15),(20,25),(30,35)))
                    elif pulse_start_index is None:
                        active = frame == pulse_frame
                    else:
                        active = pulse_start_index <= step < pulse_start_index + pulse_length
                    search_folder = base / pulse_policy / 'inter4_stack_3008' if active else uniform_folder
                pred.append(tracker.track(read_search_stack(search_folder, uniform_folder, frame, history_mode, event_mode))['target_bbox'])
                reset_LIF_layers(tracker.network)
        del tracker
        torch.cuda.empty_cache()
        pred = np.asarray(pred, dtype=float)
        overlaps = np.asarray([iou_xywh(a, b) for a, b in zip(pred, boxes[ids])])
        predictions[policy] = pred
        scores[policy] = {'mean_iou': float(np.mean(overlaps[1:])),
                          'success_auc_21': success_auc(overlaps[1:]),
                          'center_precision_20px': float(np.mean(np.linalg.norm(
                              pred[1:, :2] + pred[1:, 2:]/2 - boxes[ids[1:], :2] - boxes[ids[1:], 2:]/2,
                              axis=1) <= 20))}
    paired = {}
    for policy in policies[1:]:
        pair_iou = np.asarray([iou_xywh(a, b) for a, b in zip(predictions['uniform'][1:], predictions[policy][1:])])
        paired[policy] = {'mean_prediction_iou': float(np.mean(pair_iou)),
                          'mean_iou_delta': scores[policy]['mean_iou'] - scores['uniform']['mean_iou']}
    return {'sequence': name, 'evaluated_frames': len(ids), 'first_frame': start_frame,
            'max_label_gap_us': int(max(gap[ids])), 'init_box': initial_box,
            'checkpoint': str(checkpoint) if checkpoint is not None else 'default',
            'processed_root': str(processed_root),
            'partition_metadata_sha256': {p: hashlib.sha256((base / p / 'inter4_stack_3008' / 'partition_metadata.jsonl').read_bytes()).hexdigest() for p in policies},
            'torch_cpu_threads': torch.get_num_threads(), 'opencv_cpu_threads': cv2.getNumThreads(),
            'history_mode': history_mode, 'event_mode': event_mode, 'history_channel_rgb_index': 0,
            'physical_time_dynamics': physical_time, 'timing_order': timing_order, 'oracle_reset_diagnostic': oracle_reset, 'oracle_reset_interval': oracle_reset_interval,
                        'pulse_policy': pulse_policy, 'pulse_frame': pulse_frame,
            'pulse_start_index': pulse_start_index, 'pulse_length': pulse_length, 'pulse_schedule': pulse_schedule,
            'input_delay_steps': input_delay_steps,
            'scores': scores, 'paired': paired,
            'frame_numbers': [int(records[i]['frame']) for i in ids],
            'ground_truth_xywh': boxes[ids].tolist(),
            'ground_truth_time_us': label_times_us[ids].tolist(),
            'predictions_xywh': {p: predictions[p].tolist() for p in policies}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--processed-root', type=Path, required=True)
    p.add_argument('--sequences', nargs='+', required=True)
    p.add_argument('--policies', nargs='+', default=['uniform', 'jitter', 'fixed_count'])
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--physical-time', action='store_true')
    p.add_argument('--timing-order', choices=['identity', 'reverse'], default='identity')
    p.add_argument('--history-mode', choices=['native', 'uniform'], default='native')
    p.add_argument('--event-mode', choices=['native', 'uniform'], default='native')
    p.add_argument('--cpu-threads', type=int, default=0)
    p.add_argument('--opencv-threads', type=int, default=None)
    p.add_argument('--oracle-reset', action='store_true', help='validation-only diagnostic with periodic previous-frame GT resets')
    p.add_argument('--oracle-reset-interval', type=int, default=1, help='reset tracker template/box from prior-frame GT every N scored steps in oracle mode')
    p.add_argument('--input-delay-steps', type=int, default=0, help='signed source offset in scored frames: positive is stale input; negative is offline lookahead')
    p.add_argument('--pulse-policy', choices=['jitter','fixed_count'])
    p.add_argument('--pulse-frame', type=int, help='actual frame ID for a one-frame policy pulse, followed by uniform inputs')
    p.add_argument('--pulse-start-index', type=int, help='scored-frame index for the start of a consecutive pulse')
    p.add_argument('--pulse-length', type=int, help='number of consecutive scored frames in a pulse')
    p.add_argument('--pulse-schedule', choices=['clustered20','spaced4x5'], help='20 active frames within a common 35-frame window')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    if args.pulse_policy is None:
        if args.pulse_frame is not None or args.pulse_start_index is not None or args.pulse_length is not None:
            p.error('pulse options require --pulse-policy')
    else:
        single = args.pulse_frame is not None
        window = args.pulse_start_index is not None and args.pulse_length is not None and args.pulse_schedule is None
        schedule = args.pulse_start_index is not None and args.pulse_length is None and args.pulse_schedule is not None
        if sum((single,window,schedule)) != 1 or (args.pulse_start_index is not None and single):
            p.error('specify one pulse form: --pulse-frame, start-index plus length, or start-index plus schedule')
        if args.pulse_length is not None and args.pulse_length < 1:
            p.error('--pulse-length must be positive')
    output = [evaluate(args.data_root, args.processed_root, name, args.max_frames, args.checkpoint, args.physical_time, args.oracle_reset, tuple(args.policies), args.timing_order, args.history_mode, args.event_mode, args.pulse_policy, args.pulse_frame, args.pulse_start_index, args.pulse_length, args.pulse_schedule, args.input_delay_steps, args.oracle_reset_interval) for name in args.sequences]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps([{k:v for k,v in row.items() if k not in ('predictions_xywh','frame_numbers')} for row in output], indent=2))


if __name__ == '__main__':
    main()

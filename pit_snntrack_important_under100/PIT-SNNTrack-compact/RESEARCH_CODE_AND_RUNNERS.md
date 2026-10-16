# Candidate changes and experiment runners

These are source listings for review, not automatically applied patches.

## research_20261011/repo/lib/train/actors/SDTrack.py

```python
import os
from .task_alignment import task_alignment
from . import BaseActor
from .partition_consistency import endpoint_state_consistency
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ...models.SDTrack.SDTrack_tiny_LIF_T4D1_model import LIF


def move_nested_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return type(value)(move_nested_to_device(item, device) for item in value)
    if isinstance(value, tuple):
        return tuple(move_nested_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_nested_to_device(item, device) for key, item in value.items()}
    return value

def reset_LIF_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, LIF):  # 检查是否是 LIF 层
            module.to_zero()


def snapshot_LIF_states(model):
    return [module.v for module in model.modules()
            if isinstance(module, LIF) and isinstance(module.v, torch.Tensor)]


def _unwrap_bin_tensor(value):
    # sequence-mode processing and LTR collate retain one singleton image-list level
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'expected a tensor for one temporal bin, got {type(value).__name__}')
    return value

def pack_four_bins(images):
    bins = tuple(_unwrap_bin_tensor(image) for image in images)
    if len(bins) != 4 or any(x.ndim != 4 for x in bins):
        raise ValueError(f'expected four [C,B,H,W] bin tensors, got {[tuple(x.shape) for x in bins]}')
    return torch.stack(bins, 0).permute(0, 2, 1, 3, 4).contiguous()

class SDTrackActor(BaseActor):
    """ Actor for training SDTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg



    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # Four-bin inputs are nested Python lists, which TensorDict.to does not recurse into.
        for key, value in list(data.items()):
            if key not in ('epoch', 'settings'):
                data[key] = move_nested_to_device(value, self.settings.device)
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        if self.cfg.MODEL.NEURON == 'ILIF' and self.cfg.MODEL.T == 1:
            template_list = []
            for i in range(self.settings.num_template):
                template_img_i = data['template_images'][i].view(-1,
                                                                *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
                template_list.append(template_img_i)

            search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)


            if len(template_list) == 1:
                template_list = template_list[0]
            
            out_dict = self.net(template=template_list,
                                search=search_img)


        elif self.cfg.MODEL.NEURON == 'LIF' and self.cfg.MODEL.T == 4:
            t0 = _unwrap_bin_tensor(data['template_images'][0][0])          # [C, B, h, w]
            t1 = _unwrap_bin_tensor(data['template_images'][0][1])
            t2 = _unwrap_bin_tensor(data['template_images'][0][2])
            t3 = _unwrap_bin_tensor(data['template_images'][0][3])
            data['template_images'] = torch.stack((t0, t1, t2, t3), 0).permute(0, 2, 1, 3, 4).contiguous()

            data['search_images'] = pack_four_bins(data['search_images'][0])

            pair_enabled = (getattr(self.cfg.TRAIN, 'PARTITION_CONSISTENCY', False)
                            and 'paired_search_images' in data)
            if pair_enabled:
                paired_search = pack_four_bins(data['paired_search_images'][0])
                rng_cpu_before = torch.get_rng_state()
                rng_cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                out_dict = self.net(template=data['template_images'], search=data['search_images'])
                primary_states = snapshot_LIF_states(self.net)
                rng_cpu_after = torch.get_rng_state()
                rng_cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                reset_LIF_layers(self.net)
                torch.set_rng_state(rng_cpu_before)
                if rng_cuda_before is not None:
                    torch.cuda.set_rng_state_all(rng_cuda_before)
                paired_out = self.net(template=data['template_images'], search=paired_search)
                paired_states = snapshot_LIF_states(self.net)
                reset_LIF_layers(self.net)
                # Consume stochastic-layer randomness once per training sample, while
                # using identical masks for the two partition views.
                torch.set_rng_state(rng_cpu_after)
                if rng_cuda_after is not None:
                    torch.cuda.set_rng_state_all(rng_cuda_after)
                if len(primary_states) != len(paired_states):
                    raise RuntimeError('partition branches produced different LIF state sets')
                out_dict['_partition_pair_output'] = paired_out
                out_dict['_partition_state_pairs'] = list(zip(primary_states, paired_states))
            else:
                out_dict = self.net(template=data['template_images'],
                                    search=data['search_images'])
                reset_LIF_layers(self.net)

        elif self.cfg.MODEL.NEURON == 'ILIF' and self.cfg.MODEL.T == 2:
            t0 = data['template_images'][0][0]          # [C, B, h, w]
            t1 = data['template_images'][0][1]
            data['template_images'] = torch.stack((t0, t1), 0).permute(0, 2, 1, 3, 4).contiguous()

            s0 = data['search_images'][0][0]
            s1 = data['search_images'][0][1]
            data['search_images'] = torch.stack((s0, s1), 0).permute(0, 2, 1, 3, 4).contiguous()

            
            out_dict = self.net(template=data['template_images'],
                                search=data['search_images'])

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if not torch.isfinite(pred_boxes).all():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        pair = pred_dict.get('_partition_pair_output')
        state_pairs = pred_dict.get('_partition_state_pairs')
        pair_task_loss = None
        box_consistency = None
        state_consistency = None
        if pair is not None:
            pair_boxes = pair['pred_boxes']
            if not torch.isfinite(pair_boxes).all():
                raise ValueError('Paired partition network output is NAN! Stop Training')
            pair_queries = pair_boxes.size(1)
            pair_boxes_vec = box_cxcywh_to_xyxy(pair_boxes).view(-1, 4)
            pair_gt_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, pair_queries, 1)).view(-1, 4).clamp(0.0, 1.0)
            pair_giou_loss, _ = self.objective['giou'](pair_boxes_vec, pair_gt_vec)
            pair_l1_loss = self.objective['l1'](pair_boxes_vec, pair_gt_vec)
            if 'score_map' in pair:
                pair_location_loss = self.objective['focal'](pair['score_map'], gt_gaussian_maps)
            else:
                pair_location_loss = torch.zeros((), device=pair_boxes.device, dtype=pair_boxes.dtype)
            pair_task_loss = (self.loss_weight['giou'] * pair_giou_loss
                              + self.loss_weight['l1'] * pair_l1_loss
                              + self.loss_weight['focal'] * pair_location_loss)
            # Keep the average task-loss scale comparable to the single-view baseline.
            loss = 0.5 * (loss + pair_task_loss)

            box_consistency = F.smooth_l1_loss(pred_boxes, pair_boxes)
            if state_pairs:
                state_consistency = endpoint_state_consistency(state_pairs)
            else:
                state_consistency = torch.zeros((), device=pred_boxes.device, dtype=pred_boxes.dtype)
            loss = (loss
                    + float(getattr(self.cfg.TRAIN, 'PARTITION_BOX_WEIGHT', 0.5)) * box_consistency
                    + float(getattr(self.cfg.TRAIN, 'PARTITION_STATE_WEIGHT', 0.1)) * state_consistency * (min(1.0, max(0.0, (int(gt_dict['epoch']) - 1) / int(os.environ['PIT_STATE_WARMUP']))) if int(os.environ.get('PIT_STATE_WARMUP', '0')) else 1.0))
        alignment_metrics = {}
        state_warmup = int(os.environ.get('PIT_STATE_WARMUP', '0'))
        if state_warmup and pair is not None:
            epoch = int(gt_dict['epoch'])
            state_factor = min(1.0, max(0.0, (epoch - 1) / state_warmup))
            alignment_metrics['Alignment/state_factor'] = state_factor
        mode = os.environ.get('PIT_TASK_ALIGNMENT', 'off')
        if mode != 'off' and pair is not None:
            alignment, alignment_metrics = task_alignment(pred_dict, pair, gt_bbox, mode)
            factor = min(1.0, max(0.0, (int(gt_dict['epoch']) - 1) / 4.0))
            loss = loss + 0.1 * factor * alignment
            alignment_metrics['Alignment/factor'] = factor
            alignment_metrics['Alignment/loss'] = float(alignment.detach())
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Nonfinite tracking/consistency loss')
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if pair_task_loss is not None:
                status.update({"Loss/partition_pair_task": pair_task_loss.item(),
                               "Loss/partition_box_consistency": box_consistency.item(),
                               "Loss/partition_state_consistency": state_consistency.item()})
            status.update(alignment_metrics)
            return loss, status
        else:
            return loss

```

## research_20261011/repo/lib/train/actors/partition_consistency.py

```python
"""Numerically stable endpoint membrane agreement for partition-paired training."""
import torch
import torch.nn.functional as F


def endpoint_state_consistency(state_pairs):
    if not state_pairs:
        raise ValueError("state consistency requires nonempty endpoint states")
    losses = []
    for state_a, state_b in state_pairs:
        if state_a.shape != state_b.shape:
            raise ValueError("paired endpoint states have different shapes")
        # AMP states can exceed sqrt(float16.max). FP32 accumulation avoids
        # a silently infinite RMS denominator that suppresses the objective.
        a, b = state_a.float(), state_b.float()
        scale = torch.maximum(a.detach().square().mean().sqrt(),
                              b.detach().square().mean().sqrt()).clamp_min(1e-3)
        losses.append(F.smooth_l1_loss(a / scale, b / scale))
    return torch.stack(losses).mean()

```

## research_20261011/repo/lib/train/actors/task_alignment.py

```python
"""Training-only task-space alignment between equivalent event partitions.
Quality selection uses training annotations only. No inference GT or teacher.
"""
import torch
import torch.nn.functional as F

def aligned_iou(boxes, gt_xywh):
    boxes=boxes.float().mean(1)
    lo=boxes[:,:2]-boxes[:,2:]/2; hi=boxes[:,:2]+boxes[:,2:]/2
    gl=gt_xywh[:,:2].float(); gh=gl+gt_xywh[:,2:].float()
    inter=(torch.minimum(hi,gh)-torch.maximum(lo,gl)).clamp_min(0).prod(-1)
    union=boxes[:,2:].clamp_min(0).prod(-1)+gt_xywh[:,2:].clamp_min(0).prod(-1)-inter
    return (inter/union.clamp_min(1e-7)).clamp(0,1)

def task_alignment(primary, paired, gt_xywh, mode):
    if mode not in ('symmetric','directional','gated'):
        raise ValueError(mode)
    a=primary['pred_boxes'].float(); b=paired['pred_boxes'].float()
    with torch.no_grad():
        qa=aligned_iou(a,gt_xywh); qb=aligned_iou(b,gt_xywh)
        use_a=qa>=qb
        gate=((torch.maximum(qa,qb)>=0.3)&((qa-qb).abs()>=0.05)).float() if mode=='gated' else torch.ones_like(qa)
    scale=gt_xywh[:,2:].float().clamp_min(0.05)
    scale=torch.cat((scale,scale),-1)[:,None,:]
    def distribution(score):
        p=score.float().flatten(1).clamp_min(1e-6)
        return p/p.sum(-1,keepdim=True).clamp_min(1e-6)
    pa=distribution(primary['score_map']); pb=distribution(paired['score_map'])
    if mode=='symmetric':
        box=F.smooth_l1_loss(a/scale,b/scale,reduction='none').mean((1,2))
        box=0.5*(F.smooth_l1_loss(a/scale,b.detach()/scale,reduction='none').mean((1,2)) + F.smooth_l1_loss(b/scale,a.detach()/scale,reduction='none').mean((1,2)))
        response=0.5*((pa.detach()*(pa.detach().log()-pb.log())).sum(-1)+(pb.detach()*(pb.detach().log()-pa.log())).sum(-1))
    else:
        choose=use_a[:,None,None]
        teacher=torch.where(choose,a,b).detach(); student=torch.where(choose,b,a)
        box=F.smooth_l1_loss(student/scale,teacher/scale,reduction='none').mean((1,2))
        teacher_p=torch.where(use_a[:,None],pa,pb).detach()
        student_p=torch.where(use_a[:,None],pb,pa)
        response=(teacher_p*(teacher_p.log()-student_p.log())).sum(-1).clamp_min(0)
    result=((box+response)*gate).mean()
    if not torch.isfinite(result): raise FloatingPointError('nonfinite task alignment')
    return result, {'Alignment/gate_fraction':float(gate.mean()),'Alignment/quality_primary':float(qa.mean()),
      'Alignment/quality_pair':float(qb.mean()),'Alignment/box':float((box*gate).mean()),
      'Alignment/response':float((response*gate).mean())}

```

## research_20261011/repo/lib/train/trainers/ltr_trainer.py

```python
import os
import datetime
from collections import OrderedDict

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError('nonfinite unscaled gradient norm')
                        stats['Optim/grad_norm'] = float(grad_norm)
                        stats['Optim/clipped'] = float(grad_norm > self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError('nonfinite unscaled gradient norm')
                        stats['Optim/grad_norm'] = float(grad_norm)
                        stats['Optim/clipped'] = float(grad_norm > self.settings.grad_clip_norm)
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    stats['Optim/amp_scale'] = self.scaler.get_scale()
                    stats['Optim/overflow_skip'] = float(self.scaler.get_scale() < old_scale)

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

```

## research_20261011/run_queue.py

```python
"""Serialized exploratory research queue; never opens official test data."""
import os,sys,json,time,subprocess,signal,hashlib,shutil,fcntl,traceback,concurrent.futures
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent; REPO=RUN/'repo'; ROOT=Path('/root/autodl-tmp/pit_snntrack')
PY=str(ROOT/'venv/bin/python'); C=json.loads((RUN/'contract.json').read_text())
DEADLINE=time.monotonic()+C['wall_budget_days']*86400
sys.path.insert(0,str(REPO))
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def status(phase,**kwargs):
    row={'phase':phase,'time_unix':time.time(),**kwargs};write(RUN/'status.json',row);print(json.dumps(row),flush=True)
def verify():
    assert hashlib.sha256(Path(C['initialization']).read_bytes()).hexdigest()==C['initialization_sha256']
    for rel,expected in C['code_sha256'].items():
        assert hashlib.sha256((REPO/rel).read_bytes()).hexdigest()==expected, 'source changed '+rel
    assert hashlib.sha256((REPO/'experiments/SDTrack/SDTrack-research-task_gated-20261011-smoke.yaml').read_bytes()).hexdigest()==C['smoke_configuration_sha256']
    for name,a in C['arms'].items():
        assert hashlib.sha256((REPO/'experiments/SDTrack'/f"{a['config']}.yaml").read_bytes()).hexdigest()==C['configuration_sha256'][name]
def dependencies():
    for path,success,failures in [
      (ROOT/'controlled_pc_20261010/status.json','completed_validation_matrix',('failed_or_budget_expired',)),
      (ROOT/'followup_pc_20261010/stress_status.json','completed',('failed',))]:
        status('waiting_for_dependency',dependency=str(path),required=success)
        while time.monotonic()<DEADLINE:
            try:s=json.loads(path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(30);continue
            if s['phase']==success:break
            if s['phase'] in failures:raise RuntimeError('upstream failed: '+str(path))
            time.sleep(30)
        else:raise TimeoutError('dependency deadline expired')
def idle():
    stable=0
    while time.monotonic()<DEADLINE:
        util,free=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()[0].split(','))
        disk=shutil.disk_usage(RUN).free
        stable=stable+1 if util<=20 and free>=16000 and disk>=C['minimum_disk_free_bytes'] else 0
        if stable>=3:return
        if not stable:status('waiting_for_capacity',gpu_util=util,gpu_free_mib=free,disk_free=disk)
        time.sleep(10)
    raise TimeoutError('capacity deadline expired')
def environment(extra=None):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
      'PYTHONDONTWRITEBYTECODE':'1','ESOT500_VIEW_ROOT':str(ROOT/'esot500_training_view'),
      'SDTRACK_FINETUNE_CHECKPOINT':C['initialization'],'SDTRACK_KEEP_LATEST_CHECKPOINT':'1',
      'PIT_TASK_ALIGNMENT':'off','PIT_STATE_WARMUP':'0'}
    env.update(extra or {});return env
def execute(command,log,env,timeout):
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        while process.poll() is None:
            if time.monotonic()-start>timeout or time.monotonic()>DEADLINE:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('own process group timed out')
            with (RUN/'telemetry.jsonl').open('a') as f:
                f.write(json.dumps({'time':time.time(),'pid':process.pid,'disk_free':shutil.disk_usage(RUN).free})+'\n')
            time.sleep(10)
        if process.returncode:raise RuntimeError('command failed: '+str(log))
    return time.monotonic()-start

def evaluate(tag,weights):
    folder=RUN/'sequence_results'/tag;folder.mkdir(parents=True,exist_ok=True)
    def task(name):
        path=folder/(name+'.json')
        if path.exists():return json.loads(path.read_text())[0]
        cmd=[PY,'-m','partition.evaluate_esot500','--data-root',str(ROOT/'datasets/ESOT500-L'),
          '--processed-root',str(ROOT/'esot500_processed'),'--sequences',name,'--checkpoint',str(weights),
          '--cpu-threads','1','--opencv-threads','1','--output',str(path)]
        with (folder/(name+'.log')).open('w') as stream:
            subprocess.run(cmd,cwd=REPO,env=environment(),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=3600)
        return json.loads(path.read_text())[0]
    idle();status('evaluating',tag=tag)
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:rows=list(pool.map(task,C['validation_sequences']))
    assert [r['sequence'] for r in rows]==C['validation_sequences']
    write(RUN/(tag+'_validation.json'),rows)
    return time.monotonic()-started

def train(name,seed,smoke=False):
    import torch
    arm=C['arms'][name];tag=('smoke_' if smoke else '')+name+'_seed'+str(seed)
    output=RUN/(tag+'_validation.json');save=RUN/tag;weights=save/'final_weights.pth.tar'
    if not smoke and output.exists():return
    idle();verify()
    if not weights.exists():
        if save.exists():raise RuntimeError('incomplete existing arm; inspect before resume: '+tag)
        config=arm['config']+('-smoke' if smoke else '')
        status('smoke_training' if smoke else 'training',tag=tag,config=config)
        elapsed=execute([PY,'lib/train/run_training.py','--script','SDTrack','--config',config,'--save_dir',str(save),'--seed',str(seed),'--use_wandb','0'],
          RUN/(tag+'_train.log'),environment(arm['env']),C['per_training_timeout_seconds'])
        import re
        lines=(RUN/(tag+'_train.log')).read_text(errors='replace').splitlines()
        epoch_rows=[]
        for line in lines:
            match=re.search(r'\[train: (\d+), (\d+) / (\d+)\]',line)
            if match and match[2]==match[3]:
                metrics={k:float(v) for k,v in re.findall(r'((?:Loss|Optim|Alignment)/[A-Za-z0-9_]+): ([0-9.eE+-]+)',line)}
                epoch_rows.append({'epoch':int(match[1]),'metrics_epoch_average':metrics})
        write(RUN/(tag+'_optimization.json'),{'epoch_summaries':epoch_rows,'interpretation':'clipped is fraction of batches clipped within epoch; gradient norm is total unscaled norm, not task-vs-state gradient attribution'})
        epoch=1 if smoke else C['epochs']
        checkpoint=save/'checkpoints/train/SDTrack'/config/f'SDTrack_ep{epoch:04d}.pth.tar'
        state=torch.load(checkpoint,map_location='cpu');assert state['epoch']==epoch
        if not smoke:torch.save({'net':state['net'],'epoch':epoch,'arm':name,'seed':seed},weights)
        del state
        checkpoint.resolve().relative_to(RUN.resolve())
        checkpoint.unlink()
        write(RUN/(tag+'_runtime.json'),{'training_seconds':elapsed,'smoke':smoke})
    if smoke:return
    seconds=evaluate(tag,weights)
    runtime=json.loads((RUN/(tag+'_runtime.json')).read_text());runtime['validation_seconds']=seconds
    write(RUN/(tag+'_runtime.json'),runtime)

def metric(row,key):
    return row['paired'][key]['mean_prediction_iou'] if key in ('jitter','fixed_count') else row['scores'][(key[:-4] if key.endswith('_auc') else key)]['success_auc_21']
def load(name,seed):
    return {r['sequence']:r for r in json.loads((RUN/(name+'_seed'+str(seed)+'_validation.json')).read_text())}
def difference(name,seed):
    a=load(name,seed);b=load('reference',seed)
    for seq in C['validation_sequences']:
        for key in ('frame_numbers','init_box','ground_truth_xywh','partition_metadata_sha256'):
            assert a[seq][key]==b[seq][key],(name,seed,seq,key)
    return {k:np.array([metric(a[s],k)-metric(b[s],k) for s in C['validation_sequences']])
      for k in ('uniform_auc','jitter_auc','fixed_count_auc','jitter','fixed_count')}
def selection():
    report={};winners=[]
    for name,arm in C['arms'].items():
        if name=='reference':continue
        delta={k:float(v.mean()) for k,v in difference(name,42).items()}
        delta['tracking_score']=float(np.mean([delta[k] for k in ('uniform_auc','jitter_auc','fixed_count_auc')]))
        delta['eligible']=min(delta[k] for k in ('uniform_auc','jitter_auc','fixed_count_auc'))>=-0.01 and delta['jitter']>0 and delta['tracking_score']>0
        delta['group']=arm['group'];report[name]=delta
    for group in ('hyperparameter','method'):
        eligible=[n for n,r in report.items() if r['group']==group and r['eligible']]
        if eligible:winners.append(sorted(eligible,key=lambda n:(-report[n]['tracking_score'],n))[0])
    write(RUN/'selection.json',{'scope':'exploratory validation selection, not independent confirmation','winners':winners,'screen':report})
    return winners

def analyze(winners):
    contrasts={};seeds=[42,123,456];names=C['validation_sequences']
    indices=np.random.default_rng(20261011).integers(0,len(names),(10000,len(names)))
    for name in winners:
        for key in ('uniform_auc','jitter_auc','fixed_count_auc','jitter','fixed_count'):
            arr=np.stack([difference(name,s)[key] for s in seeds]);per_sequence=arr.mean(0)
            contrasts[name+'_minus_reference_'+key]={'mean':float(per_sequence.mean()),
              'sequence_bootstrap_ci95':np.quantile(per_sequence[indices].mean(1),[.025,.975]).tolist(),
              'per_seed_macro':dict(zip(map(str,seeds),arr.mean(1).tolist())),
              'per_sequence_seed_mean':dict(zip(names,per_sequence.tolist()))}
    write(RUN/'analysis.json',{'scope':'exploratory, validation was used for selection; intervals nominal and not selection-adjusted',
      'training_seeds':seeds,'independent_sequences':len(names),'contrasts':contrasts,
      'official_test_access':False,'next_gate':'freeze method and hyperparameters before new held-out benchmark/test; disclose historical official test access'})

def main():
    lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();dependencies()
    if not (RUN/'gpu_smoke_passed.json').exists():
        train('task_gated',42,smoke=True)
        write(RUN/'gpu_smoke_passed.json',{'passed':True,'time_unix':time.time()})
    for name in C['arms']:train(name,42)
    winners=selection()
    if winners:
        for seed in C['confirmation_seeds']:
            for name in ['reference']+winners:train(name,seed)
    if len(winners)==2:
        import yaml
        hp,method=winners
        cfg=yaml.safe_load((REPO/'experiments/SDTrack'/f"{C['arms'][method]['config']}.yaml").read_text())
        assert not (set(C['arms'][hp]['updates']) & set(C['arms'][method]['updates']))
        cfg['TRAIN'].update(C['arms'][hp]['updates'])
        name='combined';config='SDTrack-research-combined-20261011'
        config_path=REPO/'experiments/SDTrack'/f'{config}.yaml'
        if config_path.exists():
            assert yaml.safe_load(config_path.read_text())==cfg
        else:config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        C['arms'][name]={'config':config,'env':C['arms'][method]['env'],'updates':{**C['arms'][method]['updates'],**C['arms'][hp]['updates']},'group':'interaction'}
        C['configuration_sha256'][name]=hashlib.sha256(config_path.read_bytes()).hexdigest()
        write(RUN/'combined_design.json',{'hyperparameter':hp,'method':method,'arm':C['arms'][name],'config_sha256':C['configuration_sha256'][name]})
        for seed in [42]+C['confirmation_seeds']:train(name,seed)
        indices=np.random.default_rng(20261011).integers(0,len(C['validation_sequences']),(10000,len(C['validation_sequences'])))
        interactions={}
        for key in ('uniform_auc','jitter_auc','fixed_count_auc','jitter','fixed_count'):
            arr=np.stack([difference(name,seed)[key]-difference(hp,seed)[key]-difference(method,seed)[key] for seed in [42,123,456]])
            per_sequence=arr.mean(0)
            interactions[key]={'mean':float(per_sequence.mean()),'sequence_bootstrap_ci95':np.quantile(per_sequence[indices].mean(1),[.025,.975]).tolist()}
        write(RUN/'interaction_analysis.json',{'scope':'validation-selected 2x2 factorial; exploratory unadjusted intervals','interaction':interactions})
        winners.append(name)
    analyze(winners);status('completed',winners=winners,official_test_access=False)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));traceback.print_exc();raise

```

## research_20261011/partition_summary_prototype.py

```python
"""Standalone research prototype; not connected to any active training run.
Linear exponential event summaries at a common physical endpoint.
Raw sufficient statistics are additive; nonlinear normalization must follow merging.
"""
import numpy as np

def endpoint_summary(events,height,width,end_us,tau_us):
    if tau_us<=0:raise ValueError('tau_us must be positive')
    valid=(events['x']>0)&(events['x']<width)&(events['y']>0)&(events['y']<height)
    e=events[valid]
    if np.any(e['timestamp']>=end_us):raise ValueError('interval is half open at end_us')
    index=e['y'].astype(np.int64)*width+e['x'].astype(np.int64)
    channel=e['polarity'].astype(np.int64)
    age=(float(end_us)-e['timestamp'].astype(np.float64))/float(tau_us)
    weights=np.exp(-age)
    count=np.zeros((2,height,width),dtype=np.float64)
    moment=count.copy()
    for polarity in (0,1):
        pick=channel==polarity
        count[polarity]=np.bincount(index[pick],minlength=height*width).reshape(height,width)
        moment[polarity]=np.bincount(index[pick],weights=weights[pick],minlength=height*width).reshape(height,width)
    return count,moment

def check_partition_composition():
    rng=np.random.default_rng(20261011)
    e=np.zeros(1000,dtype=[('timestamp','i8'),('x','i2'),('y','i2'),('polarity','?')])
    e['timestamp']=np.sort(rng.integers(0,100,1000));e['x']=rng.integers(0,8,1000);e['y']=rng.integers(0,8,1000);e['polarity']=rng.integers(0,2,1000)
    cases=0;max_error=0.
    for tau in (10.,30.,100.):
        reference=endpoint_summary(e,8,8,100,tau)
        for edges in ([0,25,50,75,100],[0,15,48,82,100],[0,0,50,50,100],[0,99,99,99,100]):
            chunks=[endpoint_summary(e[(e['timestamp']>=lo)&(e['timestamp']<hi)],8,8,100,tau) for lo,hi in zip(edges[:-1],edges[1:])]
            merged=[sum(chunk[i] for chunk in chunks) for i in (0,1)]
            assert np.array_equal(merged[0],reference[0])
            error=float(np.max(np.abs(merged[1]-reference[1])));assert error<1e-12
            max_error=max(max_error,error);cases+=1
    return {'cases':cases,'raw_count_exact':True,'max_exponential_summary_error':max_error,
      'scope':'same events, fixed physical endpoint; includes duplicate timestamps and empty bins. This validates the unquantized accumulator, not model accuracy or SNN-state invariance.'}
if __name__=='__main__':
    import json
    from pathlib import Path
    result=check_partition_composition();Path(__file__).with_name('physical_summary_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

```

## research_20261011/temporal_locked_drop_path.py

```python
"""Standalone candidate, not wired into the sealed training queue.
Drop residual paths per sample, sharing masks over all temporal bins.
Pair-view sharing relies on restoring the CPU/CUDA RNG before the paired forward.
"""
import torch
from torch import nn
class TemporalLockedDropPath(nn.Module):
    def __init__(self,p=0.):
        super().__init__()
        if not 0<=p<1:raise ValueError('p must be in [0,1)')
        self.p=float(p)
    def forward(self,x):
        if x.ndim<3:raise ValueError('expected [T,B,...]')
        if not self.training or self.p==0:return x
        shape=(1,x.shape[1])+(1,)*(x.ndim-2)
        keep=(torch.rand(shape,device=x.device)>=self.p).to(x.dtype)
        return x*keep/(1-self.p)
def check():
    torch.manual_seed(20261011);x=torch.ones(4,10000,2,2,2)
    layer=TemporalLockedDropPath(.1).train();rng=torch.get_rng_state();a=layer(x)
    torch.set_rng_state(rng);b=layer(x)
    assert torch.equal(a,b) and torch.equal(a[0],a[-1])
    assert abs(float(a.mean())-1)<.02
    assert torch.equal(TemporalLockedDropPath(0).train()(x),x)
    assert torch.equal(layer.eval()(x),x)
    return {'time_locked_mask':True,'pair_locked_under_rng_replay':True,'zero_rate_identity':True,'eval_identity':True,
      'empirical_mean_at_p01':float(a.mean()),'scope':'CPU mask correctness only; no performance measurement, no active model change'}
if __name__=='__main__':
    import json
    from pathlib import Path
    result=check();Path(__file__).with_name('temporal_drop_path_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

```

## controlled_pc_20261010/run.py

```python
"""Bounded idle-GPU feasibility pilot. Never accesses the official test split."""
import json, os, subprocess, time, shutil, signal, traceback, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "pit-sdtrack"
sys.path.insert(0, str(REPO))
RUN = Path(__file__).resolve().parent
PY = str(ROOT / "venv/bin/python")
CONTRACT = json.loads((RUN / "contract.json").read_text())
DEADLINE = time.monotonic() + CONTRACT["wait_and_execution_wall_limit_seconds"]

def status(phase, **extra):
    row = {"phase": phase, "timestamp_unix": time.time(), **extra}
    (RUN / "status.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)

def capacity():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=10)
    util, free = map(float, result.splitlines()[0].split(","))
    return {"gpu_util": util, "gpu_free_mib": free, "disk_free": shutil.disk_usage(ROOT).free}

def wait_idle():
    stable = 0
    while time.monotonic() < DEADLINE:
        row = capacity()
        safe = row["gpu_util"] <= 20 and row["gpu_free_mib"] >= 16000 and row["disk_free"] >= CONTRACT["minimum_disk_free_bytes"]
        stable = stable + 1 if safe else 0
        status("waiting_for_idle_gpu", stable=stable, **row)
        if stable >= 3:
            return
        time.sleep(min(30, max(0, DEADLINE - time.monotonic())))
    raise TimeoutError("pilot wall budget expired before idle GPU became available")

def execute(command, log, timeout):
    duration = min(timeout, DEADLINE-time.monotonic())
    if duration <= 0:
        raise TimeoutError("pilot wall budget exhausted")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               ESOT500_VIEW_ROOT=str(ROOT / "esot500_training_view"),
               SDTRACK_FINETUNE_CHECKPOINT=CONTRACT["initialization"], SDTRACK_KEEP_LATEST_CHECKPOINT="1")
    status("running", command=command, log=str(log))
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        started=time.monotonic()
        while process.poll() is None:
            if time.monotonic()-started > duration:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError("hard command timeout; stopped only pilot process group")
            with (RUN/"telemetry.jsonl").open("a") as f:
                f.write(json.dumps({"time":time.time(),"pid":process.pid,**capacity()})+"\n")
            time.sleep(15)
        if process.returncode:
            raise RuntimeError("pilot command failed, see "+str(log))

def main():
    import torch, hashlib
    import fcntl
    lock=(RUN/"queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert hashlib.sha256(Path(CONTRACT["initialization"]).read_bytes()).hexdigest()==CONTRACT["initialization_sha256"], "initialization changed"
    for relative,expected in CONTRACT["code_sha256"].items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest()==expected, "source changed: "+relative
    for seed in CONTRACT["seeds"]:
        for arm, config in CONTRACT["arms"].items():
            tag=arm+"_seed"+str(seed)
            wait_idle()
            save=RUN/tag
            if save.exists():
                raise RuntimeError("refuse overwrite/retrain existing arm: "+tag)
            execute([PY,"lib/train/run_training.py","--script","SDTrack","--config",config,"--save_dir",str(save),"--seed",str(seed),"--use_wandb","0"],
                    RUN/(tag+"_train.log"), CONTRACT["per_training_timeout_seconds"])
            checkpoint=save/"checkpoints/train/SDTrack"/config/"SDTrack_ep0010.pth.tar"
            if not checkpoint.exists():
                raise RuntimeError("missing final successful training checkpoint")
            state=torch.load(checkpoint,map_location="cpu")
            assert state["epoch"]==10
            weights=save/"final_weights.pth.tar"
            torch.save({"net":state["net"],"epoch":10,"seed":seed,"arm":arm,"source_checkpoint":str(checkpoint)},weights)
            del state
            # This path was created by this arm, under this isolated run.
            assert checkpoint.resolve().is_relative_to(RUN.resolve()) if hasattr(Path, "is_relative_to") else str(checkpoint.resolve()).startswith(str(RUN.resolve())+"/")
            checkpoint.unlink()
            wait_idle()
            execute([PY,str(RUN/"evaluate.py"),"--checkpoint",str(weights),"--tag",tag],
                    RUN/(tag+"_eval.log"), CONTRACT["per_evaluation_timeout_seconds"])
    execute([PY,str(RUN/"analyze.py")],RUN/"analysis.log",120)
    status("completed_validation_matrix", official_test_access=False)

if __name__=="__main__":
    try: main()
    except Exception as exc:
        status("failed_or_budget_expired",error=str(exc))
        traceback.print_exc()
        raise

```

## controlled_pc_20261010/evaluate.py

```python
"""Reuse the historical exact-parity evaluator scheduling policy."""
import argparse,sys,json
from pathlib import Path
run=Path(__file__).resolve().parent
root=run.parent
sys.path.insert(0,str(root))
import run_validation_scheduled as scheduler
p=argparse.ArgumentParser()
p.add_argument("--checkpoint",type=Path,required=True)
p.add_argument("--tag",required=True)
args=p.parse_args()
contract=json.loads((run/"contract.json").read_text())
assert scheduler.NAMES==contract["validation_sequences"]
output=run/(args.tag+"_validation.json")
assert not output.exists(), "refuse overwrite validation"
settings=json.loads((root/"runtime_eval_policy.json").read_text())
assert settings["accepted_exact_parity"] is True
rows,metrics=scheduler.run_set(run.name+"_"+args.tag,"native",args.checkpoint,
 settings["workers"],settings["torch_threads_per_worker"],settings["opencv_threads_per_worker"])
assert set(r["sequence"] for r in rows)==set(contract["validation_sequences"])
output.write_text(json.dumps(rows,indent=2))
(run/(args.tag+"_runtime.json")).write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics))

```

## controlled_pc_20261010/analyze.py

```python
"""Compare matched arms without treating seeds or frames as independent sequences."""
import json
from pathlib import Path
import numpy as np
run=Path(__file__).resolve().parent
contract=json.loads((run/"contract.json").read_text())
names=contract["validation_sequences"]
seeds=contract["seeds"]
arms=list(contract["arms"])
data={}
for seed in seeds:
    for arm in arms:
        rows=json.loads((run/(arm+"_seed"+str(seed)+"_validation.json")).read_text())
        assert len(rows)==len(names)
        data[seed,arm]={r["sequence"]:r for r in rows}
        assert set(data[seed,arm])==set(names)
for seed in seeds:
    for name in names:
        reference=data[seed,"paired"][name]
        for arm in arms:
            row=data[seed,arm][name]
            for key in ("frame_numbers","init_box","ground_truth_xywh","partition_metadata_sha256"):
                assert row[key]==reference[key],(seed,arm,name,key)
rng=np.random.default_rng(contract["bootstrap_seed"])
indices=rng.integers(0,len(names),size=(contract["bootstrap_replicates"],len(names)))
def measure(row,metric):
    if metric=="uniform_auc":return row["scores"]["uniform"]["success_auc_21"]
    return row["paired"][metric]["mean_prediction_iou"]
contrasts={}
for control in ("paired","box","state","baseline"):
    for metric in ("jitter","uniform_auc","fixed_count"):
        delta=np.array([[measure(data[seed,"full"][name],metric)-measure(data[seed,control][name],metric)
                         for name in names] for seed in seeds])
        per_sequence=delta.mean(axis=0)
        contrasts["full_minus_"+control+"_"+metric]={
            "mean":float(per_sequence.mean()),
            "sequence_bootstrap_ci95":np.quantile(per_sequence[indices].mean(axis=1),[.025,.975]).tolist(),
            "per_seed_macro_delta":dict(zip(map(str,seeds),delta.mean(axis=1).tolist())),
            "per_sequence_seed_mean_delta":dict(zip(names,per_sequence.tolist()))}
result={"stage":"validation_only","primary":"full_minus_paired_jitter","seed_count":len(seeds),"sequence_count":len(names),
        "contrasts":contrasts,"secondary_multiplicity":"exploratory, unadjusted",
        "inference_scope":"Bootstrap independent sequences after averaging matched training seeds; seed variability reported separately. No official test or acceptance claim."}
(run/"analysis.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

```

## followup_pc_20261010/run_stress_queue.py

```python
"""Queued unseen-partition validation; starts only after the primary matrix completes."""
import json,subprocess,time,os,concurrent.futures,traceback,fcntl
from pathlib import Path
root=Path(__file__).resolve().parents[1];run=Path(__file__).resolve().parent;matrix=root/"controlled_pc_20261010"
python=str(root/"venv/bin/python");repo=root/"pit-sdtrack"
contract=json.loads((run/"contract.json").read_text());original=json.loads((matrix/"contract.json").read_text())
names=original["validation_sequences"]
def status(phase,**kwargs):
    tmp=run/"stress_status.tmp";tmp.write_text(json.dumps({"phase":phase,"timestamp_unix":time.time(),**kwargs},indent=2));tmp.replace(run/"stress_status.json")
def idle():
    stable=0;deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        util,free=map(float,subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(","))
        stable=stable+1 if util<=20 and free>=16000 else 0
        if stable>=3:return
        time.sleep(5)
    raise TimeoutError("idle GPU unavailable after primary matrix completion")
def main():
    lock=(run/"stress.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status("waiting_for_primary_matrix")
    deadline=time.monotonic()+16*86400
    while time.monotonic()<deadline:
        state=json.loads((matrix/"status.json").read_text())
        if state["phase"]=="completed_validation_matrix":break
        if state["phase"]=="failed_or_budget_expired":raise RuntimeError("primary matrix failed; do not bypass it")
        time.sleep(60)
    else:raise TimeoutError("primary matrix wait budget expired")
    audit=json.loads((run/"stress_metadata_audit.json").read_text());assert audit["status"]=="passed"
    for seed in contract["training_seeds"]:
      for strength in contract["severities"]:
       for partition_seed in contract["partition_seeds"]:
        for arm in contract["arms"]:
            tag=f"{arm}_seed{seed}_strength{strength}_partitionseed{partition_seed}"
            output=run/(tag+"_stress.json")
            if output.exists():continue
            idle()
            checkpoint=matrix/f"{arm}_seed{seed}"/"final_weights.pth.tar"
            assert checkpoint.exists()
            processed=root/"esot500_jitter_sweep"/f"strength_{strength}_seed_{partition_seed}"/"processed"
            status("evaluating",tag=tag,completed_sets=len(list(run.glob("*_stress.json"))),planned_sets=54)
            folder=run/"sequence_results"/tag;folder.mkdir(parents=True,exist_ok=True)
            def task(name):
                path=folder/(name+".json");log=folder/(name+".log")
                if path.exists():return json.loads(path.read_text())[0]
                cmd=[python,"-m","partition.evaluate_esot500","--data-root",str(root/"datasets/ESOT500-L"),
                 "--processed-root",str(processed),"--sequences",name,"--checkpoint",str(checkpoint),
                 "--policies","uniform","jitter","--cpu-threads","1","--opencv-threads","1","--output",str(path)]
                with log.open("w") as stream:
                    subprocess.run(cmd,cwd=repo,env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1"},
                     stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=3600)
                return json.loads(path.read_text())[0]
            started=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:rows=list(pool.map(task,names))
            assert [r["sequence"] for r in rows]==names
            output.write_text(json.dumps({"arm":arm,"training_seed":seed,"partition_seed":partition_seed,"strength":strength,
             "elapsed_seconds":time.monotonic()-started,"rows":rows},indent=2))
    subprocess.run([python,str(run/"analyze_stress.py")],check=True)
    status("completed",completed_sets=54)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        status("failed",error=str(exc));traceback.print_exc();raise

```

## followup_pc_20261010/analyze_stress.py

```python
"""Average partition realizations and training seeds within sequence, then bootstrap sequences."""
import json
from pathlib import Path
import numpy as np
run=Path(__file__).resolve().parent;root=run.parent
contract=json.loads((run/"contract.json").read_text());names=json.loads((root/"controlled_pc_20261010/contract.json").read_text())["validation_sequences"]
data={}
for path in run.glob("*_stress.json"):
    x=json.loads(path.read_text());key=(x["arm"],x["training_seed"],x["strength"],x["partition_seed"])
    assert key not in data
    data[key]={r["sequence"]:r for r in x["rows"]}
assert len(data)==54
contrasts={}
rng=np.random.default_rng(20261010);ix=rng.integers(0,len(names),size=(10000,len(names)))
for control in ("box","paired"):
 for strength in contract["severities"]:
    values=[]
    for name in names:
        effects=[]
        for seed in contract["training_seeds"]:
         for partition_seed in contract["partition_seeds"]:
            a=data["full",seed,strength,partition_seed][name];b=data[control,seed,strength,partition_seed][name]
            for key in ("frame_numbers","ground_truth_xywh","init_box","partition_metadata_sha256"):
                assert a[key]==b[key],(control,strength,seed,partition_seed,name,key)
            effects.append(a["paired"]["jitter"]["mean_prediction_iou"]-b["paired"]["jitter"]["mean_prediction_iou"])
        values.append(float(np.mean(effects)))
    v=np.array(values)
    contrasts[f"full_minus_{control}_strength{strength}"]={"mean":float(v.mean()),
      "sequence_bootstrap_ci95":np.quantile(v[ix].mean(axis=1),[.025,.975]).tolist(),"per_sequence":dict(zip(names,values))}
result={"stage":"historical_validation_unseen_partition_robustness","completed_sets":54,"sequences":len(names),"contrasts":contrasts,
 "primary_followup":"full_minus_box_strength50","inference_scope":"Exploratory; 9 independent sequences, not 81 independent seed/partition replicates. No official test access."}
(run/"stress_analysis.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

```

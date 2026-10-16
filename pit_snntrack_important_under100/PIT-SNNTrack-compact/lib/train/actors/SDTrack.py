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
                    + float(getattr(self.cfg.TRAIN, 'PARTITION_STATE_WEIGHT', 0.1)) * state_consistency)
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
            return loss, status
        else:
            return loss

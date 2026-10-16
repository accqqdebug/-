"""ESOT500-L train/validation views using the SDTrack event-stack interface."""
from pathlib import Path

from lib.train.dataset.FE108 import EOTB
from lib.train.data import opencv_loader


class ESOT500(EOTB):
    def __init__(self, root, image_loader=opencv_loader):
        # The view root contains only one pre-registered split, so do not read
        # FE108's unrelated train/validation sequence lists.
        super().__init__(root=root, image_loader=image_loader, split=None)

    def get_name(self):
        return 'esot500'

    def get_paired_frames(self, seq_id, frame_ids, cfg, policy='jitter'):
        """Read a second partition for the same frames, without labels or templates.

        The prepared view layout is <view>/<policy>/<split>/<sequence>. The
        default training root is <view>/uniform/train, so the paired policy is
        resolved to <view>/<policy>/train. This method is called only by the
        sampler when the opt-in training flag is enabled.
        """
        if cfg.MODEL.T != 4:
            raise ValueError('paired ESOT500 partitions require T=4')
        if policy not in ('jitter', 'fixed_count'):
            raise ValueError(f'unsupported paired partition policy: {policy}')

        root = Path(self.root)
        split = root.name
        view_root = root.parent.parent
        pair_root = view_root / policy / split
        sequence = self.sequence_list[seq_id]
        stack_root = pair_root / sequence / 'inter4_stack_3008'
        if not stack_root.is_dir():
            raise FileNotFoundError(stack_root)

        frames = []
        for frame_id in frame_ids:
            images = tuple(self.image_loader(str(stack_root / f'{frame_id + 1:04d}_{i + 1}.png'))
                           for i in range(4))
            if any(image is None for image in images):
                raise FileNotFoundError(f'missing paired event image in {stack_root} at frame {frame_id + 1}')
            frames.append(images)
        return frames

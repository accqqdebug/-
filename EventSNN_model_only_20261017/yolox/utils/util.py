import torch

def warp_decay(decay):
    import math
    return torch.tensor(math.log(decay / (1 - decay)))

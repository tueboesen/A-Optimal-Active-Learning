import random
import numpy as np
import torch
import torch.nn as nn

def fix_seed(seed: int, include_cuda: bool = True) -> None:
    """
    Set the seed in order to create reproducible results, note that setting the seed also does it for gpu calculations, which slows them down.
    :param seed: an integer to fix the seed to
    :param include_cuda: whether to fix the seed for cuda calculations as well
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def determine_network_param(net: type[nn.Module]) -> int:
    """
    Determines the number of learnable parameters in a neural network
    :param net: the neural network to determine the number of parameters in.
    :return:
    """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

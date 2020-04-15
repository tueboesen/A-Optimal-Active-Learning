import torch
import torch.nn.functional as F
import torch.nn as nn

# def CrossEntropyLoss_weighted(target=None, input=None, point_weight=1):
#     loss_fnc_soft = nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
#     #however if we do not want the softmax normalization, then we need to use our custom norm, and then do NLLLoss, (maximum log likelihood)
#     loss_fnc = nn.NLLLoss(reduction='none',ignore_index=-1)
#     loss = point_weight * loss_fnc(norm(input), target)
#     return loss

def cross_entropy_probabilities(target=None, input=None, point_weight=1):
    assert input.size() == target.size()
    nx, _ = target.shape
    if abs(torch.sum(torch.abs(target)) / nx - 1) < 1e-6:
        loss = -(point_weight * (target * torch.log(F.softmax(input, dim=1))).sum(dim=1)).mean()
    else:
        loss = -(point_weight * (F.softmax(target, dim=1) * torch.log(F.softmax(input, dim=1))).sum(dim=1)).mean()
    return loss

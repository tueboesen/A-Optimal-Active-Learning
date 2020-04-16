import torch
import torch.nn.functional as F


# def CrossEntropyLoss_weighted(target=None, input=None, point_weight=1):
#     loss_fnc_soft = nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
#     #however if we do not want the softmax normalization, then we need to use our custom norm, and then do NLLLoss, (maximum log likelihood)
#     loss_fnc = nn.NLLLoss(reduction='none',ignore_index=-1)
#     loss = point_weight * loss_fnc(norm(input), target)
#     return loss

class cross_entropy_probabilities(torch.nn.Module):

    def __init__(self,reduction='none'):
        self.reduction = reduction
        super(cross_entropy_probabilities,self).__init__()

    def forward(self, input, target, point_weight=1):
        assert input.size() == target.size()
        nx, _ = target.shape
        if abs(torch.sum(torch.abs(target)) / nx - 1) < 1e-6: #is the target already a probability?
            loss = -(point_weight * (target * torch.log(F.softmax(input, dim=1))).sum(dim=1))
        else:
            loss = -(point_weight * (F.softmax(target, dim=1) * torch.log(F.softmax(input, dim=1))).sum(dim=1))
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


# def cross_entropy_probabilities(target=None, input=None, point_weight=1):
#     assert input.size() == target.size()
#     nx, _ = target.shape
#     if abs(torch.sum(torch.abs(target)) / nx - 1) < 1e-6:
#         loss = -(point_weight * (target * torch.log(F.softmax(input, dim=1))).sum(dim=1)).mean()
#     else:
#         loss = -(point_weight * (F.softmax(target, dim=1) * torch.log(F.softmax(input, dim=1))).sum(dim=1)).mean()
#     return loss

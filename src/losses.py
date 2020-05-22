import torch
import torch.nn as nn
import torch.nn.functional as F

def select_loss_fnc(loss_type,use_probabilities=False):
    '''
    Selects the type of loss function to use. Choices are currently Cross-entropy or Mean-Square-Estimate.
    :param loss_type: 'CE' or 'MSE'
    :param use_probabilities:
    :return:
    '''
    if loss_type == 'CE':
        if use_probabilities:
            loss_fnc = cross_entropy_probabilities(reduction='none')
        else:
            loss_fnc = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
    elif loss_type == 'MSE':
        loss_fnc = MSE_custom(reduction='sum_to_samples')
    else:
        raise ValueError("Undefined loss_fnc selected")

    return loss_fnc


class cross_entropy_probabilities(torch.nn.Module):
    '''
    Cross entropy function that can handle probability targets
    '''

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

class MSE_custom(torch.nn.Module):
    '''
    This class works just like the normal MSE loss function, with the extra option of using reduction='sum_to_samples'
    which sums over all other dimension and reduce the dimension down to the number of samples. Which is useful if you want to put individual weight on each point.
    '''
    def __init__(self,reduction='sum_to_samples'):
        self.reduction = reduction
        super(MSE_custom,self).__init__()

    def forward(self, input, target):
        if self.reduction == 'sum_to_samples':
            return F.mse_loss(input, target, reduction='none').sum(dim=1)
        else:
            return F.mse_loss(input, target, reduction=self.reduction)
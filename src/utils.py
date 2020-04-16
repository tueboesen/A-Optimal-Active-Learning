import random

import numpy as np
import torch
import torch.nn.functional as F


# Define some functions
def normalizeImage(I):
    I = I - torch.min(I)
    I = 2 * I / torch.max(I) - 1
    return I


def make3chan(X):
    Xout = torch.zeros(X.shape[0], 3, X.shape[2], X.shape[3])
    for i in range(3):
        Xout[:, i, :, :] = X.squeeze(1)
    return Xout


def clear_grad(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param._grad = None


def conv3x3(x, K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=1)


def conv3x3T(x, K):
    """3x3 convolution transpose with padding"""
    # K = torch.transpose(K,0,1)
    return F.conv_transpose2d(x, K, stride=1, padding=1)


def tvnorm(T):
    t = torch.sum(T ** 2, 1).unsqueeze(1) / T.shape[1]
    return T / torch.sqrt(t + 1e-3)


def DoubleSymLayer(x, K, A, B):
    z = conv3x3(x, K)  # - conv3x3T(x,K)
    z = A * tvnorm(z) + B
    z = F.relu(z)
    z = conv3x3T(z, K)
    return z


# pairwise distance matrix
def getDistMatrix(T, R):
    n = T.shape[0]
    T = T.view(n, -1)
    R = R.view(n, -1)

    # C  = torch.zeros(n,n)
    # for i in range(n):
    #    C[i,:] = torch.sum((T[i,:]-R)**2,1) #/torch.sum((R)**2,1)

    C = torch.sum(T ** 2, 1).unsqueeze(1) + torch.sum(R ** 2, 1).unsqueeze(0) - 2 * torch.matmul(T, R.t())
    return C


# normalize matrix to become a probability
def normalizeMatrix(C, k=15):
    n = C.shape[0]

    mx = torch.max(C)
    Cp = torch.exp(C - mx)
    d1 = 1 / torch.sum(Cp, 0);

    for i in range(k):
        d1 = 1 / torch.sum(torch.diag(1 / torch.sum(Cp @ torch.diag(d1), 1)) @ Cp, 0)
        d2 = 1 / torch.sum(Cp @ torch.diag(d1), 1)
        d2 = 1 / torch.sum(Cp @ torch.diag(1 / torch.sum(torch.diag(d2) @ Cp, 0)), 1)

        D1 = torch.diag(d1)
        D2 = torch.diag(d2);

        # print(i,torch.norm(torch.sum(D2@C@D1,0)-1).item()/n,torch.norm(torch.sum(D2@C@D1,1)-1).item()/n)

    return D2 @ Cp @ D1


def fix_seed(seed, include_cuda=True):
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


def getDistMatrix(T, R, W=1):
    nT = T.shape[0]
    nR = R.shape[0]

    T = T.view(nT, -1)
    R = R.view(nR, -1)

    C = torch.sum(T ** 2, 1).unsqueeze(1) + torch.sum(R ** 2, 1).unsqueeze(0) - 2 * torch.matmul(T, R.t())
    return W * C


def Benchmark(dataloader,metric,nruns=10):
    distsA = []
    distsB = []
    distsAB_pair = []
    distsAB_cross = []
    for i in range(nruns):
        dataloader_iter = iter(dataloader)
        A1,B1 = next(dataloader_iter)
        if len(dataloader) > 1:
            A2,B2 = next(dataloader_iter)
        else:
            A2,B2 = next(iter(dataloader))
        dA = metric(A1, A2) / len(A1)
        dB = metric(B1, B2) / len(B1)
        dAB_pair = (metric(A1, B1) + metric(A2, B2)) / (2 * len(B1))
        dAB_cross = (metric(A1, B2) + metric(A2, B1)) / (2 * len(B1))
        distsA.append(dA.item())
        distsB.append(dB.item())
        distsAB_pair.append(dAB_pair.item())
        distsAB_cross.append(dAB_cross.item())

    return distsA, distsB, distsAB_pair, distsAB_cross


def Benchmark_TV(dataloader,nruns=10):
    TV_A = []
    TV_B = []
    for i in range(nruns):
        dataloader_iter = iter(dataloader)
        A1,B1 = next(dataloader_iter)
        TV_A.append(TV_loss(A1).item())
        TV_B.append(TV_loss(B1).item())
    return TV_A, TV_B


def TV_loss(img):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).sum()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).sum()
    return tv_h + tv_w

def determine_network_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def create_label_probabilities_from_dataset(dataset,MaxValue=90):
    '''
    This function creates a label probability vector U \in R^{n,nc}.
    :param dataset: dataset to build the probability vector from
    :param MaxValue: The value to assign to a labelled point
    :return: U
    '''
    n = len(dataset)
    if dataset.use_label_probabilities:
        n,nc = dataset.labels.shape
        U = np.zeros((n,nc))
        for i,obs in enumerate(dataset.labels):
            if obs[0].item() != -1:
                U[i, :] = -MaxValue/(nc-1)
                idx = torch.argmax(obs).item()
                U[i, idx] = MaxValue
    else:
        classes = np.unique(dataset.labels_true)
        nc = len(classes)
        U = np.zeros((n,nc))
        for i,obs in enumerate(dataset.labels):
            if obs != -1:
                U[i,:] = -MaxValue/(nc-1)
                U[i,obs] = MaxValue
    return U



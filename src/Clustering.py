import numpy as np
import torch
from scipy.sparse import identity, triu, tril, diags
from scipy.sparse.linalg import spsolve, cg, LinearOperator
from scipy.special import softmax


def SSL_clustering(alpha, L, Yobs, balance_weights=False, w=None):
    '''
    Minimizes the objective function:
    U = arg min_Y 1/2 ||Y - Yobs||_W^2 + alpha/2 * Y'*L*Y
    s.t. Ye = 0
    which has the closed form solution:
    U = (W + alpha*L)^-1 W * Yobs * C
    :param alpha: hyperparameter
    :param L: Graph Laplacian
    :param Yobs: labelled data
    :param balance_weights: If true it will ensure that the weights of each class adds up to 1. (this should be used if the classes have different number of sampled points)
    :return:
    '''
    TOL = 1e-12;
    MAXITER = 2000;
    if isinstance(Yobs, torch.Tensor):
        Yobs = Yobs.numpy()
    n,nc = Yobs.shape
    idxs = np.nonzero(Yobs[:, 0])[0]
    nidxs = len(idxs)
    tmp = np.zeros(n)
    if w is None:
        if balance_weights:
            labels = np.argmax(Yobs[idxs, :], axis=1)
            nlabelsprclass = np.bincount(labels,minlength=10)
            for idx,w in enumerate(nlabelsprclass):
                if w != 0:
                    idxs_i = idxs[labels == idx]
                    tmp[idxs_i] = 1/w
        else:
            tmp[idxs] = 1
    else:
        tmp = w
    W = diags(tmp)
    I_nc = identity(nc)
    e = np.ones((nc,1))
    C = I_nc - (e @ e.T) / (e.T @ e) #Make sure this is vector products except the division which is elementwise
    A = (1 / n * alpha * L + 1 / nidxs * W)
    b = 1 / nidxs * W @ Yobs @ C #Also matrix products
    def precond(x):
        return spsolve(tril(A, format='csc'), (A.diagonal() * spsolve(triu(A, format='csc'), x, permc_spec='NATURAL')), permc_spec='NATURAL')
    M = LinearOperator(matvec=precond, shape=(n, n), dtype=float)
    U = np.empty_like(Yobs)
    for i in range(nc):
        U[:,i], _ = cg(A, b[:,i], M=M,tol=TOL,maxiter=MAXITER)
    return U


def SSL_clustering_AL(alpha, L, Yobs, w):
    '''
    Minimizes the objective function:
    U = arg min_Y 1/2 ||Y - Yobs||_W^2 + alpha/2 * Y'*L*Y
    s.t. Ye = 0
    which has the closed form solution:
    U = (W + alpha*L)^-1 W * Yobs * C
    :param alpha: hyperparameter
    :param L: Graph Laplacian
    :param Yobs: labelled data
    :param balance_weights: If true it will ensure that the weights of each class adds up to 1. (this should be used if the classes have different number of sampled points)
    :return:
    '''
    TOL = 1e-12;
    MAXITER = 2000;
    if isinstance(Yobs, torch.Tensor):
        Yobs = Yobs.numpy()
    n,nc = Yobs.shape
    W = diags(w)
    A = (alpha * L + W)
    b = W @ Yobs
    def precond(x):
        return spsolve(tril(A, format='csc'), (A.diagonal() * spsolve(triu(A, format='csc'), x, permc_spec='NATURAL')), permc_spec='NATURAL')
    M = LinearOperator(matvec=precond, shape=(n, n), dtype=float)
    U = np.empty_like(Yobs)
    for i in range(nc):
        U[:,i], _ = cg(A, b[:,i], M=M,tol=TOL,maxiter=MAXITER)
    return U


def convert_pseudo_to_prob(v,use_softmax=False):
    '''
    Converts a pseudo probability array to a probability array
    :param v: is a pseudo probability array, that is subject to ve = 0
    :param use_softmax:
    :return:
    '''
    if use_softmax:
        cpv = softmax(v,axis=1)
    else:
        maxval = np.sum(np.abs(v), axis=1)
        vshifted = v - np.min(v,axis=1)[:,None]
        cpv = vshifted / (maxval[:,None]+1e-10)
    return cpv

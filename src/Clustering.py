import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import cg, LinearOperator
from scipy.special import softmax
def SSL_clustering(alpha, L, Yobs, w, eta, TOL=1e-9,MAXITER=10000):
    """
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
    """
    if isinstance(Yobs, torch.Tensor):
        Yobs = Yobs.numpy()
    n,nc = Yobs.shape
    W = diags(w)
    if eta == 1:
        H_lambda = lambda x: alpha*((L @ x)) + (W @ x)
    elif eta == 2:
        H_lambda = lambda x: alpha*(L.T @ (L @ x)) + (W @ x)
    elif eta == 3:
        H_lambda = lambda x: alpha*(L @ (L.T @ (L @ x))) + (W @ x)
    elif eta == 4:
        H_lambda = lambda x: alpha * (L @ (L @ (L.T @ (L @ x)))) + (W @ x)
    else:
        raise NotImplementedError('Not implemented')
    A = LinearOperator((n, n), H_lambda)
    b = W @ Yobs
    U = np.empty_like(Yobs)
    for i in range(nc):
        U[:,i], stat = cg(A, b[:,i], tol=TOL,maxiter=MAXITER)
    return U

def convert_pseudo_to_prob(v,use_softmax=False):
    """
    Converts a pseudo probability array to a probability array
    :param v: is a pseudo probability array, that is subject to ve = 0
    :param use_softmax:
    :return:
    """
    if use_softmax:
        cpv = softmax(v,axis=1)
    else:
        maxval = np.sum(np.abs(v), axis=1)
        vshifted = v - np.min(v,axis=1)[:,None]
        cpv = vshifted / (maxval[:,None]+1e-10)
    return cpv

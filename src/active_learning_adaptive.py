import time

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse import identity, diags
from scipy.sparse.linalg import cg, LinearOperator

from src.Clustering import SSL_clustering


def run_active_learning_adaptive(idx_labels,L,w,y,c):
    t0 = time.time()
    n, nc = y.shape
    yi = np.zeros((n, 1))
    yobs = np.zeros_like(y)
    yobs[idx_labels,:] = y[idx_labels,:]
    for j in range(nc):
        yi[:, 0] = yobs[:, j]
        yi = np.sign(yi)
        yi = SSL_clustering(c.AL_alpha, L, yi, w, c.L_eta, TOL=1e-12)
        t1 = time.time()
        idx_labels,w = OEDA(idx_labels, w, L, yi, c)
        yobs[idx_labels, :] = y[idx_labels, :] #Ask the oracle for new labels
        t2 = time.time()
        c.LOG.debug("Clustering {:2.2f}, OEDA {:2.2f}".format(t1 - t0, t2 - t1))
    return idx_labels,w

def OEDA(idx_learned,w,L,y,c,use_stochastic_approximate=True):
    '''
    Finds the next ns points to learn, according to:
    min_w \alpha^2 ||H^{-1} L y ||^2 + \sigma^2 \Trace (W H^{-2} W)
    s.t. 0<= w
    :param w: the weight of the points already learned
    :param L: Graph Laplacian
    :param y: Labels (The code can handle y being a probability vector, but it doesn't lead to good results so far)
    :param idx_learned: Indices of points already learned, will be iteratively updated as new points are found.
    :param use_stochastic_approximate: If true it uses a vector of +-1 to stochasticically approximate the inverse matrices.
    :return: w with the new indices included
    '''
    t0 = time.time()
    if c.AL_nlabels_pr_class <= 0:
        return w
    if use_stochastic_approximate:
        v = np.sign(np.random.normal(0,1,(y.shape[0],1)))
    else:
        v = identity(L.shape[0]).tocsc()
    f,df = getOEDA(w,L,y,v,c)
    indices = np.argsort(np.abs(df))[::-1]
    i = 0
    idx_excluded = []
    idx_new = set()
    for idx in indices:
        if idx not in (idx_learned and idx_excluded):
            i += 1
            idx_learned.append(idx)
            idx_new.add(idx)
            if i >= c.AL_nlabels_pr_class:
                break
            else:
                # Remove all neighbouring points from the potential candidates
                aa = L.nonzero()
                tmp = np.where(aa[0] == idx)[0]
                idxs = aa[1][tmp]
                idx_excluded += idxs.tolist()
    if i < c.AL_nlabels_pr_class: #This only happens if we have somehow ruled out every single point...
        #Lets just select the rest at random then
        nremain = c.AL_nlabels_pr_class - i
        full_set = set(range(df.shape[0]))
        set_pos = full_set.difference(idx_learned)
        idxs = list(set_pos)
        np.random.shuffle(idxs)
        for i in range(nremain):
            idx_learned.append(idxs[i])
        c.LOG.info("Adding {} Random points!".format(nremain))
    w[list(idx_learned)] = c.AL_w0
    return idx_learned,w

def getOEDA(w,L,y,v,c):
    '''
    Computes the value and derivatives of:
    f(w) =  \alpha^2 ||H^{-1} L y ||^2 + \sigma^2 \Trace (W H^{-2} W)
    with bias = ||H^{-1} L y ||^2  and variance =  \Trace (W H^{-2} W)
    :param w: the weight of the points already learned
    :param L: Graph Laplacian
    :param y: Labels (The code can handle y being a probability vector, but it doesn't lead to good results so far)
    :param v: can be either a vector or a matrix, if it is a random vector +-1 the matrix inverse will be stochastically approximated, if it is an identity matrix the inverse will be exact (but much slower)
    :return:
    '''
    n = y.shape[0]
    W = diags(w)
    if c.L_eta == 1:
        H_lambda = lambda x: c.AL_alpha*((L @ x)) + (W @ x)
        Ly = L @ y
    elif c.L_eta == 2:
        Ly = L @ (L @ y)
        H_lambda = lambda x: c.AL_alpha*(L.T @ (L @ x)) + (W @ x)
    elif c.L_eta == 3:
        Ly = L @ (L @ (L @ y))
        H_lambda = lambda x: c.AL_alpha*(L @ (L.T @ (L @ x))) + (W @ x)
    elif c.L_eta == 4:
        Ly =L @ (L @ (L @ (L @ y)))
        H_lambda = lambda x: c.AL_alpha * (L @ (L @ (L.T @ (L @ x)))) + (W @ x)
    else:
        raise ValueError('Not implemented')
    H = LinearOperator((n, n), H_lambda)

    bias = cgmatrix(H, Ly,TOL=1e-6, MAXITER=10000, c=c)
    biasSq = np.trace(bias.T @ bias)
    H_bias = cgmatrix(H, bias,TOL=1e-6, MAXITER=10000, c=c)
    dbiasSq = - 2 * np.sum(bias * H_bias,axis=1)

    if c.AL_sigma > 0:
        Wv = W @ v
        Q = cgmatrix(H, Wv,TOL=1e-12, MAXITER=10000, c=c)
        var = np.trace(Q.T @ Q)
        H_Q = cgmatrix(H, Q,TOL=1e-6, MAXITER=10000, c=c)
        dvar = np.squeeze(np.array((2 * np.sum((v-Q)*H_Q,axis=1))))
    else:
        var = 0
        dvar = np.zeros_like(dbiasSq)
    if c.AL_beta > 0:
        cost = np.sum(w)
        dcost = c.AL_beta
    else:
        cost = 0
        dcost = 0
    f = c.AL_alpha**2 * biasSq + c.AL_sigma**2 * var + c.AL_beta * cost
    df = c.AL_alpha**2 * dbiasSq + c.AL_sigma**2 * dvar + dcost
    return f,df


def cgmatrix(A,B,TOL=1e-08,MAXITER=None,M=None,x0=None,callback=None,ATOL=None,c=None):
    '''
    Computes the conjugate gradient solution to Ax=B, where B is a Matrix
    :param A:
    :param B:
    :param tol:
    :param maxiter:
    :param M:
    :param x0:
    :param callback:
    :param atol:
    :return:
    '''
    n,nrhs = B.shape
    t0 = time.time()
    x = np.zeros(B.shape)
    status = -np.ones(nrhs)
    residual = -np.ones(nrhs)
    for i in range(nrhs):
        if sparse.issparse(B):
            b = B[:,i].todense()
        else:
            b = B[:,i]
        x[:,i], status[i] = cg(A, b, x0=x0, tol=TOL, maxiter=MAXITER, M=M, callback=callback, atol=ATOL)
        if c.mode == 'debug':
            bsol = A @ x[:,i]
            residual[i] = norm(bsol - b)

    t1 = time.time()
    c.LOG.debug("CG took {} s, exited with status {} and had residual {}".format(t1-t0, status, residual))
    return x

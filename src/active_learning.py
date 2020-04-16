import time

import numpy as np
from scipy import sparse
from scipy.sparse import identity, diags
from scipy.sparse.linalg import cg


def update_laplacian(L,y,Lmax,idxs,y_truth,idxs_learned=None):
    if idxs_learned is None:
        idxs_learned = set()
    if isinstance(idxs,np.ndarray):
        idxs_to_learn = set(idxs).difference(idxs_learned)
    else:
        idxs_to_learn = set([idxs]).difference(idxs_learned)

    for idx in idxs_to_learn:
        idxs_learned.add(idx)
        _, nn_idxs = L[idx, :].nonzero()
        nn_set = set(nn_idxs)
        nn_usable = nn_set.difference(idxs_learned)
        y[idx, :] = y_truth[idx, :]
        L[idx, idx] = 0
        for idxj in nn_usable:  # Might need to be a list
            y[idxj, :] = y[idxj, :] - L[idx, idxj] / Lmax * y_truth[idx, :]
            L[idx, idxj] = 0
            L[idxj, idxj] = L[idxj, idxj] + L[idxj, idx]
            L[idxj, idx] = 0
    return L,y,idxs_learned

def OEDB(w,L,alpha,beta,lr,maxIter=200,Ifix=[]):
    #we seek the w that minimizes the following objective:
    #obj = trace(H^{-1}) + beta * ||w||_1, st 0<=w<=1
    #where H = L + diag(w)
    #In the following let:
    #bias = trace(H^{-1})
    t0 = time.time()
    v = np.sign(np.random.normal(0,1,(L.shape[0],1)))
    # v = identity(L.shape[0]).tocsc()
    print('Iter    lsIter    nnz(w)       objt        phit     sum(w)      mu       time(s)')

    for i in range(maxIter):
        bias,dbias = getOEDB(w,L,v,alpha)

        obj = alpha * bias + beta * np.sum(w);
        lsiter = 1
        while True:
            wtry = w - lr * (alpha * dbias + beta)
            wtry[wtry < 0] = 0
            wtry[wtry > 1] = 1
            wtry[Ifix] = 1

            biast,_ = getOEDB(wtry,L,v,alpha)
            objt = alpha * biast + beta * np.sum(wtry)
            t1 = time.time()
            print('{:3d}       {:1d}        {:4d}      {:4.2f}     {:4.2f}    {:4.2f}    {:3.1e}    {:.1f}'.format(i,lsiter,np.count_nonzero(wtry),objt,biast,np.sum(wtry),lr,t1-t0))

            if objt < obj:
                break
            lr = lr / 2
            lsiter = lsiter + 1
            if lsiter > 5:
                break;
        if lsiter == 1:
            lr = 1.3 * lr
        w = wtry
    return w


def getOEDB(w,L,v,alpha):

    H = L + alpha * diags(w)
    z,_ = cg(H, v, x0=None, tol=1e-08, maxiter=None, M=None, callback=None, atol=None)
    bias = (v.T @ z[:,None]).diagonal().sum() # we don't have trace on sparse matrices, so we do this instead.
    dbias = np.squeeze(np.asarray(-np.sum(z[:,None]*z[:,None],axis=1))) #Note that this only works for the stochastic version, with a real matrix, it should be z.multiply(z)
    return bias,dbias


def OEDA(w,L,y,alpha,sigma,lr,ns,idx_learned,use_stochastic_approximate=True,safety_stop=200):
    '''
    Finds the next ns points to learn, according to:
    min_w \alpha^2 ||H^{-1} L y ||^2 + \sigma^2 \Trace (W H^{-2} W)
    s.t. 0<= w
    :param w: the weight of the points already learned
    :param L: Graph Laplacian
    :param y: Labels (The code can handle y being a probability vector, but it doesn't lead to good results so far)
    :param alpha: hyperparameter
    :param sigma: hyperparameter - strength of variance term
    :param lr: learning rate
    :param ns: Number of points to learn
    :param idx_learned: Indices of points already learned, will be iteratively updated as new points are found.
    :param use_stochastic_approximate: If true it uses a vector of +-1 to stochasticically approximate the inverse matrices.
    :param safety_stop: This function risk running for a very long time trying to find all the ns samples required, and could even be stuck in an infinite loop if all samples have been learned. This sets the maximum number of iteration to run.
    :return: w with the new indices included
    '''
    t0 = time.time()
    if use_stochastic_approximate:
        v = np.sign(np.random.normal(0,1,(y.shape[0],1)))
    else:
        v = identity(L.shape[0]).tocsc() #TODO there might be a problem here, test that it works
    print('Iter  found      f         bias         var      time(s)')
    nfound = 0
    i = 0
    while True:
        f,df,bias,_,var,_ = getOEDA(w,L,y,v,alpha,sigma)
        ind = np.argmax(np.abs(df))
        w[ind] = w[ind] - lr*df[ind]
        if ind not in idx_learned:
            idx_learned.add(ind)
            nfound += 1
            if nfound >= ns:
                i += 1
                t1 = time.time()
                print("{:3d}  {:3d}    {:3.2e}    {:3.2e}    {:3.2e}   {:.1f}".format(i,nfound,f,alpha**2*bias,sigma**2*var, t1-t0))
                break
        i += 1
        t1 = time.time()
        print("{:3d}   {:3d}    {:3.2e}    {:3.2e}    {:3.2e}     {:.1f}".format(i, nfound, f, alpha ** 2 * bias,
                                                                              sigma ** 2 * var, t1 - t0))
        if i>=safety_stop:
            break
    return w

def getOEDA(w,L,y,alpha,sigma,v):
    '''
    Computes the value and derivatives of:
    f(w) =  \alpha^2 ||H^{-1} L y ||^2 + \sigma^2 \Trace (W H^{-2} W)
    with bias = ||H^{-1} L y ||^2  and variance =  \Trace (W H^{-2} W)
    :param w: the weight of the points already learned
    :param L: Graph Laplacian
    :param y: Labels (The code can handle y being a probability vector, but it doesn't lead to good results so far)
    :param alpha: hyperparameter
    :param sigma: hyperparameter - strength of variance term
    :param v: can be either a vector or a matrix, if it is a random vector +-1 the matrix inverse will be stochastically approximated, if it is an identity matrix the inverse will be exact (but much slower)
    :return:
    '''
    W = diags(w)
    H = L + alpha * W
    bias = cgmatrix(H, L @ y)
    biasSq = np.trace(bias.T @ bias)
    Q = cgmatrix(H, W @ v)
    var = np.trace(Q.T @ Q)
    f = alpha**2 * biasSq + sigma**2 * var
    tmp = cgmatrix(H, bias)
    dbiasSq = - 2 * np.sum(bias * tmp,axis=1)
    tmp = cgmatrix(H, Q)
    dvar = 2 * np.sum((v-Q)*tmp,axis=1)
    df = alpha**2 * dbiasSq + sigma**2 * dvar
    return f,df,biasSq,dbiasSq,var,dvar

def cgmatrix(A,B,tol=1e-08,maxiter=None,M=None,x0=None,callback=None,atol=None):
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
    nrhs = B.shape[1]
    x = np.zeros_like(B)
    for i in range(nrhs):
        x[:,i], _ = cg(A, B[:,i], x0=x0, tol=tol, maxiter=maxiter, M=M, callback=callback, atol=atol)
    return x

def OEDA_validator():
    def numgrid(n):
        """
        NUMGRID Number the grid points in a two dimensional region.
        G = NUMGRID('R',n) numbers the points on an n-by-n grid in
        an L-shaped domain made from 3/4 of the entire square.
        adapted from C. Moler, 7-16-91, 12-22-93.
        Copyright (c) 1984-94 by The MathWorks, Inc.
        """
        x = np.ones((n, 1)) * np.linspace(-1, 1, n)
        y = np.flipud(x.T)
        G = (x > -1) & (x < 1) & (y > -1) & (y < 1) & ((x > 0) | (y > 0))
        G = np.where(G, 1, 0)  # boolean to integer
        k = np.where(G)
        G[k] = 1 + np.arange(len(k[0]))
        return G

    def delsq(G):
        """
        DELSQ  Construct five-point finite difference Laplacian.
        delsq(G) is the sparse form of the two-dimensional,
        5-point discrete negative Laplacian on the grid G.
        adapted from  C. Moler, 7-16-91.
        Copyright (c) 1984-94 by The MathWorks, Inc.
        """
        [m, n] = G.shape
        # Indices of interior points
        G1 = G.flatten()
        p = np.where(G1)[0]
        N = len(p)
        # Connect interior points to themselves with 4's.
        i = G1[p] - 1
        j = G1[p] - 1
        s = 4 * np.ones(p.shape)

        # for k = north, east, south, west
        for k in [-1, m, 1, -m]:
            # Possible neighbors in k-th direction
            Q = G1[p + k]
            # Index of points with interior neighbors
            q = np.where(Q)[0]
            # Connect interior points to neighbors with -1's.
            i = np.concatenate([i, G1[p[q]] - 1])
            j = np.concatenate([j, Q[q] - 1])
            s = np.concatenate([s, -np.ones(q.shape)])
        # sparse matrix with 5 diagonals
        return sparse.csr_matrix((s, (i, j)), (N, N))
    G = numgrid(18)
    L = delsq(G)
    alpha = 1
    sigma = 1
    w = np.random.rand(L.shape[0])
    y = np.random.rand(L.shape[0],3)
    v = identity(L.shape[0]).toarray()
    f,df,bias,dbias,var,dvar = getOEDA(w, y, L, v,alpha,sigma)
    h = 1e-1
    dw = np.random.rand(L.shape[0])
    bias_1o = 1
    bias_2o = 1
    var_1o = 1
    var_2o = 1
    for i in range(10):
        fp, dfp, biasp, dbiasp, varp, dvarp = getOEDA(w+h*dw, y, L, v, alpha, sigma)
        bias_1 = np.abs(biasp-bias)
        bias_2 = np.abs(biasp-bias-h*(dw[:,None].T @ dbias)[0])
        var_1 = np.abs(varp - var)
        var_2 = np.abs(varp - var - h*(dw[:,None].T @ dvar)[0])
        print("{:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}".format(h,bias_1,bias_2,var_1,var_2,bias_1o/bias_1,bias_2o/bias_2,var_1o/var_1,var_2o/var_2))
        bias_1o = bias_1
        bias_2o = bias_2
        var_1o = var_1
        var_2o = var_2
        h /= 2

if __name__ == '__main__':
    OEDA_validator()

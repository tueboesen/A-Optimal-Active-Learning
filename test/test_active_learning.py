import numpy as np
from scipy import sparse
from scipy.sparse import identity

from src.active_learning import getOEDA


def teest_OEDA():
    '''
    https://www.dummies.com/education/math/calculus/calculating-error-bounds-for-taylor-polynomials/
    :return:
    '''
    # TODO Update this to the actual residual of the taylor series

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
    f,df,bias,dbias,var,dvar = getOEDA(w, L, y, alpha, sigma, v)
    h = 1e-1
    dw = np.random.rand(L.shape[0])
    bias_1o = 1
    bias_2o = 1
    var_1o = 1
    var_2o = 1
    for i in range(10):
        fp, dfp, biasp, dbiasp, varp, dvarp = getOEDA(w+h*dw, L, y, alpha, sigma, v)
        bias_1 = np.abs(biasp-bias)
        bias_2 = np.abs(biasp-bias-h*(dw[:,None].T @ dbias)[0])
        var_1 = np.abs(varp - var)
        var_2 = np.abs(varp - var - h*(dw[:,None].T @ dvar)[0])
        print("{:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.2e}   {:3.10e}   {:3.10e}   {:3.10e}   {:3.10e}".format(h,bias_1,bias_2,var_1,var_2,bias_1o/bias_1,bias_2o/bias_2,var_1o/var_1,var_2o/var_2))
        if i > 5:
            assert np.abs(bias_1o/bias_1 -2)<h*10, 'The bias is not calculated correctly'
            assert np.abs(bias_2o/bias_2 - 4) < h * 10, 'The bias derivative is not calculated correctly'
            assert np.abs(var_1o/var_1 - 2) < h * 10, 'The variance is not calculated correctly'
            assert np.abs(var_2o/var_2 - 4) < h * 10, 'The variance derivative is not calculated correctly'
        bias_1o = bias_1
        bias_2o = bias_2
        var_1o = var_1
        var_2o = var_2
        h /= 2
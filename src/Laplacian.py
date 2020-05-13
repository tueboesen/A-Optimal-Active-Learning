import time

import hnswlib
import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix


def ANN_hnsw(x, k=10, euclidian_metric=False, union=True, eff=None,cutoff=False):
    '''
    Calculates the approximate nearest neighbours using the Hierarchical Navigable Small World Graph for fast ANN search. see: https://github.com/nmslib/hnswlib
    :param x: 2D numpy array with the first dimension being different data points, and the second the features of each point.
    :param k: Number of neighbours to compute
    :param euclidian_metric: Determines whether to use cosine angle or euclidian metric. Possible options are: 'l2' (euclidean) or 'cosine'
    :param union: The adjacency matrix will be made symmetrical, this determines whether to include the connections that only go one way or remove them. If union is True, then they are included.
    :param eff: determines how accurate the ANNs are built, see https://github.com/nmslib/hnswlib for details.
    :param cutoff: Includes a cutoff distance, such that any connection which is smaller than the cutoff is removed. If True, the cutoff is automatically calculated, if False, no cutoff is used, if a number, it is used as the cutoff threshold. Note that the cutoff has a safety built in that makes sure each data point has at least one neighbour to minimize the risk of getting a disjointed graph.
    :return: Symmetric adjacency matrix, mean distance of all connections (including the self connections)
    '''
    nsamples = len(x)
    dim = len(x[0])
    # Generating sample data
    data = x
    data_labels = np.arange(nsamples)
    if eff is None:
        eff = nsamples

    # Declaring index
    if euclidian_metric:
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
    else:
        p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=nsamples, ef_construction=eff, M=500)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(eff)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k)
    t2 = time.time()

    if cutoff:
        if type(cutoff) is bool: # Automatically determine the threshold
            dd_mean = np.mean(distances)
            dd_var = np.var(distances)
            dd_std = np.sqrt(dd_var)
            threshold = dd_mean+dd_std
        else:
            threshold = cutoff
        useable = distances < threshold
        useable[:,1] = True # This should hopefully prevent any element from being completely disconnected from the rest (however it might just make two remote elements join together apart from the rest)
    else:
        useable = distances == distances


    Js = []
    Is = []
    for i,(subnn,useable_row) in enumerate(zip(labels,useable)):
        for (itemnn,useable_element) in zip(subnn,useable_row):
            if useable_element:
                Js.append(itemnn)
                Is.append(i)
    Vs = np.ones(len(Js),dtype=np.int64)
    A = csr_matrix((Vs, (Is,Js)),shape=(nsamples,nsamples))
    A.setdiag(0)
    if union:
        A = (A + A.T).sign()
    else:
        A = A.sign()
        dif = (A-A.T)
        idx = dif>0
        A[idx] = 0
    A.eliminate_zeros()
    return A, np.mean(distances)




def compute_laplacian(features,metric='l2',knn=9,union=True,cutoff=False,Lsym=True):
    '''
    Computes a knn Graph-Laplacian based on the features given.
    Note that there is room for improvement here, the graph laplacian could be built directly on the distances found by the ANN search (which are approximate) this would inherently ensure that the ANNs actually match the metric used in the graph laplacian, and make it faster.
    :param features: Features the graph laplacian will be built on. These can either be given as a torch tensor or numpy array. The first dimension should contain the number of samples, all other dimensions will be flattened.
    :param metric: The metric to use when computing approximate neares neighbours. Possible options are l2 or cosine
    :param knn: number of nearest neighbours to compute
    :param union: The adjacency matrix will be made symmetrical, this determines whether to include the connections that only go one way or remove them. If union is True, then they are included.
    :param cutoff: Includes a cutoff distance, such that any connection which is smaller than the cutoff is removed. If True, the cutoff is automatically calculated, if False, no cutoff is used, if a number, it is used as the cutoff threshold. Note that the cutoff has a safety built in that makes sure each data point has at least one neighbour to minimize the risk of getting a disjointed graph.
    :return: Graph Laplacian, Adjacency matrix
    '''
    t1 = time.time()
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = features.reshape(features.shape[0], -1)
    A, dd = ANN_hnsw(features, euclidian_metric=metric, union=union, k=knn, cutoff=cutoff)
    t2 = time.time()
    if metric == 'l2':
        L, L_sym = Laplacian_Euclidian(features, A, dd)
    elif metric == 'cosine':
        L, L_sym = Laplacian_angular(features, A)
    else:
        raise ValueError('{} is not an implemented metric'.format(metric))
    t3 = time.time()
    print('ANN = {}'.format(t2-t1))
    print('L = {}'.format(t3-t2))
    if Lsym:
        L_select = L_sym
    else:
        L_select = L
    return L_select,A



def Laplacian_Euclidian(X, A, sigma, dt=None):
    '''
    Computes the Graph Laplacian as: L_ij = A_ij * exp(- ||X_i - X_j||_2^2 / sigma)
    :param X: 2D numpy array with the first dimension being different data points, and the second the features of each point.
    :param A: Adjacency matrix built with the same metric.
    :param sigma: characteristic distance
    :param dt: datatype the returned Laplacian should have, if not used, it will default to whatever the datatype of X is.
    :return: Graph Laplacian, Normalized symmetric Graph Laplacian
    '''
    if dt is None:
        dt = X.dtype
    A = A.tocoo()
    n,_=A.shape
    I = A.row
    J = A.col
    tmp = np.sum((X[I] - X[J]) ** 2, axis=1)
    V = np.exp(-tmp / (sigma))
    W = coo_matrix((V, (I, J)), shape=(n, n))
    D = coo_matrix((n, n), dtype=dt)
    if np.min(np.abs(np.sum(W,axis=0))) == 0:
        print("If a point has zero here, it means that it is so far away from all other points, that the exponential distance is infinite, hence similarity is zero, and is effectively unconnected. This might be a problem")
    coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
    Dh = np.sqrt(D)
    np.reciprocal(Dh.data, out=Dh.data)
    L = D - W
    L_sym = Dh @ L @ Dh
    # (abs(L - L.T) > 1e-10).nnz == 0
    L_sym = 0.5 * (L_sym.T + L_sym)
    return L,L_sym

def Laplacian_angular(X, A,dt=None):
    '''
    Computes the Graph Laplacian with cosine angular metric: L_ij = A_ij * (1 - (X_i X_j') /(||X_i||*||X_j||)
    :param X: 2D numpy array with the first dimension being different data points, and the second the features of each point.
    :param A: Adjacency matrix built with the same metric.
    :param dt: datatype the returned Laplacian should have, if not used, it will default to whatever the datatype of X is.
    :return: Graph Laplacian, Normalized symmetric Graph Laplacian
    '''
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if dt is None:
        dt = X.dtype
    A = A.tocoo()
    n, _ = A.shape
    I = A.row
    J = A.col
    V = 1-np.sum((X[I] * X[J]), axis=1)/(np.sqrt(np.sum(X[I]**2, axis=1))*np.sqrt(np.sum(X[J]**2, axis=1)))
    V[V<0] = 0 #Numerical precision can sometimes lead to some elements being less than zero.
    if np.max(V) < 1:
        print("some elements of V are larger than 1. This means that some neighbours are less than ortogonal, hence absolutely terrible neighbours. What are you doing?")
    # assert np.max(V) < 1, "some elements of V are larger than 1. This means that some neighbours are less than ortogonal, hence absolutely terrible neighbours. What are you doing?"
    W = coo_matrix((V, (I, J)), shape=(n, n))
    D = coo_matrix((n, n), dtype=dt)
    coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
    Dh = np.sqrt(D)
    np.reciprocal(Dh.data, out=Dh.data)
    L = D - W
    L_sym = Dh @ L @ Dh
    L_sym = 0.5 * (L_sym.T + L_sym)
    return L, L_sym

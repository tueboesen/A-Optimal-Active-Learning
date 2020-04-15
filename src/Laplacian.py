from annoy import AnnoyIndex
import time
import scipy.io as sio
from os import walk
import h5py
import numpy as np
import datetime
import statistics
from scipy.sparse import coo_matrix,csr_matrix, identity,triu,tril,diags,spdiags
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spsolve_triangular
import torch
import hnswlib
from scipy.special import softmax
import matplotlib.pyplot as plt
from torch import optim
import matplotlib.cm as cm


def ANN_hnsw(x, k=10, euclidian_metric=False, union=True, eff=None,cutoff=False):
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
    p.init_index(max_elements=nsamples, ef_construction=eff, M=200)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(eff)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k)
    t2 = time.time()

    if cutoff:
        #Remove elements from A which are below a certain threshold
        dd_mean = np.mean(distances)
        dd_var = np.var(distances)
        dd_std = np.sqrt(dd_var)
        threshold = dd_mean+dd_std
        useable = distances < threshold
        useable[:,1] = True #This should hopefully prevent any element from being completely disconnected from the rest (however it might just make two remote elements join together apart from the rest)
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




def compute_laplacian(features,metric='l2',knn=9,union=True,cutoff=False):
    #Calculates the graph Laplacian
    #features should be either a numpy array or tensor (tensors will be converted to numpy array)
    #first dimension of features should be different samples, all other dimensions will be flattened.
    t1 = time.time()
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = features.reshape(features.shape[0], -1)
    A, dd = ANN_hnsw(features, euclidian_metric=metric, union=union, k=knn, cutoff=cutoff)
    t2 = time.time()
    if metric == 'l2':
        L, _ = Laplacian_Euclidian(features, A, dd)
    else:
        L, _ = Laplacian_angular(features, A)
    t3 = time.time()
    print('ANN = {}'.format(t2-t1))
    print('L = {}'.format(t3-t2))
    return L,A



def Laplacian_Euclidian(X,A,dist,distfactor=1,dt=None):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if dt is None:
            dt = X.dtype
        A = A.tocoo()
        n,_=A.shape
        I = A.row
        J = A.col
        tmp = np.sum((X[I] - X[J]) ** 2, axis=1)
        V = np.exp(-tmp/(dist*distfactor))
        W = coo_matrix((V, (I, J)), shape=(n, n))
        D = coo_matrix((n, n), dtype=dt)
        coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
        Dh = np.sqrt(D)
        np.reciprocal(Dh.data, out=Dh.data)
        L = D - W
        L_sym = Dh @ L @ Dh
        # (abs(L - L.T) > 1e-10).nnz == 0
        L_sym = 0.5 * (L_sym.T + L_sym)
        return L,L_sym

def Laplacian_ICEL(X, A,alpha):
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocoo()
    nx, _ = A.shape
    Is = A.row
    Js = A.col

    V = np.sum((X[Is] * X[Js]) ** 3, axis=1)
    assert np.min(V) > 0, print("some elements of V are less than zero")
    Aa = coo_matrix((V, (Is, Js)), shape=(nx, nx))
    W = Aa.T + Aa
    D = coo_matrix((nx, nx))
    coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
    Dh = np.sqrt(D)
    np.reciprocal(Dh.data, out=Dh.data)
    Ww = Dh @ W @ Dh
    I = identity(nx)
    L = (I - alpha * Ww)
    return L

def Laplacian_angular(X, A,dt=None):
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
    assert np.max(V) < 1, print("some elements of V are larger than 1")  # This means that it has found terrible neighbors, this could happen if the sample size is very small.
    W = coo_matrix((V, (I, J)), shape=(n, n))
    D = coo_matrix((n, n), dtype=dt)
    coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
    Dh = np.sqrt(D)
    np.reciprocal(Dh.data, out=Dh.data)
    L = D - W
    L_sym = Dh @ L @ Dh
    L_sym = 0.5 * (L_sym.T + L_sym)
    return L, L_sym


def SSL_clustering(alpha, L, Uobs,balance_weights=False):
    TOL = 1e-12;
    MAXITER = 2000;
    if isinstance(Uobs, torch.Tensor):
        Uobs = Uobs.numpy()
    n,nc = Uobs.shape
    idxs = np.nonzero(Uobs[:,0])[0]
    nidxs = len(idxs)
    tmp = np.zeros(n)
    if balance_weights:
        labels = np.argmax(Uobs[idxs, :], axis=1)
        nlabelsprclass = np.bincount(labels,minlength=10)
        for idx,w in enumerate(nlabelsprclass):
            if w != 0:
                idxs_i = idxs[labels == idx]
                tmp[idxs_i] = 1/w
    else:
        tmp[idxs] = 1
    W = diags(tmp)
    I_nc = identity(nc)
    e = np.ones((nc,1))
    C = I_nc - (e @ e.T) / (e.T @ e) #Make sure this is vector products except the division which is elementwise
    A = (1 / n * alpha * L + 1 / nidxs * W)
    b = 1 / nidxs * W @ Uobs @ C #Also matrix products
    def precond(x):
        return spsolve(tril(A, format='csc'), (A.diagonal() * spsolve(triu(A, format='csc'), x, permc_spec='NATURAL')), permc_spec='NATURAL')
    M = LinearOperator(matvec=precond, shape=(n, n), dtype=float)
    U = np.empty_like(Uobs)
    for i in range(nc):
        U[:,i], _ = cg(A, b[:,i], M=M,tol=TOL,maxiter=MAXITER)
    return U

def create_uobs_from_dataset(dataset):
    CC = 10
    n = len(dataset)
    if dataset.use_label_probabilities:
        n,nc = dataset.labels.shape
        Uobs = np.zeros((n,nc))
        for i,obs in enumerate(dataset.labels):
            if obs[0].item() != -1:
                Uobs[i, :] = -CC
                idx = torch.argmax(obs).item()
                Uobs[i, idx] = CC * nc - CC
    else:
        classes = np.unique(dataset.labels_true)
        nc = len(classes)
        Uobs = np.zeros((n,nc))
        for i,obs in enumerate(dataset.labels):
            if obs != -1:
                Uobs[i,:] = -CC
                Uobs[i,obs] = CC*nc-CC
    return Uobs


def find_weights_from_labels(y,idxs,include_class_balance=True):
    '''
    :param y: label probabilities
    :param idxs: known labels
    :return: w: weights for the training
    '''
    #This function finds the training weights
    #The training weights can include a lot of different components
    #Entropy (how certain we are of the label being correct.)
    #Class balance (classes with few examples get higher weights, but it isn't just that. The weight is based on the total amount of weight in a class.)
    #Class_balance makes sure that the total weight in each class is equal.
    #The weight vector is made such that it will sum to n, (hence each point will have an average weight of 1).

    #First calculate the entropy weight.
    n,nc = y.shape
    labels = np.argmax(y,axis=1)
    w = np.zeros(n)
    # ysort = np.sort(y,axis=1)
    # w_entropy = ysort[:,-1] - ysort[:,-2]
    w_entropy = y[np.arange(y.shape[0]),labels]
    w_entropy[idxs] = 1 #All the known labels we set to 1.

    #Now lets balance that weight among the classes
    if include_class_balance:
        #Find the total weight in a class:
        w_sum = np.zeros(nc)
        for i in range(nc):
            w_sum[i] = np.sum(w_entropy[labels == i])
            w[labels == i] = (n/nc) * w_entropy[labels == i] / w_sum[i]
    else:
        w = w_entropy / np.sum(w_entropy) * n
    return w


def create_cardinal_Weight(U):
    #This one should not just balance the weight according to the cardinal number, but to the entropy weights.
    #Hence it should come after
    nx, nc = U.shape
    dt = U.dtype
    e1 = np.ones((nx, 1), dtype=dt)
    weights = nx / nc * U @ (U.T @ e1)
    return weights



def analyse_probability_matrix(U,dataset,LOG,L):
    Cpred = np.argmax(U,axis=1)
    nc = dataset.nc
    classes = np.asarray(range(nc))
    if dataset.use_label_probabilities:
        Ctrue = torch.argmax(dataset.labels_true,dim=1).numpy()
        idxselect = np.nonzero((dataset.labels[:,0] >= 0).numpy())
        Cselect = torch.argmax(dataset.labels[idxselect,:][0],dim=1).numpy()
    else:
        Ctrue = dataset.labels_true
        Cselect = [num for num in dataset.labels if num >= 0]
    nselect = np.bincount(Cselect,minlength=10)
    A = np.zeros((nc,nc),dtype=int)
    for i in classes: #pred class
        Ctrue_i = np.asarray(Ctrue)[Cpred == i]
        for j in classes: #true class
            A[i,j] = sum(Ctrue_i == j)

    LOG.info("Labels selected:")
    LOG.info("Class       : {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:>8s}".format(0,1,2,3,4,5,6,7,8,9,'total'))
    LOG.info("selected (#): {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(*nselect,np.sum(nselect)))
    LOG.info("selected (%): {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(*nselect/sum(nselect)*100))
    LOG.info(" ")
    LOG.info("Based on labels selected, the clustering predicted:")
    LOG.info("Predicted \\ True {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:>8s}".format(0,1,2,3,4,5,6,7,8,9,'total'))
    LOG.info("------------------------------------------------------------------------------------------------")
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(0,*A[0,:],np.sum(A[0,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(1,*A[1,:],np.sum(A[1,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(2,*A[2,:],np.sum(A[2,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(3,*A[3,:],np.sum(A[3,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(4,*A[4,:],np.sum(A[4,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(5,*A[5,:],np.sum(A[5,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(6,*A[6,:],np.sum(A[6,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(7,*A[7,:],np.sum(A[7,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(8,*A[8,:],np.sum(A[8,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(9,*A[9,:],np.sum(A[9,:])))
    LOG.info("------------------------------------------------------------------------------------------------")
    LOG.info("        {:>6s} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format('total',*np.sum(A[:,:],axis=0),np.sum(A)))
    LOG.info(" ")
    LOG.info("Accuracy = {}%".format(sum(Cpred == Ctrue)/len(Ctrue)*100))



    #Lets also probe the certainty of all points.
    plot_distribution_matrix(U)

    #Lets look at L*U
    Usoft = softmax(U,axis=1)
    UL2 = np.zeros_like(U)
    for i in range(U.shape[1]):
        UL2[:,i] = L @ Usoft[:,i]
    plot_distribution_matrix(UL2)
    return

def plot_distribution_matrix(v,save=None,iter=None):
    cpv = softmax(v,axis=1)
    maxval = np.sum(np.abs(v), axis=1)
    vshifted = v - np.min(v,axis=1)[:,None]
    cpv2 = vshifted / (maxval[:,None]+1e-10)
    fig = plt.figure(figsize=[10, 10])
    plt.subplot(1, 3, 1);
    plt.imshow(cpv, cmap=cm.jet, aspect='auto', vmin=0, vmax=1)
    plt.title('softmax')
    plt.colorbar()
    plt.subplot(1, 3, 2);
    plt.imshow(cpv2, cmap=cm.jet, aspect='auto', vmin=0, vmax=1)
    plt.title('linear')
    plt.colorbar()

    plt.subplot(1, 3, 3);
    hot_v = np.zeros_like(v)
    labels = np.argmax(v,axis=1)
    for i,label in enumerate(labels):
        if sum(np.max(v[i,:]) == v[i,:]) > 1:
            pass
        else:
            hot_v[i,label] = 1
    plt.imshow(hot_v, cmap=cm.gray, aspect='auto')
    plt.pause(1)
    if save:
        fileloc = "{}/{}_{}.png".format(save, 'distribution',iter)
        fig.savefig(fileloc)
    return

def convert_pseudo_to_prob(v,use_softmax=False):
    if use_softmax:
        cpv = softmax(v,axis=1)
    else:
        maxval = np.sum(np.abs(v), axis=1)
        vshifted = v - np.min(v,axis=1)[:,None]
        cpv = vshifted / (maxval[:,None]+1e-10)
    return cpv


def analyse_features(U,dataset,LOG,save=None,iter=None):
    if isinstance(U, torch.Tensor):
        U = U.numpy()
    nc = dataset.nc
    classes = np.asarray(range(nc))
    Cpred = np.argmax(U,axis=1)
    Ctrue = dataset.labels_true
    if dataset.use_label_probabilities:
        Ctrue = torch.argmax(dataset.labels_true,dim=1).numpy()
        idxselect = np.nonzero((dataset.labels[:,0] >= 0).numpy())
        Cselect = torch.argmax(dataset.labels[idxselect,:][0],dim=1).numpy()
    else:
        Ctrue = dataset.labels_true
        Cselect = [num for num in dataset.labels if num >= 0]
    nselect = np.bincount(Cselect,minlength=10)
    A = np.zeros((nc,nc),dtype=int)
    for i in classes: #pred class
        Ctrue_i = np.asarray(Ctrue)[Cpred == i]
        for j in classes: #true class
            A[i,j] = sum(Ctrue_i == j)

    LOG.info("Based on labels selected, the neural network predicted:")
    LOG.info("Predicted \\ True {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:>8s}".format(0,1,2,3,4,5,6,7,8,9,'total'))
    LOG.info("------------------------------------------------------------------------------------------------")
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(0,*A[0,:],np.sum(A[0,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(1,*A[1,:],np.sum(A[1,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(2,*A[2,:],np.sum(A[2,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(3,*A[3,:],np.sum(A[3,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(4,*A[4,:],np.sum(A[4,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(5,*A[5,:],np.sum(A[5,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(6,*A[6,:],np.sum(A[6,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(7,*A[7,:],np.sum(A[7,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(8,*A[8,:],np.sum(A[8,:])))
    LOG.info("        {:6d} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format(9,*A[9,:],np.sum(A[9,:])))
    LOG.info("------------------------------------------------------------------------------------------------")
    LOG.info("        {:>6s} | {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} {:6d} |{:8d}".format('total',*np.sum(A[:,:],axis=0),np.sum(A)))
    LOG.info(" ")
    LOG.info("Accuracy = {}%".format(sum(Cpred == Ctrue)/len(Ctrue)*100))

    #Lets also probe the certainty of all points.
    plot_distribution_matrix(U,save,iter)

    Uprob = softmax(U, axis=1)
    Uprob_max = Uprob[np.arange(Uprob.shape[0]),Cpred]
    idx_wrong = (Cpred != Ctrue)
    idx_right = (Cpred == Ctrue)
    Uprob_max_wrong = Uprob_max[idx_wrong]
    Uprob_max_right = Uprob_max[idx_right]
    np.mean(Uprob_max_wrong)
    np.var(Uprob_max_wrong)
    np.mean(Uprob_max_right)
    np.var(Uprob_max_right)

    fig = plt.figure(figsize=[10, 10])
    plt.subplot(1, 2, 1);
    plt.hist(Uprob_max_wrong,bins=100)
    tit = "wrong labels certainty: {:.4f} +- {:.4f}".format(np.mean(Uprob_max_wrong), np.var(Uprob_max_wrong))
    LOG.info(tit)
    plt.title(tit)
    plt.subplot(1, 2, 2);
    plt.hist(Uprob_max_right,bins=100)
    tit = "right labels certainty: {:.4f} +- {:.4f}".format(np.mean(Uprob_max_right),np.var(Uprob_max_right))
    LOG.info(tit)
    plt.title(tit)
    plt.pause(1)
    if save:
        fileloc = "{}/{}_{}.png".format(save, 'histogram',iter)
        fig.savefig(fileloc)
    return
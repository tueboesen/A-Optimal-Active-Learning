import random
import time
import copy
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse import identity, diags
from scipy.sparse.linalg import cg, LinearOperator

from src.Clustering import SSL_clustering, SSL_clustering_AL, SSL_clustering_sq, SSL_clustering_1vsall
from src.Laplacian import compute_laplacian
from src.dataloader import set_labels
from src.optimization import train, eval_net
from src.report import analyse_probability_matrix, analyse_features
from src.utils import update_results, save_results, setup_results
from src.visualization import plot_results, preview, debug_circles


def select_active_learning_method(name,c,dataset):
    '''
    Selects the active learning method to use. Each method is defined by a class.
    :param name: name of the method to use
    :param c: context variable, which contains all the input parameters
    :param labels: the true labels. These are needed by the passive learning method, when they need to decide which labels to select in a label balanced scheme.
    :return: a callable function returns the next indices.
    '''
    if name == 'active_learning_adaptive':
        method = Adaptive_active_learning(c['alpha'], c['beta'], c['sigma'], c['lr_AL'], c['nlabels_pr_class'], c['use_1_vs_all'],dataset.plabels_true)
    elif name == 'passive_learning':
        method = passive_learning_learning(dataset.labels_true, c['nlabels_pr_class'], class_balance=False)
    elif name == 'passive_learning_balanced':
        method = passive_learning_learning(dataset.labels_true, c['nlabels_pr_class'], class_balance=True)
    else:
        raise ValueError("Method has not been implemented yet")
    return method

def run_active_learning(net,optimizer,loss_fnc,dataloader_train,dataloader_validate,c,LOG,AL_fnc,L,device,saveprefix=None):
    '''
    Active learning main routine. This function assumes an adaptive active learning approach, where the known labels are adaptively probed.
    The routine, starts by knowing c['nlabels'] equally balanced labels between all the classes.

    :param net: neural net to train on the labels found
    :param optimizer:
    :param loss_fnc:
    :param dataloader_train:
    :param dataloader_validate:
    :param c:
    :param LOG:
    :param AL_fnc:
    :param L:
    :param device:
    :return:
    '''
    nc = dataloader_train.dataset.nc
    results = setup_results()

    # We start by randomly assigning nlabels
    dataloader_train = set_labels(c['nlabels'], dataloader_train, class_balance=True)
    preview(dataloader_train,save="{}_{}.png".format(saveprefix, 'Initial_labels'))
    y = dataloader_train.dataset.plabels.numpy()
    idxs = np.nonzero(y[:, 0])[0]
    w = np.zeros(L.shape[0])
    w[idxs] = 1e7
    L = L + 1e-3 * identity(L.shape[0])
    # L = L.T @ L
    idx_learned = list(idxs)
    t0 = time.time()
    for i in range(c['epochs_AL']):

        # We use Active learning to find the next batch of data points to label
        idx_learned,w = AL_fnc(idx_learned,L,y,w,LOG,c,(dataloader_train.dataset.imgs).numpy(),"{}{}".format(saveprefix,i))

        # With the data points found, we update the labels in our dataset
        dataloader_train = set_labels(idx_learned, dataloader_train)
        yobs = dataloader_train.dataset.plabels

        # We predict the labels for all the unknown points
        # y = SSL_clustering(c['alpha'], L, yobs, balance_weights=True)
        # y = SSL_clustering_AL(c['alpha'], L, yobs, w)
        y = SSL_clustering_1vsall(c['alpha'], L, yobs, w, TOL=1e-12, iterated_laplacian=c['iterated_laplacian'])
        cluster_acc = analyse_probability_matrix(y, dataloader_train.dataset, LOG, L,saveprefix=saveprefix,iter=i)
        # y = SSL_clustering_1vsall(c['alpha'], L, yobs, w)
        # cluster_acc = analyse_probability_matrix(y, dataloader_train.dataset, LOG, L,saveprefix=saveprefix,iter=i)

        if c['use_SL']:
            y[idx_learned] = dataloader_train.dataset.plabels[idx_learned]  # We update the known plabels to their true value
            dataloader_train = set_labels(y, dataloader_train)  # We save the label probabilities in y, into the dataloader
            # train a network on this data
            netAL, validator_acc = train(net, optimizer, dataloader_train, loss_fnc, LOG, device=device, dataloader_validate=dataloader_validate,
                          epochs=c['epochs_SL'], use_probabilities=c['use_label_probabilities'])
            features_netAL = eval_net(netAL, dataloader_train.dataset, device=device)  # we should probably have the option of combining these with the previous features.
            learning_acc = analyse_features(features_netAL, dataloader_train.dataset, LOG, save=c['result_dir'], iter=i)
            if c['recompute_L']:
                L, A = compute_laplacian(features_netAL, metric=c['metric'], knn=c['knn'], union=True)
                L = L + 1e-2 * identity(L.shape[0])
                # L = L.T @ L
        else:
            learning_acc = 0
            validator_acc = 0
        update_results(results, idx_learned, cluster_acc, learning_acc, validator_acc)
        LOG.info("iter:{}, acc:{:2.2f}%, time:{:.2f}".format(i,cluster_acc,time.time()-t0))
    return results, net



class Adaptive_active_learning():
    def __init__(self,alpha,beta,sigma,lr,nlabels_pr_class,use_1_vs_all,plabels,debug=False):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.lr = lr
        self.plabels = plabels
        self.nlabels_pr_class = nlabels_pr_class
        self.use_1_vs_all = use_1_vs_all
        self.debug = debug
        super(Adaptive_active_learning, self).__init__()

    def __call__(self, idx_learned,L,yobs,w,LOG,c,xy,saveprefix):
        idx_learned = set(idx_learned)
        n, nc = yobs.shape
        # w = np.zeros(n)
        # w[idx_learned] = 1
        if self.use_1_vs_all:
            yi = np.zeros((n, 1))
            for j in range(nc):
                yi[:, 0] = yobs[:, j]
                yi = np.sign(yi)
                t0 = time.time()
                yi = SSL_clustering_AL(self.alpha, L, yi,w,iterated_laplacian=c['iterated_laplacian'])
                t1 = time.time()
                w = OEDA_v2(w, L, yi, self.alpha, self.beta, self.sigma, self.lr, self.nlabels_pr_class, idx_learned, LOG,c,xy,saveprefix="{}_{}".format(saveprefix,j))
                t2 = time.time()
                idx = np.nonzero(w)[0]
                yobs[idx,:] = self.plabels[idx,:]
                idx_learned.update(idx)
                if c['mode'] == 'debug':
                    LOG.info("Clustering {:2.2f}, OEDA {:2.2f}".format(t1 - t0, t2 - t1))
        else:
            t0 = time.time()
            yobs = SSL_clustering_AL(self.alpha, L, yobs, w)
            t1 = time.time()
            w = OEDA_v2(w, L, yobs, self.alpha, self.beta, self.sigma, self.lr, self.nlabels_pr_class, idx_learned, LOG,xy)
            t2 = time.time()
            idx = np.nonzero(w)[0]
            idx_learned.update(idx)
            print("Clustering {:2.2f}, OEDA {:2.2f}".format(t1-t0,t2-t1))
        return list(idx_learned),w


class passive_learning_learning():
    def __init__(self,labels,nlabels_pr_class,class_balance=True):
        self.nlabels_pr_class = nlabels_pr_class
        self.labels = labels
        self.class_balance = class_balance
        super(passive_learning_learning, self).__init__()

    def __call__(self, idx_learned,L,yobs,w,LOG,*args):
        n,nc = yobs.shape
        wbase = w[list(idx_learned)[0]]

        labels = self.labels
        if self.class_balance:
            labels_unique = np.asarray(range(nc))
            for i, label in enumerate(labels_unique):
                indices = np.where(labels == label)[0]
                indices = list(set(indices).difference(set(idx_learned)))  # Remove known indices
                assert len(indices) >= self.nlabels_pr_class, "There were not enough datapoints of class {} left. Needed {} indices, but there are only {} available. Try increasing the dataset.".format(i, self.nlabels_pr_class, len(indices))
                np.random.shuffle(indices)
                idx_learned += indices[0:self.nlabels_pr_class]
        else:
            indices = list(set(range(n)).difference(set(idx_learned)))
            random.shuffle(indices)
            idx_learned += indices[:self.nlabels_pr_class*nc]
        w[list(idx_learned)] = wbase
        return idx_learned,w



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

def OEDA_v2(w,L,y,alpha,beta,sigma,lr,ns,idx_learned,LOG,c,xy,use_stochastic_approximate=True,safety_stop=2000,saveprefix=None,debug=False):
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
    if ns <= 0:
        return w
    if use_stochastic_approximate:
        v = np.sign(np.random.normal(0,1,(y.shape[0],1)))
    else:
        v = identity(L.shape[0]).tocsc() #TODO there might be a problem here, test that it works
    f,df,bias,dbias,var,dvar,cost,dcost,bias_pp = getOEDA(w,L,y,alpha,beta,sigma,v,c)
    LOG.info("f = {:2.2e}, bias = {:2.2e}, var = {:2.2e}".format(f, alpha**2 * bias, sigma**2 * var))
    indices = np.argsort(np.abs(df))[::-1]
    i = 0
    idx_learned_in = copy.deepcopy(idx_learned)
    idx_excluded = []
    idx_new = set()

    L2 = L.T @ L
    for idx in indices:
        if idx not in (idx_learned and idx_excluded):
            i += 1
            idx_learned.add(idx)
            idx_new.add(idx)
            if i >= ns:
                break
            else:
                # Remove all neighbouring points from the potential candidates
                aa = L2.nonzero()
                tmp = np.where(aa[0] == idx)[0]
                idxs = aa[1][tmp]
                idx_excluded += idxs.tolist()
    w[list(idx_learned)] = 1e7
    t1 = time.time()
    if (c['mode'] == 'debug') or (c['mode'] == 'paper'):
        debug_circles(xy,y,df,dbias,dvar,idx_learned_in,idx_new,saveprefix)
    t2 = time.time()
    # print("GetOEDA {:2.2f}, viz {:2.2f}".format(t1-t0,t2-t1))
    return w



def getOEDA(w,L,y,alpha,beta,sigma,v,c):
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
    n = y.shape[0]
    W = diags(w)
    m = c['iterated_laplacian']
    debug = c['mode'] == 'debug'
    # Ly = np.linalg.matrix_power(L, c['iterated_laplacian']) @ y
    if m == 1:
        H_lambda = lambda x: alpha*((L @ x)) + (W @ x)
        Ly = L @ y
    elif m == 2:
        Ly = L @ L @ y
        H_lambda = lambda x: alpha*(L.T @ (L @ x)) + (W @ x)
    elif m == 3:
        Ly = L @ L @ L @ y
        H_lambda = lambda x: alpha*(L @ (L.T @ (L @ x))) + (W @ x)
    elif m == 4:
        Ly =L @ L @ L @ L @ y
        H_lambda = lambda x: alpha * (L @ (L @ (L.T @ (L @ x)))) + (W @ x)
    else:
        raise ValueError('Not implemented')
    H = LinearOperator((n, n), H_lambda)

    bias = cgmatrix(H, Ly,TOL=1e-6, MAXITER=10000, debug=debug)
    biasSq = np.trace(bias.T @ bias)
    H_bias = cgmatrix(H, bias,TOL=1e-6, MAXITER=10000, debug=debug)
    dbiasSq = - 2 * np.sum(bias * H_bias,axis=1)

    if sigma > 0:
        Wv = W @ v
        Q = cgmatrix(H, Wv,TOL=1e-12, MAXITER=10000, debug=debug)
        var = np.trace(Q.T @ Q)
        H_Q = cgmatrix(H, Q,TOL=1e-6, MAXITER=10000, debug=debug)
        dvar = np.squeeze(np.array((2 * np.sum((v-Q)*H_Q,axis=1))))
    else:
        var = 0
        dvar = np.zeros_like(dbiasSq)
    if beta > 0:
        cost = np.sum(w)
        dcost = beta
    else:
        cost = 0
        dcost = 0
    f = alpha**2 * biasSq + sigma**2 * var + beta * cost
    df = alpha**2 * dbiasSq + sigma**2 * dvar + dcost
    return f,df,biasSq,dbiasSq,var,dvar,cost,dcost, bias

def cgmatrix(A,B,TOL=1e-08,MAXITER=None,M=None,x0=None,callback=None,ATOL=None,debug=False):
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
        if debug:
            bsol = A @ x[:,i]
            residual[i] = norm(bsol - b)

    t1 = time.time()
    if debug:
        print("CG took {} s, exited with status {} and had residual {}".format(t1-t0, status, residual))
    return x


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


def OEDA(w,L,y,alpha,beta,sigma,lr,ns,idx_learned,LOG,xy,use_stochastic_approximate=True,safety_stop=2000,debug=False):
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
    if ns <= 0:
        return w
    if use_stochastic_approximate:
        v = np.sign(np.random.normal(0,1,(y.shape[0],1)))
    else:
        v = identity(L.shape[0]).tocsc() #TODO there might be a problem here, test that it works
    LOG.info('Iter  found      f         bias         var      time(s)')
    lr_base = lr
    nfound = 0
    i = 0
    lrw = lr_base
    dirw = np.zeros_like(w)
    while True:
        f,df,bias,dbias,var,dvar,cost,dcost,bias_pp = getOEDA(w,L,y,alpha,sigma,v,beta)
        ind = np.argmax(np.abs(df))
        bias_pp2 = np.sum(np.abs(bias_pp),axis=1)
        debug_circles(xy, df, idx_learned, y, dbias, bias_pp2,ind)
        # redo=False
        # for idx in idx_learned:
        #     if np.abs(df[idx]) > 0.1*np.abs(df[ind]):
        #         ite = 0
        #         while True:
        #             wtry = copy.deepcopy(w)
        #             wtry[idx] =  w[idx] - lrw * df[idx]
        #             if wtry[idx] <=0:
        #                 print("How did we get here?")
        #             ftry, _, _, _, _, _,_,_ = getOEDA(wtry, L, y, alpha, sigma, v,beta)
        #             if ftry <= f:
        #                 w[idx] = wtry[idx]
        #                 if ite == 0:
        #                     lrw *= 1.3
        #                 break
        #             else:
        #                 lrw *= 0.5
        #                 ite += 1
        #             LOG.info("{:3d}-{:2d}   {:3d}    {:3.2e}    {:3.2e}    {:3.2e} ".format(i,ite, nfound, f, ftry, lrw ))
        #         redo=True
        # if redo:
        #     LOG.info("{:3d}   {:3d}    {:3.2e}    {:3.2e}    {:3.2e} ".format(i, nfound, f, alpha ** 2 * bias,
        #                                                                                 sigma ** 2 * var))
        #     continue
        w[ind] = w[ind] - lr*df[ind]
        if ind not in idx_learned:
            lr = lr_base
            idx_learned.add(ind)
            nfound += 1
            if nfound >= ns:
                i += 1
                t1 = time.time()
                LOG.info(
                    "{:3d}   {:3d}    {:3.2e}    {:3.2e}    {:3.2e}     {:.1f}".format(i, nfound, f, alpha ** 2 * bias,
                                                                                       sigma ** 2 * var, t1 - t0))
                break
        else:
            lr = lr*1.3
        i += 1
        t1 = time.time()
        LOG.info("{:3d}   {:3d}    {:3.2e}    {:3.2e}    {:3.2e}     {:.1f}".format(i, nfound, f, alpha ** 2 * bias,
                                                                              sigma ** 2 * var, t1 - t0))
        if i>=safety_stop:
            break
    return w

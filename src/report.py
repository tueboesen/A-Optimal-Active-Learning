import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from scipy.special import softmax

from src.visualization import vizualize_circles


def analyse_probability_matrix(U,dataset,LOG,L,saveprefix=None,iter=None):
    '''
    Analyses a probability matrix U
    :param U: probability matrx (nsamples,nclasses)
    :param dataset:
    :param LOG:
    :param L:
    :return:
    '''
    #TODO this function should not require dataset, and is horribly static. It should be updated.
    Cpred = np.argmax(U,axis=1)
    nc = dataset.nc
    classes = np.asarray(range(nc))
    Ctrue = dataset.labels_true
    Cselect = [num for num in dataset.labels if num >= 0]
    nselect = np.bincount(Cselect,minlength=nc)
    A = np.zeros((nc,nc),dtype=int)
    labels_unique = np.unique(Ctrue)

    #Lets take a quick look at each class individually and see how we did:
    n = len(Ctrue)
    for i in range(nc):
        Ctrue_binary = - np.ones(n)
        idx = np.where(Ctrue == np.float32(i))[0]
        Ctrue_binary[idx] = 1
        ui = U[:,i]
        uip = copy.deepcopy(ui)
        uip[ui > 0] = 1
        uip[ui <= 0 ] = -1
        r = np.sum(np.abs(uip-Ctrue_binary)) / (2*n)
        LOG.info("Class {}, Error {:2.2f}% ".format(i,r*100))

    for i in classes: #pred class
        Ctrue_i = np.asarray(Ctrue)[Cpred == i]
        for j in classes: #true class
            A[i,j] = sum(Ctrue_i == j)

    with np.printoptions(formatter={'all':lambda x: "{:6d}".format(x)}):
        LOG.info("Labels selected:")
        LOG.info("Class       : {} {:>8s}".format(labels_unique,'total'))
        LOG.info("selected (#): {} {:8d}".format(nselect,np.sum(nselect)))
    with np.printoptions(formatter={'all':lambda x: "{:6.2f}".format(x)}):
        LOG.info("selected (%): {}".format(nselect/sum(nselect)*100))
    LOG.info(" ")
    LOG.info("Based on labels selected, the clustering predicted:")
    with np.printoptions(formatter={'all':lambda x: "{:6d}".format(x)}):
        LOG.info("Predicted \\ True {}  {:>8s}".format(labels_unique,'total'))
        LOG.info("------------------------------------------------------------------------------------------------")
        for i in labels_unique:
            LOG.info("        {}        {} {:8d}".format(i,A[i,:],np.sum(A[i,:])))
        LOG.info("------------------------------------------------------------------------------------------------")
        LOG.info("     {:>6s}      {} {:8d}".format('total',np.sum(A[:,:],axis=0),np.sum(A)))
        LOG.info(" ")
    Accuracy = sum(Cpred == Ctrue)/len(Ctrue)*100
    LOG.info("Accuracy = {}%".format(Accuracy))

    #Lets also probe the certainty of all points.
    plot_distribution_matrix(U)

    if dataset.name == 'circles':
        vizualize_circles(U,dataset, saveprefix=saveprefix,iter=iter)
    # #Lets look at L*U
    # Usoft = softmax(U,axis=1)
    # UL2 = np.zeros_like(U)
    # for i in range(U.shape[1]):
    #     UL2[:,i] = L @ Usoft[:,i]
    # plot_distribution_matrix(UL2)
    return Accuracy

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
    if save:
        fileloc = "{}/{}_{}.png".format(save, 'distribution',iter)
        fig.savefig(fileloc)
    plt.close(fig)
    return

def analyse_features(U,dataset,LOG,save=None,iter=None):
    '''
    Analyses the output from the neural network. Note that this is almost the same routine as analyse probability matrix, and should be merged with that at some point.
    Main difference is that it plots a histogram of the label certainty of the right and wrong labels
    :param U:
    :param dataset:
    :param LOG:
    :param save: location to save the various figures to
    :param iter: unique identifier to save with the figure, to prevent it overwriting previous iterations figures.
    :return:
    '''
    if isinstance(U, torch.Tensor):
        U = U.numpy()
    nc = dataset.nc
    classes = np.asarray(range(nc))
    Cpred = np.argmax(U,axis=1)
    Ctrue = dataset.labels_true
    Cselect = [num for num in dataset.labels if num >= 0]
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
    Accuracy = sum(Cpred == Ctrue)/len(Ctrue)*100
    LOG.info("Accuracy = {}%".format(Accuracy))

    #Lets also probe the certainty of all points.
    # plot_distribution_matrix(U,save,iter)

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
    if save:
        fileloc = "{}_{}_{}.png".format(save, 'histogram',iter)
        fig.savefig(fileloc)
        fileloc = "{}_{}_{}.txt".format(save, 'histogram', iter)
        with open(fileloc, 'w') as f:
            f.write("U : {} \n".format(U.tolist()))
            f.write("idx_right : {} \n".format(idx_right.tolist()))
            f.write("idx_wrong : {} \n".format(idx_wrong.tolist()))

    plt.close(fig)
    return Accuracy
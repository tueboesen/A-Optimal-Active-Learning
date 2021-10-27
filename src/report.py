import numpy as np


def analyse_probability_matrix(y_pred,y_true,idx_labels,c):
    """
    Analyses a probability matrix U
    :param U: probability matrx (nsamples,nclasses)
    :param dataset:
    :param LOG:
    :param L:
    :return:
    """
    n,nc = y_true.shape
    c_pred = np.argmax(y_pred,axis=1)
    c_true = np.argmax(y_true,axis=1)
    classes = np.asarray(range(nc))

    nlabels_selected_pr_class = np.bincount(c_pred[idx_labels],minlength=nc)
    A = np.zeros((nc,nc),dtype=int)
    for i in classes: #pred class
        c_true_i = c_true[c_pred == i]
        for j in classes: #true class
            A[i,j] = sum(c_true_i == j)

    with np.printoptions(formatter={'all':lambda x: "{:6d}".format(x)}):
        c.LOG.info("Labels selected:")
        c.LOG.info("Class       : {} {:>8s}".format(classes,'total'))
        c.LOG.info("selected (#): {} {:8d}".format(nlabels_selected_pr_class,np.sum(nlabels_selected_pr_class)))
    with np.printoptions(formatter={'all':lambda x: "{:6.2f}".format(x)}):
        c.LOG.info("selected (%): {}".format(nlabels_selected_pr_class/sum(nlabels_selected_pr_class)*100))
    c.LOG.info(" ")
    c.LOG.info("Based on labels selected, the clustering predicted:")
    with np.printoptions(formatter={'all':lambda x: "{:6d}".format(x)}):
        c.LOG.info("Predicted \\ True {}  {:>8s}".format(classes,'total'))
        c.LOG.info("------------------------------------------------------------------------------------------------")
        for i in classes:
            c.LOG.info("        {}        {} {:8d}".format(i,A[i,:],np.sum(A[i,:])))
        c.LOG.info("------------------------------------------------------------------------------------------------")
        c.LOG.info("     {:>6s}      {} {:8d}".format('total',np.sum(A[:,:],axis=0),np.sum(A)))
        c.LOG.info(" ")
    Accuracy = sum(c_pred == c_true)/len(c_true)*100
    c.LOG.info("Accuracy = {}%".format(Accuracy))
    return Accuracy


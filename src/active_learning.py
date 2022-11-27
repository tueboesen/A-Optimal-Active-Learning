import inspect

import numpy as np
import scipy
from scipy.special import softmax

from src.Clustering import SSL_clustering
from src.active_learning_adaptive import run_active_learning_adaptive
from src.active_learning_ms import run_active_learning_ms
from src.dataloader import set_labels, subset_of_dataset
from src.optimization import train
from src.passive_learning import run_passive_learning
from src.report import analyse_probability_matrix
from src.results import create_result_subcategories, update_result

sparse_matrix = scipy.sparse._csr.csr_matrix
array = np.ndarray


def initial_labeling(mode: str, nlabels: int, dataloader):
    """
    Determines how the initial labelling of datapoints is done.
    Current options are:
    balanced: which selects equal amounts of labels from each class
    random: which randomly draws the labels among all samples
    bayesian: IS NOT CURRENTLY IMPLEMENTED, but the idea would be to select initial labels based on the baysian approach of the graph Laplacian, as mentioned in the paper.
    """
    if mode == 'balanced':
        dataloader, idx = set_labels(nlabels, dataloader, class_balance=True)
    elif mode == 'random':
        dataloader, idx = set_labels(nlabels, dataloader, class_balance=False)
    elif mode == 'bayesian':
        raise NotImplementedError("{} has not been implemented in function {}".format(mode, inspect.currentframe().f_code.co_name))
    else:
        raise NotImplementedError("{} has not been implemented in function {}".format(mode, inspect.currentframe().f_code.co_name))
    return dataloader, idx


def run_active_learning(mode: str, y: array, idx_labels: list, L: sparse_matrix, c, dl_train=None, dl_test=None, net=None, optimizer=None) -> (list, dict):
    """
    Main driver for different types of active learning and passive learning
    :param mode: type of learning
    :param y: true label pseudo-probabilities (before softmax)
    :param idx_labels: indices of known labels
    :param L: graph laplacian
    :param c: configuration parameters
    :param dl_train: dataloader for training
    :param dl_test: dataloader for testing
    :param net: neural network
    :param optimizer: optimizer
    :return: (indices of known labels, results)
    """

    n, nc = y.shape
    result = create_result_subcategories()
    dl_org = dl_train
    w = np.zeros(n)
    w[idx_labels] = c.AL_w0
    idx_pseudo = []
    label_pseudo = []
    cluster_acc = 0
    for i in range(c.AL_iterations):
        if i != 0:
            if mode == 'active_learning_adaptive':
                idx_labels, w = run_active_learning_adaptive(idx_labels, L, w, y, c)
            elif mode == 'passive_learning':
                idx_labels = run_passive_learning(idx_labels, y, c.AL_nlabels_pr_class, class_balance=False)
                w[idx_labels] = c.AL_w0
            elif mode == 'passive_learning_balanced':
                idx_labels = run_passive_learning(idx_labels, y, c.AL_nlabels_pr_class, class_balance=True)
                w[idx_labels] = c.AL_w0
            elif mode == 'active_learning_ms':
                delta0 = 0.005
                delta1 = 0.001
                delta = (delta1 - delta0) / c.AL_iterations * i + delta0
                idx_labels, idx_pseudo, label_pseudo = run_active_learning_ms(dl_org, idx_labels, c, net, c.AL_nlabels_pr_class * nc, delta)
            else:
                raise NotImplementedError("{} has not been implemented in function {}".format(mode, inspect.currentframe().f_code.co_name))
        if mode == 'active_learning_ms':
            dl_train = subset_of_dataset(dl_org.dataset, idx_labels, idx_pseudo, label_pseudo, dl_org.batch_size)
            use_probabilities = False
        else:
            if L is None:
                use_probabilities = False
            else:
                y_pred = SSL_clustering(c.AL_alpha, L, y, w, c.L_eta)
                cluster_acc = analyse_probability_matrix(y_pred, y, idx_labels, c)
                y_prob = softmax(y_pred, axis=1)
                dl_train = set_labels(y_prob, dl_train)
                use_probabilities = True
        if c.SL_at_each_step:
            if i == 0:
                epochs = c.SL_epochs_init
            else:
                epochs = c.SL_epochs
            net, tmp = train(net, optimizer, dl_train, c.SL_loss_type, c.LOG, device=c.device,
                             dataloader_test=dl_test,
                             epochs=epochs, use_probabilities=use_probabilities)
            c.LOG.info('AL: {:4d}  Accuracy: {:3.2f}%   idx_label: {:4d}  idx_pseudo: {:4d}'.format(i, tmp, len(idx_labels), len(idx_pseudo)))
            result['test_acc'].append(tmp)
        update_result(result, idx_labels, cluster_acc)
    return idx_labels, result

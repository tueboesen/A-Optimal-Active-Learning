import inspect

import numpy as np
from scipy.sparse import identity

from src.Clustering import SSL_clustering
from src.active_learning_adaptive import run_active_learning_adaptive
from src.dataloader import set_labels
from src.optimization import train
from src.passive_learning import run_passive_learning
from src.report import analyse_probability_matrix
from src.results import setup_result, update_result


def initial_labeling(mode,nlabels,dataloader):
    if mode == 'balanced':
        dataloader,idx = set_labels(nlabels, dataloader, class_balance=True)
    elif mode == 'random':
        dataloader,idx = set_labels(nlabels, dataloader, class_balance=False)
    elif mode == 'bayesian':
        raise NotImplementedError("{} has not been implemented in function {}".format(mode,inspect.currentframe().f_code.co_name))
    else:
        raise NotImplementedError("{} has not been implemented in function {}".format(mode,inspect.currentframe().f_code.co_name))
    return dataloader,idx

def run_active_learning(mode,y,idx_labels,L,c,dl_train=None,dl_test=None,net=None,optimizer=None,loss_fnc=None):
    n,nc = y.shape
    result = setup_result()
    w = np.zeros(n)
    w[idx_labels] = c.AL_w0
    L = L + c.L_tau * identity(L.shape[0])
    for i in range(c.AL_iterations):
        if mode == 'active_learning_adaptive':
            idx_labels,w = run_active_learning_adaptive(idx_labels,L,w,y,c)
        elif mode == 'passive_learning':
            idx_labels = run_passive_learning(idx_labels, y, c.AL_nlabels_pr_class, class_balance=False)
            w[idx_labels] = c.AL_w0
        elif mode == 'passive_learning_balanced':
            idx_labels = run_passive_learning(idx_labels, y, c.AL_nlabels_pr_class, class_balance=True)
            w[idx_labels] = c.AL_w0
        else:
            raise NotImplementedError("{} has not been implemented in function {}".format(mode, inspect.currentframe().f_code.co_name))
        y_pred = SSL_clustering(c.AL_alpha, L, y, w,c.L_eta)
        cluster_acc = analyse_probability_matrix(y_pred, y, idx_labels, c)
        if c.SL_at_each_step:
            dl_train = set_labels(y_pred, dl_train)
            if i == 0:
                epochs = c.SL_epochs_init
            else:
                epochs = c.SL_epochs
            net, tmp = train(net, optimizer, dl_train, loss_fnc, c.LOG, device=c.device,
                                            dataloader_test=dl_test,
                                            epochs=epochs, use_probabilities=True)
            result['test_acc'].append(tmp)
        update_result(result, idx_labels, cluster_acc)
    return idx_labels,y_pred,result


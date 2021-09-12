import inspect
import torch.nn.functional as F

import numpy as np
from scipy.sparse import identity
from scipy.special import softmax

from src.Clustering import SSL_clustering
from src.Laplacian import compute_laplacian
from src.active_learning_adaptive import run_active_learning_adaptive
from src.active_learning_ms import run_active_learning_ms
from src.dataloader import set_labels, subset_of_dataset
from src.optimization import train, eval_net
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
    dl_org = dl_train
    delta0 = 0.005
    delta1 = 0.001
    w = np.zeros(n)
    w[idx_labels] = c.AL_w0
    idx_pseudo = []
    label_pseudo = []
    cluster_acc = 0
    for i in range(c.AL_iterations):
        if i != 0:
            if mode == 'active_learning_adaptive':
                features = eval_net(net, dl_train.dataset, device=c.device)
                L, A = compute_laplacian(features, metric=c.L_metric, knn=c.L_knn, union=True)
                L = L + c.L_tau * identity(L.shape[0])
                idx_labels,w = run_active_learning_adaptive(idx_labels,L,w,y,c)
            elif mode == 'passive_learning':
                idx_labels = run_passive_learning(idx_labels, y, c.AL_nlabels_pr_class, class_balance=False)
                w[idx_labels] = c.AL_w0
            elif mode == 'passive_learning_balanced':
                idx_labels = run_passive_learning(idx_labels, y, c.AL_nlabels_pr_class, class_balance=True)
                w[idx_labels] = c.AL_w0
            elif mode == 'active_learning_ms':
                delta = (delta1-delta0)/c.AL_iterations*i + delta0
                idx_labels,idx_pseudo,label_pseudo = run_active_learning_ms(dl_org,idx_labels,y,c,net,c.AL_nlabels_pr_class*nc,delta)
            else:
                raise NotImplementedError("{} has not been implemented in function {}".format(mode, inspect.currentframe().f_code.co_name))
        if mode == 'active_learning_ms':
            dl_train = subset_of_dataset(dl_org.dataset,idx_labels,idx_pseudo,label_pseudo,dl_org.batch_size)
            use_probabilities = False
        else:
            if L is None:
                use_probabilities = False
            else:
                y_pred = SSL_clustering(c.AL_alpha, L, y, w,c.L_eta)
                cluster_acc = analyse_probability_matrix(y_pred, y, idx_labels, c)
                y_prob = softmax(y_pred*5, axis=1)
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
            result['test_acc'].append(tmp)
        update_result(result, idx_labels, cluster_acc)
        c.LOG.info('AL: {:4d}  Accuracy: {:3.2f}%   idx_label: {:4d}  idx_pseudo: {:4d}'.format(i, tmp, len(idx_labels), len(idx_pseudo)))
    return idx_labels,result


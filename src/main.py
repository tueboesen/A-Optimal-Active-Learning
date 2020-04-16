import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import identity

from src import log
from src.Clustering import SSL_clustering
from src.IO import load_autoencoder, save_state
from src.Laplacian import compute_laplacian
from src.active_learning import OEDA
from src.dataloader import Load_MNIST, set_labels
from src.losses import select_loss_fnc
from src.networks import ResidualBlock, ResNet
from src.networks_ae import select_network
from src.optimization import train_AE, eval_net, train
from src.report import analyse_probability_matrix, analyse_features
from src.utils import determine_network_param, fix_seed


def main(c):
    #Initialize things
    fix_seed(c['seed']) #Set a seed, so we make reproducible results.
    result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basename'],
        date=datetime.now(),
    )
    os.makedirs(result_dir)
    logfile_loc = "{}/{}.log".format(result_dir, 'output')
    LOG = log.setup_custom_logger('runner',logfile_loc)
    LOG.info('---------Listing all parameters-------')
    for key, value in c.items():
        LOG.info("{:30s} : {}".format(key, value))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load Dataset
    MNIST_train, MNIST_test = Load_MNIST(batch_size=c['batch_size'],nsamples=c['nsamples'],device=device,order_data=c['order_dataset'])

    if c['use_AE']: #Do we use an Autoencoder?
        if c['load_AE']:
            LOG.info("Loading autoencoder from file: {}".format(c['load_AE']))
            netAE,features = load_autoencoder(c['load_AE'],LOG,c['nsamples'],c['decode_dim'],c['network_AE'])
        else:
            LOG.info("Setting up and training an autoencoder...")
            netAE = select_network(c['network_AE'],c['decode_dim'])
            LOG.info('Number of parameters in autoencoder: {}'.format(determine_network_param(netAE)))
            optimizerAE = optim.Adam(list(netAE.parameters()), lr=c['lr_AE'],weight_decay=1e-5)
            loss_fnc_ae = nn.MSELoss(reduction='sum') #Loss function for autoencoder should always be MSE
            netAE,features = train_AE(netAE,optimizerAE,MNIST_train,loss_fnc_ae,LOG,device=device,epochs=c['epochs_AE'],save="{}/{}.png".format(result_dir, 'autoencoder'))
            state = {'features': features,
                     'epochs_AE': c['epochs_AE'],
                     'nsamples': c['nsamples'],
                     'decode_dim': c['decode_dim'],
                     'lr_AE': c['lr_AE'],
                     'npar_AE': determine_network_param(netAE),
                     'result_dir': result_dir,
                     'autoencoder_state': netAE.state_dict()}
            save_state(state, "{}/{}.pt".format(result_dir, 'autoencoder')) #We save the trained autoencoder and the encoded space, as well as some characteristica of the network and samples used to train it.
    else: # We just use the images as features directly
        features = MNIST_train.dataset.imgs

    # Calculate Laplacian
    L,A = compute_laplacian(features, metric='l2', knn=9, union=True)

    # Setup Network Geometry
    net_args = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2]
    }

    # Select loss function
    loss_fnc = select_loss_fnc(c['loss_type'],c['use_label_probabilities'])

    if c['use_adaptive_active_learning']:
        LOG.info('Starting adaptive active learning...')
        netAL = ResNet(**net_args)
        LOG.info('Number of parameters: {}'.format(determine_network_param(netAL)))
        optimizerAL = optim.Adam(list(netAL.parameters()), lr=c['lr'])
        nc = MNIST_train.dataset.nc

        MNIST_train = set_labels(c['nlabels'], MNIST_train, class_balance=True)  # Lets start by having all classes represented by some labels
        yobs = MNIST_train.dataset.plabels.numpy()
        idxs = np.nonzero(yobs[:,0])[0]
        w = np.zeros(L.shape[0])
        w[idxs] = 1
        L = L + 1e-3*identity(L.shape[0])
        idx_learned = set(idxs)
        for i in range(c['epochs_AL']):
            if c['use_1_vs_all']:
                yi = np.zeros((yobs.shape[0],1))
                for j in range(nc):
                    yi[:,0] = np.sign(yobs[:,j])
                    w = OEDA(w,L,yi,c['alpha'],c['sigma'],c['lr_AL'],c['nlabels_pr_epoch_pr_class'],idx_learned)
                    idx = np.nonzero(w)[0]
                    idx_learned.update(idx)
            else:
                w = OEDA(w,L,y,c['alpha'],c['sigma'],c['lr_AL'],c['nlabels_pr_epoch_pr_class'],idx_learned)
                idx = np.nonzero(w)[0]
                idx_learned.update(idx)
            MNIST_train = set_labels(list(idx_learned), MNIST_train)
            yobs = MNIST_train.dataset.plabels
            y = SSL_clustering(c['alpha'], L, yobs,balance_weights=True)
            analyse_probability_matrix(y, MNIST_train.dataset,LOG,L)
            y[list(idx_learned)] = MNIST_train.dataset.plabels[list(idx_learned)]   # We update the known plabels to their true value
            #train a network on this data
            MNIST_train = set_labels(y, MNIST_train)
            netAL = train(netAL, optimizerAL, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test,
                  epochs=c['epochs_SL'], use_probabilities=c['use_label_probabilities'])
            features_netAL = eval_net(netAL, MNIST_train.dataset, device=device) #we should probably have the option of combining these with the previous features.
            analyse_features(features_netAL, MNIST_train.dataset, LOG, save=result_dir,iter=i)
            L, A = compute_laplacian(features_netAL, metric='l2', knn=9, union=True)
            L = L + 1e-3 * identity(L.shape[0])

        LOG.info("Selecting {:5d} labels".format(len(idx)))
        if c['nlabels'] <= 0:
            c['nlabels'] = len(idx) #We set the number of labels to match the ones from active learning
        netAL = train(netAL, optimizerAL, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test, epochs=c['epochs_SL'], use_probabilities=c['use_label_probabilities'])
        LOG.info('DONE WITH ADAPTIVE LEARNING!')
        # Active learning step, designate which labels to use
    #
    #
    # if c['use_active_learning']:
    #     LOG.info('Starting active learning')
    #     netAL = ResNet(**net_args)
    #     npar = sum(p.numel() for p in netAL.parameters() if p.requires_grad)
    #     LOG.info('Number of parameters: {}'.format(npar))
    #     optimizerAL = optim.Adam(list(netAL.parameters()), lr=c['lr'])
    #
    #     w = np.zeros(L.shape[0])
    #     L = L + 1e-3*identity(L.shape[0])
    #     w = OEDB(w,L,c['alpha'],c['beta'],c['lr_OED'],maxIter=c['epochs_AL'])
    #     idx = list(np.nonzero(w)[0])
    #     LOG.info("Selecting {:5d} labels".format(len(idx)))
    #     if c['nlabels'] <= 0:
    #         c['nlabels'] = len(idx) #We set the number of labels to match the ones from active learning
    #     MNIST_train = set_labels(idx, MNIST_train)
    #     #Now check these labels on the clustering
    #     Uobs = create_uobs_from_dataset(MNIST_train.dataset)
    #     U = SSL_clustering(c['alpha'], L, Uobs)
    #     analyse_probability_matrix(U, MNIST_train.dataset,LOG)
    #     netAL = train(netAL, optimizerAL, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test, epochs=c['epochs_SL'])
    #
    # if c['use_passive_learning']:
    #     LOG.info('Starting passive learning')
    #     netPL = ResNet(**net_args)
    #     npar = sum(p.numel() for p in netPL.parameters() if p.requires_grad)
    #     LOG.info('Number of parameters: {}'.format(npar))
    #     optimizerPL = optim.Adam(list(netPL.parameters()), lr=c['lr'])
    #
    #
    #     LOG.info("Selecting {:5d} labels".format(len(idx)))
    #     MNIST_train = set_labels(len(idx), MNIST_train) #We randomly select the labels we preserve.
    #     Uobs = create_uobs_from_dataset(MNIST_train.dataset)
    #     U = SSL_clustering(c['alpha'], L, Uobs)
    #     analyse_probability_matrix(U, MNIST_train.dataset,LOG)
    #     netPL = train(netPL, optimizerPL, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test, epochs=c['epochs_SL'])
    #
    # if c['use_class_balanced_learning']:
    #     LOG.info('Starting class balanced learning')
    #     netCB = ResNet(**net_args)
    #     npar = sum(p.numel() for p in netCB.parameters() if p.requires_grad)
    #     LOG.info('Number of parameters: {}'.format(npar))
    #     optimizerCB = optim.Adam(list(netCB.parameters()), lr=c['lr'])
    #
    #     #We randomly select the labels we preserve.
    #     MNIST_train = set_labels(len(idx), MNIST_train, class_balance=True)
    #     LOG.info("Selecting {:5d} labels".format(len(idx)))
    #     Uobs = create_uobs_from_dataset(MNIST_train.dataset)
    #     U = SSL_clustering(c['alpha'], L, Uobs)
    #     analyse_probability_matrix(U, MNIST_train.dataset,LOG)
    #     netCB = train(netCB, optimizerCB, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test, epochs=c['epochs_SL'])
    #
    #
    #
    # if c['use_active_learning_tue']:
    #     LOG.info('Starting active learning Tue approach')
    #     assert c['use_label_probabilities'], "Error, label probabilities must be enabled."
    #     netAL = ResNet(**net_args)
    #     npar = sum(p.numel() for p in netAL.parameters() if p.requires_grad)
    #     LOG.info('Number of parameters: {}'.format(npar))
    #     optimizerAL = optim.Adam(list(netAL.parameters()), lr=c['lr'])
    #     nc = MNIST_train.dataset.nc
    #
    #     #We have a new hyper parameter
    #     theta = 0.2
    #     knn = 300
    #
    #     #First we need to calculate the Laplacian with many more neighbours and a cutoff
    #     L, A = compute_laplacian(features, metric='l2', knn=knn, union=True, cutoff=True)
    #
    #     #Next we then
    #     L2 = L.copy()
    #     L = L + 1e-3 * identity(L.shape[0])
    #     Lmax = np.max(L.diagonal())
    #     Y = MNIST_train.dataset.labels_true.numpy()
    #     n = len(Y)
    #     ns = 50
    #     wy = np.ones(n)
    #
    #     #We start without any known examples
    #     # y = np.zeros((n,nc))
    #     # idx_learned = set()
    #
    #     #We start with a few known examples
    #     MNIST_train = set_labels(nc*2, MNIST_train, class_balance=True)
    #     y = create_uobs_from_dataset(MNIST_train.dataset)
    #     idx_t = np.nonzero(y[:,0])[0]
    #     L2, y,idx_learned = update_laplacian(L2, y, Lmax, idx_t,Y)
    #
    #     for epoch in range(c['epochs_AL']):
    #         for nsi in range(ns):
    #             tmp = L2.diagonal() * wy
    #             idx = np.argmax(tmp)
    #             L2, y, idx_learned = update_laplacian(L2, y, Lmax, idx, Y,idxs_learned=idx_learned)
    #             conn = np.zeros(n)
    #             for i in range(n):
    #                 conn[i] = L2[i,:].count_nonzero()
    #             wy = (1 - (1 + theta) * np.max(y,axis=1) + theta * np.sum(y,axis=1))**(conn/2) #
    #         MNIST_train = set_labels(list(idx_learned), MNIST_train)
    #         Uobs = create_uobs_from_dataset(MNIST_train.dataset)
    #         LOG.info('Balanced clustering')
    #         y2 = SSL_clustering(c['alpha'], L, Uobs,balance_weights=True)
    #         analyse_probability_matrix(y2, MNIST_train.dataset,LOG,L)
    #         # train on this data
    #         # py = convert_pseudo_to_prob(y2,use_softmax=True)
    #         # weights = find_weights_from_labels(py,list(idx_learned))
    #         # MNIST_train = set_labels(y2, MNIST_train)
    #         # netAL = train(netAL, optimizerAL, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test,
    #         #       epochs=c['epochs_SL'],weights=weights)
    #         # features_netAL = eval_net(netAL, MNIST_train, device=device) #we should probably have the option of combining these with the previous features.
    #         # analyse_features(features_netAL, MNIST_train.dataset, LOG)
    #         # L, A = compute_laplacian(features_netAL, metric='l2', knn=knn, union=True)
    #         # L = L + 1e-3 * identity(L.shape[0])
    #         #We now need to recreate L2 again, yikes.
    #

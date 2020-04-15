import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from IO import save_state, load_autoencoder
from active_learning import OEDB, OEDA
from src.dataloader import Load_MNIST, set_labels
from networks import ResNet, ResidualBlock, Classifier
from networks_ae import autoencoder, AutoEncoder_v3, autoencoder_linear
from optimization import train, train_AE, eval_net, run_AE
from utils import fix_seed
import log
from scipy.sparse import identity

def main_classifier(c):
    fix_seed(c['seed'])
    result_dir = "{root}/{runner_name}/{metric}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basename'],
        metric=c['metric_name'],
        date=datetime.now(),
    )
    os.makedirs(result_dir)
    logfile_loc = "{}/{}.log".format(result_dir, 'output')
    LOG = log.setup_custom_logger('runner',logfile_loc)
    LOG.info('---------Listing all parameters-------')
    for key, value in c.items():
        LOG.info("{:30s} : {}".format(key, value))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    MNIST_train, MNIST_test = Load_MNIST(batch_size=c['batch_size'],nsamples=c['nsamples'],device=device,order_data=c['order_dataset'],use_label_probabilities=c['use_label_probabilities'])

    # Setup Network Geometry
    net_args = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2]
    }
    loss_fnc = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')

    if c['use_AE']:
        if c['load_AE']:
            LOG.info("Loading autoencoder")
            netAE,features = load_autoencoder(c['load_AE'],c['nsamples'],c['decode_dim'])
        else:
            LOG.info("Running Autoencoder")
            netAE = autoencoder()
            # netAE = autoencoder_linear(decode_dim=c['decode_dim'])
            npar = sum(p.numel() for p in netAE.parameters() if p.requires_grad)
            LOG.info('Number of parameters: {}'.format(npar))
            optimizerAE = optim.Adam(list(netAE.parameters()), lr=c['lr_AE'],weight_decay=1e-5)
            loss_fnc_ae = nn.MSELoss(reduction='sum')
            netAE,features = train_AE(netAE,optimizerAE,MNIST_train,loss_fnc_ae,LOG,device=device,epochs=c['epochs_AE'],save="{}/{}.png".format(result_dir, 'autoencoder'))
            state = {'features': features,
                     'epochs_AE': c['epochs_AE'],
                     'nsamples': c['nsamples'],
                     'decode_dim': c['decode_dim'],
                     'lr_AE': c['lr_AE'],
                     'npar_AE': npar,
                     'result_dir': result_dir,
                     'autoencoder_state': netAE.state_dict()}
            save_state(state, "{}/{}.pt".format(result_dir, 'autoencoder'))
    else:
        features = MNIST_train.dataset.imgs
    MNIST_train.dataset.imgs = features

    features_test = run_AE(netAE, MNIST_test, device=device)
    MNIST_test.dataset.imgs = features_test

    netAL = Classifier(c['decode_dim'])
    npar = sum(p.numel() for p in netAL.parameters() if p.requires_grad)
    LOG.info('Number of parameters: {}'.format(npar))
    optimizerAL = optim.Adam(list(netAL.parameters()), lr=c['lr'])
    netAL = train(netAL, optimizerAL, MNIST_train, loss_fnc, LOG, device=device, dataloader_validate=MNIST_test,
                  epochs=c['epochs_SL'])


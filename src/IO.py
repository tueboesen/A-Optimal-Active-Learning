import os

import torch

from src.networks_ae import select_network_ae
from src.optimization import run_AE


def save_state(state, file):
    '''
    Saves the state of a network and some additional information needed for load_autoencoder
    :param state:
    :param file:
    :return:
    '''
    torch.save(state, file)


def load_autoencoder(fileloc,LOG,nsamples,decode_dim,network_type,dataloader,device):
    '''
    Loads an autoencoder and its corresponding encoded space on the dataset it was trained on
    :param fileloc:
    :param nsamples:
    :param decode_dim:
    :param network_type:
    :param dataloader:
    :param device:
    :return:
    '''
    assert os.path.isfile(fileloc), "Error! no autoencoder found at '{}'".format(fileloc)
    data = torch.load(fileloc)
    if data['decode_dim'] != decode_dim:
        LOG.warning("The autoencoder was built with a decode dim of {}, but you have selected {}.".format(data['decode_dim'],decode_dim))
    netAE = select_network_ae(network_type, decode_dim=decode_dim)
    netAE.load_state_dict(data['autoencoder_state'])
    if data['nsamples'] != nsamples:
        LOG.info("The autoencoder was originally trained on a dataset with {} samples, but your dataset has {} samples. Rebuilding features...".format(data['nsamples'],nsamples))
        features = run_AE(netAE, dataloader, device=device)
    else:
        features = data['features']
    return netAE, features

def load_labels(file):
    with open(file, 'r') as f:
        idx = f.read()
    idx_known = eval(idx)
    idx = idx_known[0][-1]
    return idx
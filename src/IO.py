import os

import torch

from src.networks_ae import select_network


def save_state(state, file):
    '''
    Saves the state of a network and some additional information needed for load_autoencoder
    :param state:
    :param file:
    :return:
    '''
    torch.save(state, file)


def load_autoencoder(fileloc,LOG,nsamples,decode_dim,network_type):
    '''
    Loads an autoencoder and its corresponding encoded space on the dataset it was trained on
    :param fileloc:
    :param nsamples:
    :param decode_dim:
    :param network_type:
    :return:
    '''
    assert os.path.isfile(fileloc), "Error! no autoencoder found at '{}'".format(fileloc)
    data = torch.load(fileloc)
    if data['decode_dim'] != decode_dim:
        LOG.warning("The autoencoder was built with a decode dim of {}, but you have selected {}.".format(data['decode_dim'],decode_dim))
    if data['nsamples'] != nsamples:
        LOG.warning("The autoencoder was originally trained on a dataset with {} samples, but your dataset has {} samples.".format(data['nsamples'],nsamples))
    netAE = select_network(network_type, decode_dim=decode_dim)
    netAE.load_state_dict(data['autoencoder_state'])
    features = data['features']
    return netAE, features


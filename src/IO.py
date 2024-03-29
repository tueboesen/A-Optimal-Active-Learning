import os

import torch

from src.networks_ae import select_network
from src.optimization import run_AE


def save_state(state, file):
    """
    Saves the state of a network and some additional information needed for load_autoencoder
    :param state:
    :param file:
    :return:
    """
    torch.save(state, file)


def load_autoencoder(fileloc, LOG, nsamples, decode_dim, network_type, dataloader, device):
    """
    Loads an autoencoder and its corresponding encoded space on the dataset it was trained on

    :param fileloc: file location on disk
    :param LOG: a logger handler
    :param nsamples: number of samples it was trained on.
    :param decode_dim: decoding dimensions used in autoencoder
    :param network_type: network type
    :param dataloader: dataloader to run autoencoder on
    :param device: device to run autoencoder on
    :return:
    """
    assert os.path.isfile(fileloc), "Error! no autoencoder found at '{}'".format(fileloc)
    data = torch.load(fileloc)
    if data['decode_dim'] != decode_dim:
        LOG.warning("The autoencoder was built with a decode dim of {}, but you have selected {}.".format(data['decode_dim'], decode_dim))
    netAE = select_network(network_type, decode_dim=decode_dim)
    netAE.load_state_dict(data['autoencoder_state'])
    if data['nsamples'] != nsamples:
        LOG.info("The autoencoder was originally trained on a dataset with {} samples, but your dataset has {} samples. Rebuilding features...".format(data['nsamples'], nsamples))
        features = run_AE(netAE, dataloader, device=device)
    else:
        features = data['features']
    return netAE, features

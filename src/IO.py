import torch
import os

from networks_ae import autoencoder_linear, select_network


def save_state(state, file):
    torch.save(state, file)


def load_autoencoder(fileloc,nsamples,decode_dim,network_type):
    assert os.path.isfile(fileloc), "=> no autoencoder found at '{}'".format(fileloc)
    data = torch.load(fileloc)
    assert data['nsamples'] == nsamples, "Error! The autoencoder was originally trained with a different number of samples."
    # assert data['decode_dim'] == decode_dim, "Error! The autoencoder was built with a decode dim of {}, but you have selected {}. Please change decode_dim or load another autoencoder.".format(data['decode_dim'],decode_dim)

    netAE = select_network(network_type, decode_dim=decode_dim)
    netAE.load_state_dict(data['autoencoder_state'])
    features = data['features']
    return netAE, features


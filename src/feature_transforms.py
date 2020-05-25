import torch.nn as nn
import torch.optim as optim

from src.IO import load_autoencoder, save_state
from src.networks_ae import select_network
from src.optimization import train_AE
from src.utils import determine_network_param


def select_feature_transform(dataloader,c):
    if c.feature_transform == 'autoencoder':
        if c.FT_load:
            c.LOG.info("Loading autoencoder from file: {}".format(c.FT_load))
            netAE, features = load_autoencoder(c.FT_load, c.LOG, c.nsamples, c.FT_decode_dim, c.FT_network,
                                               dataloader, c.device)
            c.LOG.info("Autoencoder loaded.")
        else:
            c.LOG.info("Setting up and training an autoencoder...")
            netAE = select_network(c.FT_network, c.FT_decode_dim)
            c.LOG.info('Number of parameters in autoencoder: {}'.format(determine_network_param(netAE)))
            lr = 1e-3
            optimizerAE = optim.Adam(list(netAE.parameters()), lr=lr, weight_decay=1e-5)
            loss_fnc_ae = nn.MSELoss(reduction='sum')  # Loss function for autoencoder should always be MSE
            netAE, features = train_AE(netAE, optimizerAE, dataloader, loss_fnc_ae, c.LOG, device=c.device,
                                       epochs=c.FT_epochs, save="{}/{}.png".format(c.result_dir, 'autoencoder'))
            state = {'features': features,
                     'epochs_AE': c.FT_epochs,
                     'nsamples': c.nsamples,
                     'decode_dim': c.FT_decode_dim,
                     'lr_AE': lr,
                     'npar_AE': determine_network_param(netAE),
                     'result_dir': c.result_dir,
                     'autoencoder_state': netAE.state_dict()}
            save_state(state, "{}/{}.pt".format(c.result_dir,
                                                'autoencoder'))  # We save the trained autoencoder and the encoded space, as well as some characteristica of the network and samples used to train it.
    else:
        raise NotImplementedError("Selected feature_transform: {}, has not been implemented yet.".format(c.feature_transform))
    return features


# if c['use_AE']:  # Do we use an Autoencoder?
#     if c['load_AE']:
#         LOG.info("Loading autoencoder from file: {}".format(c['load_AE']))
#         netAE, features = load_autoencoder(c['load_AE'], LOG, c['nsamples'], c['decode_dim'], c['network_AE'],
#                                            MNIST_train, device)
#         LOG.info("Autoencoder loaded.")
#     else:
#         LOG.info("Setting up and training an autoencoder...")
#         netAE = select_network(c['network_AE'], c['decode_dim'])
#         LOG.info('Number of parameters in autoencoder: {}'.format(determine_network_param(netAE)))
#         optimizerAE = optim.Adam(list(netAE.parameters()), lr=c['lr_AE'], weight_decay=1e-5)
#         loss_fnc_ae = nn.MSELoss(reduction='sum')  # Loss function for autoencoder should always be MSE
#         netAE, features = train_AE(netAE, optimizerAE, MNIST_train, loss_fnc_ae, LOG, device=device,
#                                    epochs=c['epochs_AE'], save="{}/{}.png".format(c['result_dir'], 'autoencoder'))
#         state = {'features': features,
#                  'epochs_AE': c['epochs_AE'],
#                  'nsamples': c['nsamples'],
#                  'decode_dim': c['decode_dim'],
#                  'lr_AE': c['lr_AE'],
#                  'npar_AE': determine_network_param(netAE),
#                  'result_dir': c['result_dir'],
#                  'autoencoder_state': netAE.state_dict()}
#         save_state(state, "{}/{}.pt".format(c['result_dir'],
#                                             'autoencoder'))  # We save the trained autoencoder and the encoded space, as well as some characteristica of the network and samples used to train it.
# else:  # We just use the images as features directly
#     features = MNIST_train.dataset.imgs

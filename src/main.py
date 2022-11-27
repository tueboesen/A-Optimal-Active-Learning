import os
from datetime import datetime

import matplotlib
import torch
import torch.optim as optim

from src.feature_transforms import autoencode_features
from src.losses import select_loss_fnc
from src.networks import select_network
from src.optimization import test
from src.results import init_results, save_results

from src import log
from src.Laplacian import compute_laplacian
from src.active_learning import run_active_learning, initial_labeling
from src.dataloader import select_dataset
from src.utils import fix_seed
from src.visualization import plot_results, select_preview

matplotlib.use('Agg')


def main(c) -> None:
    """
    The main function to run that sets up everything and performs the active learning.
    :param c: a configuration args.namespace object containing all the relevant parameters for this run.
    :return:
    """
    # Initialize things
    c.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fix_seed(c.seed)  # Set a seed, so we make reproducible results.
    c.result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c.basefolder,
        date=datetime.now(),
    )

    os.makedirs(c.result_dir)
    logfile_loc = "{}/{}.log".format(c.result_dir, 'output')
    c.LOG = log.setup_custom_logger(logfile_loc, c.mode)
    c.LOG.info('---------Listing all parameters-------')
    state = {k: v for k, v in c._get_kwargs()}
    for key, value in state.items():
        c.LOG.info("{:30s} : {}".format(key, value))

    # Load Dataset
    dl_train, dl_test = select_dataset(c.dataset, c.batch_size, c.nsamples_train, c.nsamples_test, c.binary)

    # Transform features?
    if c.feature_transform == '':
        features = dl_train.dataset.imgs
    elif c.feature_transform == 'autoencoder':
        features = autoencode_features(dl_train, c)
    else:
        raise NotImplementedError("This feature transform is not currently implemented")

    # Calculate Laplacian
    L, A = compute_laplacian(features, metric=c.L_metric, knn=c.L_knn, union=True)
    # L = None

    # Save preview
    select_preview(c.dataset, dl_train, save="{}/{}.png".format(c.result_dir, 'True_classes'))

    # Setup the result data structure
    results = init_results(c.AL_methods)

    for i in range(c.nrepeats):
        for j, (method_name, method_val) in enumerate(c.AL_methods.items()):
            if method_val:
                c.LOG.info('Date:{}, Starting:{}, iteration:{}...'.format(datetime.now(), method_name, i))
                net = select_network(c.SL_network, dl_train.dataset.nc, c.LOG)
                # optimizer = optim.SGD(list(net.parameters()), lr=5e-3,weight_decay=1e-5, momentum=0.9)
                optimizer = optim.Adam(list(net.parameters()), lr=5e-3)

                dl_train, idx_labels = initial_labeling(c.initial_labeling_mode, c.nlabels, dl_train)
                idx_labels, result = run_active_learning(method_name, dl_train.dataset.plabels_true.numpy(), idx_labels, L, c, net=net, optimizer=optimizer, dl_train=dl_train, dl_test=dl_test)
                if dl_test is not None:
                    result['test_acc_end'] = test(net, c.LOG, dataloader_test=dl_test, device=c.device)
                save_results(results, result, c.result_dir, j)
                plot_results(results, j, save=c.result_dir)

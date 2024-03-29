import time

import matplotlib
import torch.nn.functional as F

from src.losses import select_loss_fnc

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import torch
import torchvision


def test(net, LOG, dataloader_test, device='cpu'):
    """
    Standard testing routine.
    :param net: neural network to test
    :param LOG: Logging function
    :param dataloader_test: a pytorch dataloader
    :param device: device to run test on
    :return:
    """
    net.to(device)
    t0 = time.time()
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images_v, labels_v, _, _ in dataloader_test:
            images_v = images_v.to(device)
            labels_v = labels_v.to(device)
            outputs_v = net(images_v)
            predicted = torch.max(outputs_v.data, 1)[1]
            total += labels_v.size(0)
            correct += (predicted == labels_v).sum()
        accuracy = 100 * float(correct) / float(total)
        t1 = time.time()
        LOG.info('Accuracy(test): {:3.2f}%  Time: {:.2f} '.format(accuracy, t1 - t0))
    net.train()
    return accuracy


def train(net, optimizer, dataloader_train, loss_type, LOG, device='cpu', dataloader_test=None, epochs=100, use_probabilities=True):
    """
    Standard training routine.
    :param net: Network to train
    :param optimizer: Optimizer to use
    :param dataloader_train: Data to train on
    :param loss_type: loss function to use
    :param LOG: LOG file handler to print to
    :param device: device to perform computation on
    :param dataloader_test: Dataloader to test the accuracy on after each epoch.
    :param epochs: Number of epochs to train
    :param use_probabilities: If False the target will be one-hot, otherwise it will be some kind of probability array (might be zero centered)
    :return:
    """
    net.to(device)
    net.train()
    t0 = time.time()
    loss_fnc = select_loss_fnc(loss_type, use_probabilities=use_probabilities)
    accuracy = 0
    for epoch in range(epochs):
        loss_epoch = 0
        for i, (images, labels, plabels, idxs) in enumerate(dataloader_train):
            images = images.to(device)
            if use_probabilities:
                target = plabels.to(device)
            else:
                labels = labels.long()
                target = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if use_probabilities:
                prob = F.softmax(outputs, dim=1)
            else:
                prob = outputs
            loss = loss_fnc(prob, target).mean()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        if dataloader_test is not None:
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images_v, labels_v, _, _ in dataloader_test:
                    images_v = images_v.to(device)
                    labels_v = labels_v.to(device)
                    outputs_v = net(images_v)
                    predicted = torch.max(outputs_v.data, 1)[1]
                    total += labels_v.size(0)
                    correct += (predicted == labels_v).sum()
                accuracy = 100 * float(correct) / float(total)
                t1 = time.time()
                LOG.info('Epoch: {:4d}  Loss(train): {:6.2f}  Accuracy(test): {:3.2f}%  Time: {:.2f} '.format(epoch, loss_epoch, accuracy, t1 - t0))
            net.train()
    return net, accuracy


def test_dataset(net, dataset, device='cpu', batchsize=501):
    """
    Much like test, this function is designed to evaluate a neural network, but on a dataset rather than a dataloader.
    #TODO merge this with test
    :param net:
    :param dataset:
    :param device:
    :param batchsize:
    :return:
    """
    net.to(device)
    net.eval()
    with torch.no_grad():
        nsamples = len(dataset)
        prob_tmp = []
        for i in range(0, nsamples, batchsize):
            batch = dataset.imgs[i:min(i + batchsize, nsamples)]
            outputi = net(batch.to(device))
            probi = F.softmax(outputi, dim=1)
            prob_tmp.append(probi.cpu())
        prob = torch.cat(prob_tmp, dim=0)
    net.train()
    return prob


def train_AE(net, optimizer, dataloader_train, loss_fnc, LOG, device='cpu', epochs=100, save=None):
    """
    Training routine for an autoencoder
    :param net: Network to train
    :param optimizer: Optimizer to use
    :param dataloader_train: Data to train on
    :param loss_fnc: loss function to use
    :param LOG: LOG file handler to print to
    :param device: device to perform computation on
    :param epochs: Number of epochs to train
    :param save: if used, should be the filename (with path) of where to save an image of 100 random examples from the training set autoencoded and compared to their originals.
    :return:
        net -  trained network
        encoded - the full dataset transformed by the fully trained network to its encoded space
    """
    net.to(device)
    t0 = time.time()
    for epoch in range(epochs):
        loss_epoch = 0
        for i, (images, _, _, _) in enumerate(dataloader_train):
            images = images.to(device)
            optimizer.zero_grad()
            _, decoded = net(images)
            loss = loss_fnc(decoded, images)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        t1 = time.time()
        LOG.info('Epoch: {:4d}  Loss: {:6.2f}  Time: {:.2f} '.format(epoch, loss_epoch, t1 - t0))
    net.eval()
    with torch.no_grad():
        batchsize = 501
        nsamples = len(dataloader_train.dataset)
        enc_tmp = []
        dec_tmp = []
        for i in range(0, nsamples, batchsize):
            batch = dataloader_train.dataset.imgs[i:min(i + batchsize, nsamples)]
            encoded, decoded = net(batch.to(device))
            enc_tmp.append(encoded.cpu())
            dec_tmp.append(decoded.cpu())
        encoded = torch.cat(enc_tmp, dim=0)
        decoded = torch.cat(dec_tmp, dim=0)
        fig = plt.figure(figsize=[10, 5])
        idxs = list(range(nsamples))
        np.random.shuffle(idxs)
        plt.subplot(1, 2, 1);
        plt.imshow(torchvision.utils.make_grid(decoded[idxs[0:100]].cpu().detach(), 10).permute(1, 2, 0))
        plt.title('decoded')
        plt.subplot(1, 2, 2);
        plt.imshow(torchvision.utils.make_grid(dataloader_train.dataset.imgs[idxs[0:100]].cpu().detach(), 10).permute(1, 2, 0))
        plt.title('original')
        plt.pause(0.4)
        if save:
            fig.savefig(save)
    return net, encoded


def run_AE(net, dataloader, device='cpu'):
    """
    This function is used on an already trained autoencoder on a dataset in smaller batches. This should be removed and merged with eval_net instead
    :param net:
    :param dataloader:
    :param device:
    :return:
    """
    # TODO merge with test()
    net.to(device)
    net.eval()
    with torch.no_grad():
        batchsize = 501
        nsamples = len(dataloader.dataset)
        enc_tmp = []
        dec_tmp = []
        for i in range(0, nsamples, batchsize):
            batch = dataloader.dataset.imgs[i:min(i + batchsize, nsamples)]
            encoded, decoded = net(batch.to(device))
            enc_tmp.append(encoded.cpu())
            dec_tmp.append(decoded.cpu())
        encoded = torch.cat(enc_tmp, dim=0)
    return encoded

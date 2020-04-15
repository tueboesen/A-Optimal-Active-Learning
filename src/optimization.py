import torch
import time
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from losses import cross_entropy_probabilities


def train(net,optimizer,dataloader_train,loss_fnc,LOG,device='cpu',dataloader_validate=None,epochs=100,weights=None):
    if isinstance(weights,np.ndarray):
        weights = ((torch.from_numpy(weights)).float()).to(device)
    if weights is None:
        weights = torch.ones(len(dataloader_train.dataset)).to(device)
    net.to(device)
    accuracy_list = []
    t0 = time.time()
    for epoch in range(epochs):
        loss_epoch = 0
        for i, (images, labels, idxs) in enumerate(dataloader_train):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if dataloader_train.dataset.use_label_probabilities:
                loss = cross_entropy_probabilities(input=outputs, target=labels) #Note we do not use the weights in this case, since the probabilities are inside the target, rather than used as weights
            else:
                labels = labels.long()
                loss = (weights[idxs]*loss_fnc(outputs, labels)).mean()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        if dataloader_validate is not None:
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images_v, labels_v,_ in dataloader_validate:
                    images_v = images_v.to(device)
                    labels_v = labels_v.to(device)
                    outputs_v = net(images_v)
                    predicted = torch.max(outputs_v.data, 1)[1]
                    total += labels_v.size(0)
                    correct += (predicted == labels_v).sum()
                accuracy = 100 * float(correct) / float(total)
                accuracy_list.append(accuracy)
                t1 = time.time()
                LOG.info('Epoch: {:4d}  Loss: {:6.2f}  Accuracy: {:3.2f}%  Time: {:.2f} '.format(epoch, loss_epoch, accuracy, t1-t0))
            net.train()
    return net

def eval_net(net,dataloader,device='cpu'):
    net.to(device)
    net.eval()
    with torch.no_grad():
        batchsize = 501
        nsamples = len(dataloader.dataset)
        output_tmp = []
        for i in range(0, nsamples, batchsize):
            batch = dataloader.dataset.imgs[i:min(i + batchsize,nsamples)]
            outputi = net(batch.to(device))
            output_tmp.append(outputi.cpu())
        output = torch.cat(output_tmp, dim=0)
    net.train()
    return output

def train_AE(net,optimizer,dataloader_train,loss_fnc,LOG,device='cpu',epochs=100,save=None):
    net.to(device)
    t0 = time.time()
    for epoch in range(epochs):
        loss_epoch = 0
        for i, (images, _, _) in enumerate(dataloader_train):
            images = images.to(device)
            optimizer.zero_grad()
            _,decoded = net(images)
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
            batch = dataloader_train.dataset.imgs[i:min(i + batchsize,nsamples)]
            encoded,decoded = net(batch.to(device))
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
    return net,encoded

def run_AE(net,dataloader,device='cpu'):
    net.to(device)
    net.eval()
    with torch.no_grad():
        batchsize = 501
        nsamples = len(dataloader.dataset)
        enc_tmp = []
        dec_tmp = []
        for i in range(0, nsamples, batchsize):
            batch = dataloader.dataset.imgs[i:min(i + batchsize,nsamples)]
            encoded,decoded = net(batch.to(device))
            enc_tmp.append(encoded.cpu())
            dec_tmp.append(decoded.cpu())
        encoded = torch.cat(enc_tmp, dim=0)

    return encoded
# loader for MNIST

import copy
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Dataset_preload(Dataset):
    '''
    This one preloads all the images!
    '''
    def __init__(self, dataset, device='cpu', nsamples=-1,order_data=False,use_label_probabilities=False):
        imgs = []
        self.labels = []
        self.device = device
        self.use_label_probabilities = use_label_probabilities
        for (img,label) in dataset:
            imgs.append(img)
            self.labels.append(label)
            if len(imgs) == nsamples:
                break

        self.imgs = torch.stack(imgs)
        if order_data:
            idxs = np.argsort(self.labels)
            self.imgs = self.imgs[idxs,:]
            self.labels = list(np.asarray(self.labels)[idxs])
        n = len(self.labels)
        nc = len(np.unique(self.labels))
        self.nc = nc
        if use_label_probabilities:
            labels = torch.zeros(n,nc)
            labels[torch.arange(labels.shape[0]),self.labels] = 1
            self.labels = labels
        self.labels_true = copy.deepcopy(self.labels)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img,label,index

    def __len__(self):
        return len(self.imgs)

def Load_MNIST(batch_size=1000,nsamples=-1, device ='cpu',order_data=False,use_label_probabilities=False, download=False):
    '''
    Loads MNIST dataset into the pytorch dataloader structure


    :param batch_size: number of data points return in each batch
    :param nsamples: number of data points in the total set
    :param device: device data is loaded to
    :param order_data: should the images be sorted according to label? (they will still be randomly drawn)
    :param use_label_probabilities: should labels be one hot, or a tensor of label probabilities
    :param download: First time it is run it should be run with download=true, afterwards it can be set to false.
    :return: MNIST_train, MNIST_test
    '''
    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    MNISTtrainset = torchvision.datasets.MNIST(root='../data', train=True, transform=trans, download=download)
    MNISTtestset = torchvision.datasets.MNIST(root='../data', train=False, transform=trans)

    MNISTtrainset_pre = Dataset_preload(MNISTtrainset,nsamples=nsamples,device=device,order_data=order_data,use_label_probabilities=use_label_probabilities)
    MNISTtestset_pre = Dataset_preload(MNISTtestset,nsamples=nsamples,device=device)

    MNIST_train = torch.utils.data.DataLoader(MNISTtrainset_pre, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    MNIST_test = torch.utils.data.DataLoader(MNISTtestset_pre, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return MNIST_train, MNIST_test

def set_labels(idx,dataloader,class_balance=False):
    '''
    Sets the data labels on a dataloader
    If idx is a list, it labels those indices according to labels_true, and leaves all others unlabelled.
    If idx is an integer, it randomly labels that many indices according to labels_true.
    If idx is an array of size (n,nc) then it labels all the labels according to this array.
    If class_balance == True, the random labelling is done in such a way that each class has the same number of labels or at most 1 off.
    :param idx: can be a list (of indices), integer, or array of probabilities (should be all samples)
    :param dataloader: Dataloader to be updated.
    :param class_balance: Only relevant if idx is an integer, it then makes sure the labels are set with equal number of labels from each class or as close as possible.
    :return: Updated dataloader
    '''
    if type(idx) is int:
        if class_balance:
            nidx = idx
            idx = []
            #First determine how many of each label we should find:
            nc = dataloader.dataset.nc
            labels_unique = np.asarray(range(nc),dtype=np.float32)
            classes = np.zeros(nc,dtype=int)
            for i in range(nidx):
                classidx = np.mod(i,nc)
                classes[classidx] += 1
            #Now classes should sum to idx, and contain how many labels of each different class we need.
            for i,label in enumerate(labels_unique):
                if dataloader.dataset.use_label_probabilities:
                    indices = np.where(torch.argmax(dataloader.dataset.labels_true,dim=1).numpy() == label)[0]
                else:
                    indices = np.where(dataloader.dataset.labels_true == label)[0]
                assert len(indices) >= classes[i]
                np.random.shuffle(indices)
                idx += indices[0:classes[i]].tolist()
        else:
            nx = list(range(len(dataloader.dataset)))
            random.shuffle(nx)
            idx = np.asarray(nx[:idx])
    elif type(idx) is np.ndarray:
        if dataloader.dataset.use_label_probabilities:
            assert dataloader.dataset.labels.shape == idx.shape, 'Dimension mismatch (not tested yet)'
            dataloader.dataset.labels = torch.from_numpy(idx).float()
        else:
            dataloader.dataset.labels[:] = np.argmax(idx,axis=1)
        return dataloader
    if dataloader.dataset.use_label_probabilities:
        dataloader.dataset.labels[:,:] = -1
    else:
        dataloader.dataset.labels[:] = [-1] * len(dataloader.dataset)
    for i in idx:
        dataloader.dataset.labels[i] = dataloader.dataset.labels_true[i]
    return dataloader

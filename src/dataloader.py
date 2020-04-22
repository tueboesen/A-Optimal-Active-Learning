# loader for MNIST

import copy
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Dataset_preload_with_label_prob(Dataset):
    '''
    This one preloads all the images! And always calculates the label probabilities as well, it is then up to the user whether to use one or the other.
    This is of course slower than just loading one of the two, but should not be that much slower.
    zero_center_label_probabilities: makes the label probabilities obey the constraint ye=0, where y is the label probabilities.
    '''
    def __init__(self, dataset, device='cpu', nsamples=-1,order_data=False,zero_center_label_probabilities=True,use_1_vs_all=-1):
        imgs = []
        self.labels = []
        self.device = device
        self.zero_center_label_probabilities= zero_center_label_probabilities
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
        if use_1_vs_all >= 0:
            ls = use_1_vs_all
            assert (ls in set(np.unique(self.labels))), "the label you have selected for use_1_vs_all_dataset is not a valid label. You selected {}, but the labels are {}".format(ls,set(np.unique(self.labels)))
            idxs = np.where(self.labels == np.float32(ls))[0]
            self.labels[:] = [1]*n
            for idx in idxs:
                self.labels[idx] = 0

        nc = len(np.unique(self.labels))
        self.nc = nc
        self.labels_true = copy.deepcopy(self.labels)
        plabels = torch.zeros(n,nc)
        plabels[torch.arange(plabels.shape[0]),self.labels] = 1
        if zero_center_label_probabilities:
            plabels = plabels - torch.mean(plabels,dim=1)[:,None]
        self.plabels = plabels
        self.plabels_true = copy.deepcopy(self.plabels)

    def __getitem__(self, index):
        img = self.imgs[index]
        plabel = self.plabels[index]
        label = self.labels[index]
        return img,label,plabel,index

    def __len__(self):
        return len(self.imgs)

def Load_MNIST(batch_size=1000,nsamples=-1, device ='cpu',order_data=False,download=True,use_1_vs_all=-1):
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

    MNISTtrainset_pre = Dataset_preload_with_label_prob(MNISTtrainset,nsamples=nsamples,device=device,order_data=order_data,use_1_vs_all=use_1_vs_all)
    MNISTtestset_pre = Dataset_preload_with_label_prob(MNISTtestset,nsamples=nsamples,device=device)

    MNIST_train = torch.utils.data.DataLoader(MNISTtrainset_pre, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    MNIST_test = torch.utils.data.DataLoader(MNISTtestset_pre, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return MNIST_train, MNIST_test

def set_labels(idx,dataloader,class_balance=False,remove_all_unknown_labels=True):
    '''
    Sets the data labels on a dataloader
    If idx is a list, it sets labels[idx] = true_labels[idx] and plabels[idx] = plabels_true[idx]
    If idx is an integer, it randomly creates a list of indices with that many elements and does the list option.
    If idx is an array of size (n,nc) then it labels all the plabels according to this array, but LEAVES THE labels UNTOUCHED.
    If class_balance == True, the random labelling is done in such a way that each class has the same number of labels or at most 1 off.
    :param idx: can be a list (of indices), integer, or array of probabilities (should be all samples)
    :param dataloader: Dataloader to be updated.
    :param class_balance: Only relevant if idx is an integer, it then makes sure the labels are set with equal number of labels from each class or as close as possible.
    :param remove_all_unknown_labels: If True, it will set all plabels to 0 and all labels=-1 before updating the known idxs.
    :return: Updated dataloader
    '''
    if type(idx) is int:
        if class_balance:
            nidx = idx
            if not remove_all_unknown_labels:
                known_idx = np.where(dataloader.dataset.labels != np.float32(-1))[0]
                idx = known_idx.tolist()
            else:
                known_idx = np.empty(0)
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
                indices = np.where(dataloader.dataset.labels_true == label)[0]
                indices = list(set(indices).difference(set(known_idx))) #Remove known indices
                assert len(indices) >= classes[i], "There were not enough datapoints of class {} left. Needed {} indices, but there are only {} available. Try increasing the dataset.".format(i,classes[i],len(indices))
                np.random.shuffle(indices)
                idx += indices[0:classes[i]]
        else:
            nx = list(range(len(dataloader.dataset)))
            random.shuffle(nx)
            idx = np.asarray(nx[:idx])
    elif type(idx) is np.ndarray:
        assert dataloader.dataset.plabels.shape == idx.shape, 'Dimension mismatch'
        dataloader.dataset.plabels = torch.from_numpy(idx).float()
        return dataloader
    if remove_all_unknown_labels:
        dataloader.dataset.plabels[:,:] = 0
        dataloader.dataset.labels[:] = [-1] * len(dataloader.dataset)
    for i in idx:
        dataloader.dataset.labels[i] = dataloader.dataset.labels_true[i]
        dataloader.dataset.plabels[i] = dataloader.dataset.plabels_true[i]
    return dataloader

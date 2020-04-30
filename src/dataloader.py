# loader for MNIST

import copy
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data import Dataset


def Load_dataset(c,device='cpu'):
    if c['dataset'] == 'mnist':
        dl_train, dl_test = Load_MNIST(batch_size=c['batch_size'], nsamples=c['nsamples'], device=device,
                                             order_data=c['order_dataset'], use_1_vs_all=c['use_1_vs_all_dataset'])
    elif c['dataset'] == 'circles':
        dl_train, dl_test = Load_circles(batch_size=c['batch_size'], nsamples=c['nsamples'], device=device)
    else:
        raise ValueError("Selected dataset: {}, has not been implemented yet.".format(c['dataset']))
    return dl_train,dl_test

class Dataset_preload_with_label_prob(Dataset):
    '''
    This one preloads all the images! And always calculates the label probabilities as well, it is then up to the user whether to use one or the other.
    This is of course slower than just loading one of the two, but should not be that much slower.
    zero_center_label_probabilities: makes the label probabilities obey the constraint ye=0, where y is the label probabilities.
    '''
    def __init__(self, dataset, device='cpu', nsamples=-1,order_data=False,zero_center_label_probabilities=True,use_1_vs_all=-1,name=''):
        imgs = []
        self.labels = []
        self.device = device
        self.zero_center_label_probabilities= zero_center_label_probabilities
        self.name = name
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
        if zero_center_label_probabilities:
            plabels = torch.zeros(n, nc)
            plabels[torch.arange(plabels.shape[0]),self.labels] = 10
            plabels = plabels - torch.mean(plabels,dim=1)[:,None]
        else:
            plabels = -torch.ones(n, nc)
            plabels[torch.arange(plabels.shape[0]), self.labels] = 1
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

    MNISTtrainset_pre = Dataset_preload_with_label_prob(MNISTtrainset,nsamples=nsamples,device=device,order_data=order_data,use_1_vs_all=use_1_vs_all,name='mnist',zero_center_label_probabilities=False)
    MNISTtestset_pre = Dataset_preload_with_label_prob(MNISTtestset,nsamples=nsamples,device=device,name='mnist',zero_center_label_probabilities=False)

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


def Load_circles(batch_size=100,nsamples=2500, device ='cpu'):
    """

    :param zones: (2xn array, which specifies circle-rings in which the data can lie)
    :param nsamples:  Maximum number of samples generated, will in practice be less.
    :param noiseratio: The ratio of noise points to samplepoints, 0 means no noise.
    :return:
    """
    nsamples_train = [round(nsamples/6), round(nsamples/3), round(nsamples/2)]
    radial_centers = [0 , 1.5, 3]
    decay_length = [0.3, 0.3 , 0.3]

    # nsamples_train = [round(nsamples/10), round(9*nsamples/10)]
    # radial_centers = [0 , 3]
    # decay_length = [0.3, 0.3]

    trainset = CreateCircleDataset(radial_centers,nsamples_train,decay_length,return_tensor=True)
    train_data = Dataset_preload_with_label_prob(trainset, nsamples=nsamples, device=device,name='circles')
    dl_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=0)

    nsamples = 1000
    nsamples_test = [round(nsamples/6), round(nsamples/3), round(nsamples/2)]
    radial_centers = [0 , 1, 3]
    decay_length = [0.1, 0.2 , 0.5]
    testset = CreateCircleDataset(radial_centers,nsamples_test,decay_length,return_tensor=True)
    test_data = Dataset_preload_with_label_prob(testset, nsamples=nsamples, device=device,name='circles')
    dl_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=0)

    return dl_train,dl_test

def relabel_dataset(dataset,known_labels_from_each):
    features = dataset[0]
    targets_truth = dataset[1]
    dt = targets_truth.dtype
    targets_clean = [i for i in targets_truth if i != -1]
    targets_unique = np.unique(targets_clean)
    idxs = []
    for target in targets_unique:
        idx = np.where(targets_truth == target)[0]
        assert len(idx) >= known_labels_from_each
        np.random.shuffle(idx)
        idxs.append(idx[0:known_labels_from_each])
    targets = - np.ones_like(targets_truth,dtype=dt)
    for idx in idxs:
        targets[idx] = targets_truth[idx]
    dataset = (features, targets)
    return dataset, targets_truth

def CreateCircleDataset(radial_centers,nsamples,decay_length,return_tensor=False):
    features_circle,target_circle = CreateGaussianCircles(radial_centers, decay_length, nsamples)
    if return_tensor:
        features_circle = torch.from_numpy(features_circle)
    dataset = zip(features_circle, target_circle)
    return dataset

def CreateCircles(zones,nsamples):
    """

    :param zones: (2xn array, which specifies circle-rings in which the data can lie)
    :param nsamples: Maximum number of samples generated, will in practice be less.
    :return:
    """

    ma=np.max(zones)
    position = np.empty((nsamples,2), dtype=np.single)
    target = np.empty(nsamples,dtype=np.int64)
    ctn = 0
    finished = False
    while not finished:
        x = ma * np.random.randn(nsamples, 1)
        y = ma * np.random.randn(nsamples, 1)
        r = x * x + y * y
        for i, zone in enumerate(zones):
            if finished:
                break
            idx = ((r > pow(zone[0], 2)) & (r < pow(zone[1], 2)))
            for (xi, yi) in zip(x[idx], y[idx]):
                position[ctn, :] = [xi, yi]
                target[ctn] = i
                ctn += 1
                if ctn == nsamples:
                    finished = True
                    break
    return position, target

def CreateGaussianCircles(radial_centers,decay_length,nsamples):
    """
    Creates gaussian circledisc
    :param zones: (2xn array, which specifies circle-rings in which the data can lie)
    :param nsamples: Maximum number of samples generated, will in practice be less.
    :return:
    """

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    ncircles = len(radial_centers)
    nsamples_tot = sum(nsamples)
    features = np.zeros((nsamples_tot,2),dtype=np.float32)
    target = np.zeros(nsamples_tot,dtype=np.int)
    idx0 = 0
    idx1 = 0
    for i in range(ncircles):
        r = radial_centers[i] + (-1)**np.random.randint(0,1) * decay_length[i] *(np.random.randn(nsamples[i], 1))
        angle = 2*np.pi*np.random.rand(nsamples[i],1)
        x,y = pol2cart(r,angle)
        idx1 += nsamples[i]
        features[idx0:idx1,0] = np.squeeze(x)
        features[idx0:idx1,1] = np.squeeze(y)
        target[idx0:idx1] = i
        idx0 = idx1
    return features,target



def AddNoise(X,ns):
    Xmin=np.min(X)
    Xmax=np.max(X)
    Xmean=(Xmax-Xmin)/2
    position = 2*Xmean*np.random.rand(ns,2)-Xmean
    position = position.astype(np.single)
    target = - np.ones((len(position)), dtype=np.int64)
    return position, target

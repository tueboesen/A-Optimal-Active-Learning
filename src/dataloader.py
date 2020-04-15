# loader for MNIST
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy
import numpy as np
import random

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

def Load_MNIST(batch_size=1000,nsamples=-1, device ='cpu',order_data=False,use_label_probabilities=False):
    '''
    :param batch_size: batch_size desired in the dataloader.
    :return: training_dataloader,testing_dataloader

    First time it is run it should be run with download=true, afterwards it can be set to false.
    '''

    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # MNIST dataset
    MNISTtrainset = torchvision.datasets.MNIST(root='../data', train=True, transform=trans, download=True)
    MNISTtestset = torchvision.datasets.MNIST(root='../data', train=False, transform=trans)

    MNISTtrainset_pre = Dataset_preload(MNISTtrainset,nsamples=nsamples,device=device,order_data=order_data,use_label_probabilities=use_label_probabilities)
    MNISTtestset_pre = Dataset_preload(MNISTtestset,nsamples=nsamples,device=device)

    MNIST_train = torch.utils.data.DataLoader(MNISTtrainset_pre, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    MNIST_test = torch.utils.data.DataLoader(MNISTtestset_pre, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return MNIST_train, MNIST_test

def Load_Maps(image_dir, resize_height=48, resize_width=48, device='cpu', batch_size=500, paired=True, nsamples=-1):
    '''
    Given a folder it finds the train/test/val folders and pairs them up.
    :param image_dir:
    :param resize_height:
    :param resize_width:
    :param mean:
    :param std:
    :return:
    '''
    folders = ['train']
    # folders = ['trainA','testA','valA','trainB','testB','valB']
    dataloaders = {}
    for folder in folders:
        data_dir_A = os.path.join(image_dir,folder+'A')
        data_dir_B = os.path.join(image_dir,folder+'B')
        if paired:
            data = Dataset_paired_preload(image_dirA=data_dir_A, image_dirB=data_dir_B, device=device, nsamples=nsamples, resize_height=resize_height, resize_width=resize_width)
        else:
            raise NotImplementedError
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloaders.update({folder: dataloader})

    return dataloaders


class Maps_Dataset(Dataset):

    def __init__(self, image_dir, resize_height=48, resize_width=48, mean=[.5, .5, .5], std=[.5, .5, .5]):
        imgs = os.listdir(image_dir)
        self.imgs = [os.path.join(image_dir, k) for k in imgs]
        self.transform = transforms.Compose([
            transforms.CenterCrop((resize_height, resize_width)),
            transforms.ToTensor(), # Transform images to Tensor,Normalize it to range [0,1]
            transforms.Normalize(mean=mean, std=std)]
        )

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transform(pil_img)

        return data

    def __len__(self):
        return len(self.imgs)


class Dataset_paired_preload(Dataset):
    '''
    This one preloads all the images and applies the transform on preload as well!
    '''
    def __init__(self, image_dirA, image_dirB, resize_height=48, resize_width=48, mean=[.5, .5, .5], std=[.5, .5, .5],device='cpu', nsamples=-1):
        dirA = os.listdir(image_dirA)
        dirB = os.listdir(image_dirB)
        self.imgs_dirA = [os.path.join(image_dirA, k) for k in dirA]
        self.imgs_dirB = [os.path.join(image_dirB, k) for k in dirB]
        self.device = device
        self.transform = transforms.Compose([
            # transforms.CenterCrop((resize_height, resize_width)),
            transforms.ToTensor(),  # Transform images to Tensor,Normalize it to range [0,1]
            transforms.Normalize(mean=mean, std=std)]
        )
        self.imgsA = []
        for img_dir in self.imgs_dirA:
            img = self.transform(Image.open(img_dir))  #.to(device)
            patches = img.data.unfold(0, 3, 3).unfold(1, resize_height, resize_height).unfold(2, resize_width, resize_width)
            tmp= patches.reshape([-1,patches.shape[3],patches.shape[4],patches.shape[5]])
            for i in range(tmp.shape[0]):
                self.imgsA.append(tmp[i].squeeze())
                if len(self.imgsA) == nsamples:
                    break
            if len(self.imgsA) == nsamples:
                break

        self.imgsB = []
        for img_dir in self.imgs_dirB:
            img = self.transform(Image.open(img_dir))  #.to(device)
            patches = img.data.unfold(0, 3, 3).unfold(1, resize_height, resize_height).unfold(2, resize_width, resize_width)
            tmp= patches.reshape([-1,patches.shape[3],patches.shape[4],patches.shape[5]])
            for i in range(tmp.shape[0]):
                self.imgsB.append(tmp[i].squeeze())
                if len(self.imgsB) == nsamples:
                    break
            if len(self.imgsB) == nsamples:
                break

    def __getitem__(self, index):
        imgA = self.imgsA[index].to(self.device)
        imgB = self.imgsB[index].to(self.device)
        return imgA, imgB

    def __len__(self):
        return len(self.imgsA)

    #To determine mean/std use:
    # mean = 0.
    # std = 0.
    # nb_samples = 0
    # for data, _ in dataloader:
    #     batch_samples = data.size(0)
    #     data = data.view(batch_samples, data.size(1), -1)
    #     mean += data.mean(2).sum(0)
    #     std += data.std(2).sum(0)
    #     nb_samples += batch_samples
    # mean /= nb_samples
    # std /= nb_samples


def set_labels(idx,dataloader,class_balance=False):
    #Sets the data labels on a dataloader
    #If idx is a list, it labels those indices according to labels_true, and leaves all others unlabelled.
    #If idx is an integer, it randomly labels that many indices according to labels_true.
    #If idx is an array of size (n,nc) then it labels all the labels according to this array.
    #If class_balance == True, the random labelling is done in such a way that each class has the same number of labels or at most 1 off.
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
                # idx.append(indices[0:classes[i]][:])
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

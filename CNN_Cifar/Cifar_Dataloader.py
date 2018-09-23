############################ Import Necessary packages ##########################################
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.core.debugger import Tracer
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy
from torch.autograd import grad
import time
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
import matplotlib
import operator
from torch.utils.data import Dataset, DataLoader

def SelectImage(Cifar_data_train,Cifar_data_test):
    tensor_train=[]
    tensor_test=[]
    tensor_train_label=[]
    tensor_test_label=[]

    # for training dataset
    for ind,data in enumerate(Cifar_data_train):
        if (ind+1)%5000==0:
            print('Training_Current process ind:%s'%(ind+1))
        # x size:3x32x32 y: int Note: x is tensor with requires_grad=False
        x=data[0]
        y=data[1]
        tensor_train.append(x)
        tensor_train_label.append(y)
        # tensor_train: 50000x3x32x32 tensor_train_label 50000x1
    for ind,data in enumerate(Cifar_data_test):
        if (ind+1)%5000==0:
            print('Testing_Current process ind:%s'%(ind+1))
        x = data[0]
        y = data[1]
        tensor_test.append(x)
        tensor_test_label.append(y)
    return tensor_train,tensor_train_label,tensor_test,tensor_test_label








class Cifar_class(Dataset):
    '''
    This is to create faster dataloader from the initialized dataset. The transformation in original Cifar 10 is slow.

    X:List:Item::math:`ChannelxHeightxWidth`: Data

    Y:List:Item:Int: label

    transform: Set to None for default.

    '''
    def __init__(self,X,Y,flag='Train',transform=None):
        self.X=X
        self.Y=Y
        self.flag=flag
        self.transform=transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, ind):
        (sample_x,sample_y)=self.X[ind],self.Y[ind]
        if self.transform != None:
            (sample_x, sample_y) = self.transform((sample_x, sample_y))

        return (sample_x, sample_y)


def SelectImage_DataGen(train_loader, test_loader, train_image=[0, 4], test_image=[5, 9]):
    '''
    This is to split the data into two groups
    :param train_loader:
    :param test_loader:
    :param train_image:
    :param test_image:
    :return:
    '''
    X_train_sampler_tensor = [] # This is used to train the sampler (dataset for train task)
    X_train_CNN_tensor = [] # The corresponding test dataset for train task
    Y_train_sampler_tensor = []
    Y_train_CNN_tensor = []
    X_test_sampler_tensor = [] # This is the dataset for training in test task
    Y_test_sampler_tensor = [] # True test dataset
    X_test_CNN_tensor = []
    Y_test_CNN_tensor = []

    for ind, data in enumerate(train_loader):
        if (ind + 1) % 5000 == 0:
            print('Ind:%s' % (ind + 1))
        x, label = torch.squeeze(data[0]), data[1]
        if label >= train_image[0] and label <= train_image[1]:
            X_train_sampler_tensor.append(x)
            Y_train_sampler_tensor.append(label)
        else:
            X_train_CNN_tensor.append(x)
            Y_train_CNN_tensor.append(label - 5)

    for ind, data in enumerate(test_loader):
        if (ind + 1) % 5000 == 0:
            print('Ind:%s' % (ind + 1))
        x, label = torch.squeeze(data[0]), data[1]
        if label >= test_image[0] and label <= test_image[1]:
            X_test_CNN_tensor.append(x)
            Y_test_CNN_tensor.append(label - 5)
        else:
            X_test_sampler_tensor.append(x)
            Y_test_sampler_tensor.append(label)
    return X_train_sampler_tensor, X_train_CNN_tensor, Y_train_sampler_tensor, Y_train_CNN_tensor, X_test_CNN_tensor, Y_test_CNN_tensor, X_test_sampler_tensor, Y_test_sampler_tensor


class GroupCifar(Dataset):
    '''
    This is used to group different dataset
    '''
    def __init__(self, x, y, group, transform=None):
        if group == 1:
            print('Data used for train sampler')
        elif group == 2:
            print('Data used for train CNN using trained sampler')
        elif group == 3:
            print('Data used for Test CNN')
        elif group == 4:
            print('Data used for test sampler training')
        self.x = x
        self.y = y
        self.group = group
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        (sample_x, sample_y) = self.x[idx], self.y[idx]
        if self.transform != None:
            (sample_x, sample_y) = self.transform((sample_x, sample_y))
        return (sample_x, sample_y)


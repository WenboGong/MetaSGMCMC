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






        

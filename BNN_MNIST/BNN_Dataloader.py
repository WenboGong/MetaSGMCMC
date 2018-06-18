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
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def SelectImage_All(All_train_loader,All_test_loader,flag_valid=False):
    train_x_list=[]
    train_y_list=[]
    test_x_list=[]
    test_y_list=[]
    valid_x_list=[]
    valid_y_list=[]
    counter=1
    for ind,data in enumerate(All_train_loader):
        if (ind+1)%5000==0:
            print('Ind:%s'%(ind+1))
        x,y=torch.squeeze(data[0]),data[1]
        if flag_valid==True and counter>50000:
            valid_x_list.append(x)
            valid_y_list.append(y)
        else:
            train_x_list.append(x)
            train_y_list.append(y)
        counter+=1
    for ind,data in enumerate(All_test_loader):
        if (ind+1)%5000==0:
            print('Ind:%s'%(ind+1))
        x,y=torch.squeeze(data[0]),data[1]
        test_x_list.append(x)
        test_y_list.append(y)
    if flag_valid==True:
        return train_x_list,train_y_list,test_x_list,test_y_list,valid_x_list,valid_y_list
    else:
        return train_x_list,train_y_list,test_x_list,test_y_list
class NewMNISTLoader(Dataset):
    def __init__(self,X,Y,flag_train=True,transform=None):
        self.X=X
        self.Y=Y
        self.train=flag_train
        self.transform=transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        (sample_x,sample_y)=self.X[idx],self.Y[idx]
        if self.transform!=None:
            (sample_x,sample_y)=self.transform((sample_x,sample_y))
        return (sample_x,sample_y)
def SelectImage(Init_MNIST_train,Init_MNIST_test,train_image=[0,4],test_image=[5,9]):
    X_train_sampler_tensor=[]
    X_train_BNN_tensor=[]
    Y_train_sampler_tensor=[]
    Y_train_BNN_tensor=[]
    X_test_sampler_tensor=[]
    Y_test_sampler_tensor=[]
    X_test_BNN_tensor=[]
    Y_test_BNN_tensor=[]
    
    for ind,data in enumerate(Init_MNIST_train):
        if (ind+1)%5000==0:
            print('Ind:%s'%(ind+1))
        X,label=torch.squeeze(data[0]),data[1]
        if label>=train_image[0] and label<=train_image[1]:
            X_train_sampler_tensor.append(X)
            Y_train_sampler_tensor.append(label)
        else:
            X_train_BNN_tensor.append(X)
            Y_train_BNN_tensor.append(label-5)
            
    for ind,data in enumerate(Init_MNIST_test):
        if (ind+1)%5000==0:
            print('Ind:%s'%(ind+1))
        X,label=torch.squeeze(data[0]),data[1]
        if label>=test_image[0] and label<=test_image[1]:
            X_test_BNN_tensor.append(X)
            Y_test_BNN_tensor.append(label-5)
        else:
            X_test_sampler_tensor.append(X)
            Y_test_sampler_tensor.append(label)
    return X_train_sampler_tensor,X_train_BNN_tensor,Y_train_sampler_tensor,Y_train_BNN_tensor,X_test_BNN_tensor,Y_test_BNN_tensor,X_test_sampler_tensor,Y_test_sampler_tensor
class GroupMNIST(Dataset):
    def __init__(self,x,y,group,transform=None):
        if group==1:
            print('Data used for train sampler')
        elif group==2:
            print('Data used for train BNN using trained sampler')
        elif group==3:
            print('Data used for Test BNN')
        elif group==4:
            print('Data used for test sampler training')
        self.x=x
        self.y=y
        self.group=group
        self.transform=transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        (sample_x,sample_y)=self.x[idx],self.y[idx]
        if self.transform!=None:
            (sample_x,sample_y)=self.transform((sample_x,sample_y))
        return (sample_x,sample_y)
    

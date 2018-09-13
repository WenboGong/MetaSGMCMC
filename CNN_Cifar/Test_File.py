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
# Import Custom packages
from Cifar_Dataloader import *
from CNN_Module import *
from Test_Module import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')
###################### Define the Default dataloader ####################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######################## Define own data loader ###########################
tensor_train,tensor_train_label,tensor_test,tensor_test_label=SelectImage(trainset,testset)
Train_data=Cifar_class(tensor_train,tensor_train_label)
Test_data=Cifar_class(tensor_test,tensor_test_label)
train_loader=DataLoader(Train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(Test_data,batch_size=64,shuffle=False)

######################## Execute the CNN output ##########################
CNN=Example_CNN()
total_iter=50
############ define loss function
loss=nn.CrossEntropyLoss()
############ define optimizer
Adam=optim.Adam(CNN.parameters(),lr=0.0005,betas=(0.9,0.99))
for iter in range(total_iter):
    counter=0
    for data in enumerate(train_loader):
        Adam.zero_grad()
        counter+=1
        x,y=data[1][0].cuda(),data[1][1].cuda()
        x.requires_grad=True
        out=CNN.forward(x)
        loss_output=loss(out,y)
        loss_output.backward()
        Adam.step()
    if iter%5==0:
        print('iter:%s Loss:%s'%(iter,loss_output.data.cpu().numpy()))

Acc=Test_Accuracy(CNN,test_loader)
print('Classification Acc:%s'%(Acc))




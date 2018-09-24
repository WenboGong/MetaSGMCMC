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
import json
from torch.utils.data import Dataset, DataLoader
# Import Custom packages
from Cifar_Dataloader import *
from CNN_Module import *
from Test_Module import *
from Util import *
from CNN_Sampler import *
# Set default tensor type in GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
########################################################################################################################
###################### Define the Default dataloader ####################
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Define time
timestr = time.strftime("%Y%m%d-%H%M%S")
Param_Adam=GenerateParameters(Type='Adam',Step_Size=0.001,Betas=(0.9,0.99),Batch_Size=500,Epoch=200,Random_Seed=[10,11,12,13,14],Num_Run=5,Precision=1e-3,Test_Interval=10)
Param_SGDM=GenerateParameters(Type='SGDM',Step_Size=0.003,Momentum=0.9,Batch_Size=500,Epoch=200,Random_Seed=[10,11,12,13,14],Num_Run=5,Precision=1e-3,Test_Interval=10)

######################## Define own data loader ###########################
tensor_train,tensor_train_label,tensor_test,tensor_test_label=SelectImage(trainset,testset)
Train_data=Cifar_class(tensor_train,tensor_train_label)
Test_data=Cifar_class(tensor_test,tensor_test_label)
train_loader=DataLoader(Train_data,batch_size=Param['Batch Size'],shuffle=True)
test_loader=DataLoader(Test_data,batch_size=Param['Batch Size'],shuffle=True)
test_loader_seq=DataLoader(Test_data,batch_size=10000, shuffle=False)
############################## CNN Optimizer Test ############################################################

for n_r in range(5):
    for i in range(2):
        if i==0:
            Param=Param_Adam
        elif i==1:
            Param=Param_SGDM
        if Param['Type']=='Adam':
            ## Set Manual Seed
            print('Num Run:%s'%(n_r))
            print('Adam Optimizer')
            torch.manual_seed(Param['Random Seed'][n_r])

            CNN=Example_CNN(CNN_out_dim=10,AF='ReLU')
            epoch=Param['Epoch']
            ############ define loss function
            loss=nn.CrossEntropyLoss()
            ############ define optimizer
            Adam=optim.Adam(CNN.parameters(),lr=Param['Step Size'],betas=Param['Betas'])
            counter=0
            Acc_list=[]
            NLL_list=[]
            for ep in range(epoch):
                print('Epoch:%s'%(ep+1))
                for data in enumerate(train_loader):
                    Adam.zero_grad()
                    x,y=data[1][0].cuda(),data[1][1].cuda()
                    out=CNN.forward(x)
                    loss_output=loss(out,y)
                    for param in CNN.parameters():
                        loss_output+=Param['Precision']*torch.sum(param**2)
                    loss_output.backward()
                    Adam.step()
                    if (counter+1)%Param['Test Interval']==0:
                        Acc, NLL = Test_Accuracy_example(CNN,test_loader)
                        print('Counter:%s Acc:%s NLL:%s' % (counter + 1, Acc, NLL))
                        Acc_list.append(Acc)
                        NLL_list.append(NLL)
                    counter+=1
            Acc_list_np = np.asarray(Acc_list)
            NLL_list_np = np.asarray(NLL_list)
            np.savetxt('./Results/%s_%s_Acc'%(Param['Type'],timestr),Acc_list_np)
            np.savetxt('./Results/%s_%s_NLL'%(Param['Type'],timestr),NLL_list_np)
            write_Dict('./Results/%s_Param_%s'%(Param['Type'],timestr),Param)
        elif Param['Type']=='SGDM':
            print('Num Run:%s' % (n_r))
            print('SGDM Optimizer')
            torch.manual_seed(Param['Random Seed'][n_r])
            CNN = Example_CNN(CNN_out_dim=10, AF='ReLU')
            epoch = Param['Epoch']
            loss = nn.CrossEntropyLoss()
            SGDM = optim.SGD(CNN.parameters(), lr=Param['Step Size'], momentum=Param['Momentum'])
            counter = 0
            Acc_list = []
            NLL_list = []
            for ep in range(epoch):
                print('Epoch:%s'%(ep+1))
                for data in enumerate(train_loader):
                    SGDM.zero_grad()
                    x,y=data[1][0].cuda(),data[1][1].cuda()
                    out=CNN.forward(x)
                    loss_output=loss(out,y)
                    for param in CNN.parameters():
                        loss_output+=Param['Precision']*torch.sum(param**2)
                    loss_output.backward()
                    SGDM.step()
                    if (counter+1)%Param['Test Interval']==0:
                        Acc, NLL = Test_Accuracy_example(CNN,test_loader)
                        print('Counter:%s Acc:%s NLL:%s' % (counter + 1, Acc, NLL))
                        Acc_list.append(Acc)
                        NLL_list.append(NLL)
                    counter+=1
            Acc_list_np = np.asarray(Acc_list)
            NLL_list_np = np.asarray(NLL_list)
            np.savetxt('./Results/%s_%s_Acc'%(Param['Type'],timestr),Acc_list_np)
            np.savetxt('./Results/%s_%s_NLL'%(Param['Type'],timestr),NLL_list_np)
            write_Dict('./Results/%s_Param_%s'%(Param['Type'],timestr),Param)
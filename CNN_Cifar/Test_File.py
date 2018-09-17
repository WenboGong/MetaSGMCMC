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

### Define parameters
Param_Adam=GenerateParameters(Type='Adam',Batch_Size=64,Step_Size=0.001,Betas=(0.9,0.99),Epoch=100,Num_Run=1,Random_Seed=[10],Precision=1e-3)
Param_SGHMC=GenerateParameters(Type='SGHMC',Step_Size=0.001,Beta=0.,Batch_Size=500,Epoch=100,Random_Seed=[10],Num_Run=1,Sigma=1.,Num_CNN=20,
                               Mom_Resample=1000000,Alpha=0.01,Interval=5000)

Param=Param_SGHMC




######################## Define own data loader ###########################
tensor_train,tensor_train_label,tensor_test,tensor_test_label=SelectImage(trainset,testset)
Train_data=Cifar_class(tensor_train,tensor_train_label)
Test_data=Cifar_class(tensor_test,tensor_test_label)
train_loader=DataLoader(Train_data,batch_size=Param['Batch Size'],shuffle=True)
test_loader=DataLoader(Test_data,batch_size=Param['Batch Size'],shuffle=True)
test_loader_seq=DataLoader(Test_data,batch_size=10000, shuffle=False)
# train_loader=trainloader
# test_loader=testloader


######################################################### CNN Block ###############################################
# Train_Acc_list=np.zeros((Param['Num Run'],int(Param['Epoch']/5)+1))
# Test_Acc_list=np.zeros((Param['Num Run'],int(Param['Epoch']/5)+1))
# counter=0
# for num_runs in range(Param['Num Run']):
#     ## Set Manual Seed
#     torch.manual_seed(Param['Random Seed'][num_runs])
#
#     CNN=Example_CNN()
#     epoch=Param['Epoch']
#     ############ define loss function
#     loss=nn.CrossEntropyLoss()
#     ############ define optimizer
#     Adam=optim.Adam(CNN.parameters(),lr=Param['Step Size'],betas=Param['Betas'])
#     for ep in range(epoch):
#         for data in enumerate(train_loader):
#             Adam.zero_grad()
#             x,y=data[1][0].cuda(),data[1][1].cuda()
#             out=CNN.forward(x)
#             loss_output=loss(out,y)
#             for param in CNN.parameters():
#                 loss_output+=Param['Precision']*torch.sum(param**2)
#             loss_output.backward()
#             Adam.step()
#         if (ep+1)%5==0:
#             Test_Acc=Test_Accuracy(CNN,test_loader)
#             Train_Acc=Test_Accuracy(CNN,train_loader)
#             print('Ep:%s Train Acc:%s Test Acc:%s'%((ep+1),Train_Acc,Test_Acc))
#             Train_Acc_list[num_runs,counter]=Train_Acc
#             Test_Acc_list[num_runs,counter]=Test_Acc
#             counter+=1
# np.savetxt('./Results/%s_Train_%s'%(Param['Type'],timestr),Train_Acc_list)
# np.savetxt('./Results/%s_Test_%s'%(Param['Type'],timestr),Test_Acc_list)
# write_Dict('./Results/%s_Param_%s'%(Param['Type'],timestr),Param)
###################################################################################################################

# ######################################################### CNN Verify Block ########################################
# Train_Acc_list=np.zeros((Param['Num Run'],int(Param['Epoch']/1)+1))
# Test_Acc_list=np.zeros((Param['Num Run'],int(Param['Epoch']/1)+1))
# counter=0
# for num_runs in range(Param['Num Run']):
#     ## Set Manual Seed
#     torch.manual_seed(Param['Random Seed'][num_runs])
#
#     CNN=Parallel_CNN(num_CNN=1,fc_size=100)
#     total_dim=CNN.get_dimension()
#     # Define weight tensor
#     weight=0.05*torch.randn(1,total_dim)
#     weight.requires_grad=True
#
#     epoch=Param['Epoch']
#     ############ define loss function
#     loss=nn.CrossEntropyLoss()
#     ############ define optimizer
#     #Adam=optim.Adam([weight],lr=Param['Step Size'],betas=Param['Betas'])
#     Adam=optim.SGD([weight],lr=Param['Step Size'],momentum=0.99)
#     for ep in range(epoch):
#         for data in enumerate(train_loader):
#             Adam.zero_grad()
#             x,y=data[1][0].cuda(),data[1][1].cuda()
#             out=CNN.forward(x,weight)
#             grad_U,_,_,_=CNN.grad_CNN(x,y,weight,50000.,sigma=22.)
#             grad_U=torch.tensor(1*grad_U.data)
#             # loss_output=loss(torch.squeeze(out),y)
#             #
#             # loss_output+=Param['Precision']*torch.sum(weight**2)
#             # loss_output.backward()
#             Adam.step()
#         if (ep+1)%1==0:
#             Test_Acc,Test_NLL=Test_Accuracy(CNN,test_loader,weight.detach(),test_number=10000.)
#             Train_Acc,Train_NLL=Test_Accuracy(CNN,train_loader,weight.detach(),test_number=50000.)
#             print('Ep:%s Train Acc:%s Test Acc:%s'%((ep+1),Train_Acc,Test_Acc))
#             Train_Acc_list[num_runs,counter]=Train_Acc
#             Test_Acc_list[num_runs,counter]=Test_Acc
#             counter+=1
###################################################################################################################

########################################## SGHMC Test #############################################################
##Define initial parameters
eps=Param['Step Size']/50000.
alpha=Param['Alpha']
beta=Param['Beta']
epoch=Param['Epoch']
num_runs=Param['Num Run']
sigma=Param['Sigma']
num_CNN=Param['Num CNN']
mom_resample=Param['Mom Resample']
interval=Param['Interval']

# Define CNN network
CNN=Parallel_CNN(num_CNN=num_CNN,fc_size=100)
total_dim=CNN.get_dimension()
# Define Sampler
SGHMC_obj=SGHMC(total_dim,CNN)
# Initialize the weight and momentum
state_pos=0.05*torch.randn(num_CNN,total_dim,requires_grad=True)
state_mom=0.*torch.randn(num_CNN,total_dim)
Sequential_Accuracy_obj=Sequential_Accuracy(test_loader_seq,CNN)
state_list=SGHMC_obj.parallel_sample(state_pos,state_mom,train_loader,data_N=50000.,num_CNN=num_CNN,mom_resample=mom_resample,
                                     total_step=epoch,eps=eps,alpha=alpha,beta=beta,sigma=sigma,interval=interval,flag_SGLD=False,
                                     test_loader=test_loader,data_len=10000.,Sequential_Accuracy=Sequential_Accuracy_obj)

######################################### Import Necessary Packages#####################################################
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
import collections
from torch.utils.data import Dataset, DataLoader
# Import Custom packages
from Cifar_Dataloader import *
from CNN_Module import *
from Test_Module import *
from Util import *
from CNN_Module import *
from CNN_Q_D import *
from Test_Module import *
from CNN_training_func import *
from CNN_Sampler import *
########################################################################################################################

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
#15768 and 61478
data_N=50000.
eps1=float(np.sqrt(0.001/data_N))
eps2=float(np.sqrt(0.001/data_N))
Scale_Q_D=float(0.01/eps2)
coef=1.
Param_NNSGHMC=GenerateParameters('NNSGHMC Test',
                                Random_Seed=10,
                                Step_Size_1=eps1,
Step_Size_2=eps2,
Offset_Q=0,
Scale_Q_D=Scale_Q_D,
Scale_D=5,
Scale_G=100,
Offset_D=0,
Epoch=100,
Sample_Interval=200,
Mom_Resample=10000000,
Num_CNN=20,
Noise_Estimation=0,
Sigma=22.,
Clamp_Q=5.,
Clamp_D_Min=0.,
Clamp_D_Max=1000,
Batch_Size=500,
Coef=coef
                            )


Param=Param_NNSGHMC
######################## Define own data-loader ####################
tensor_train,tensor_train_label,tensor_test,tensor_test_label=SelectImage(trainset,testset)
Train_data=Cifar_class(tensor_train,tensor_train_label)
Test_data=Cifar_class(tensor_test,tensor_test_label)
train_loader=DataLoader(Train_data,batch_size=Param['Batch Size'],shuffle=True)
test_loader=DataLoader(Test_data,batch_size=Param['Batch Size'],shuffle=True)
test_loader_seq=DataLoader(Test_data,batch_size=10000, shuffle=False)

##################### Define parameters #####################
torch.manual_seed(Param['Random Seed'])
num_CNN=Param['Num CNN']
CNN=Parallel_CNN(num_CNN=num_CNN,fc_size=50,out_channel=8,flat_size=8*6*6)
Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
total_dim=CNN.get_dimension()
B=Param['Noise Estimation']
Q=parallel_Q_eff(CNN,Q_MLP,clamp=Param['Clamp Q'],offset=Param['Offset Q'])
D=parallel_D_eff(CNN,D_MLP,Param['Scale G'],Param['Scale D'],Param['Offset D'],Param['Clamp D Min'],Param['Clamp D Max'],
                 Param['Scale Q D'])
Gamma=parallel_Gamma_eff(flag_D=True)

# Load stored model
Q_MLP.load_state_dict(torch.load('./tmp_model_save/Q_MLP_20180918-150545_100'))
D_MLP.load_state_dict(torch.load('./tmp_model_save/D_MLP_20180918-150545_100'))

# Define Sampler
NNSGHMC_obj=NNSGHMC(CNN,Q,D,Gamma)
# Define Training parameters
epoch=Param['Epoch']
eps1=Param['Step Size 1']
eps2=Param['Step Size 2']
sigma=Param['Sigma']
coef=Param['Coef']


# Training
weight_init = 0.1 * torch.randn(num_CNN, total_dim,requires_grad=True)
state_mom_init=0.* torch.randn(num_CNN,total_dim,requires_grad=True)

state_list,state_mom_list,_=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,train_loader,data_N,
                                                        sigma,num_CNN,epoch,1000000000,eps1,eps2,
                                                        1000000,coef,Param['Sample Interval'],0,Param['Mom Resample'],
                                                        False,test_loader,10000.)
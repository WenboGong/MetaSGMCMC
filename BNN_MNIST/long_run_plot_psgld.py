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
from BNN_Util import *
from BNN_Q_D import *
from BNN_Model_def import *
from BNN_Sampler import *
from BNN_training_func import *
from BNN_Dataloader import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def ReadStoredSamples(FilePath):
    tensor_state_list=torch.load(FilePath)
    state_list_tmp=list(torch.split(tensor_state_list,1,dim=0))
    state_list=[]
    for i in state_list_tmp:
        state_list.append(torch.squeeze(i))
    return state_list

File_Path_psgld_list=['./ReLU_Generalization_Long_Run/Test_long_run_psgld_1_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_2_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_3_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_4_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_5_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_6_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_7_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_8_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_9_step_1.4',
                      './ReLU_Generalization_Long_Run/Test_long_run_psgld_10_step_1.4'
                     ]


################################# Dataloder other than Dataset Gen ##################################
train_loader = datasets.MNIST('./BNN_MNIST/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_loader = datasets.MNIST('./BNN_MNIST/data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_X,train_Y,test_X,test_Y=SelectImage_All(train_loader,test_loader)
train_class=NewMNISTLoader(train_X,train_Y,flag_train=True)
test_class=NewMNISTLoader(test_X,test_Y,flag_train=False)

train_loader=DataLoader(train_class, batch_size=500,
                        shuffle=True)
test_loader=DataLoader(test_class,batch_size=500,shuffle=True)

MNIST_test_BNN_loader=test_loader
MNIST_test_BNN=test_class

##################################### DATA GEN ################################################
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./BNN_MNIST/data/', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=500, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./BNN_MNIST/data/', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=500, shuffle=True)
# Init_MNIST_train=datasets.MNIST('./BNN_MNIST/data/', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
# Init_MNIST_test=datasets.MNIST('./BNN_MNIST/data/', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
# X_train_sampler_tensor,X_train_BNN_tensor,Y_train_sampler_tensor,Y_train_BNN_tensor,X_test_BNN_tensor,Y_test_BNN_tensor,X_test_sampler_tensor,Y_test_sampler_tensor=SelectImage(Init_MNIST_train,Init_MNIST_test,train_image=[0,4],test_image=[5,9])

# MNIST_train_sampler=GroupMNIST(X_train_sampler_tensor,Y_train_sampler_tensor,group=1)
# MNIST_train_sampler_loader = DataLoader(MNIST_train_sampler, batch_size=500,
#                         shuffle=True)
# MNIST_test_sampler=GroupMNIST(X_test_sampler_tensor,Y_test_sampler_tensor,group=4)
# MNIST_test_sampler_loader = DataLoader(MNIST_test_sampler, batch_size=500,
#                         shuffle=True)

# MNIST_train_BNN=GroupMNIST(X_train_BNN_tensor,Y_train_BNN_tensor,group=2)
# MNIST_train_BNN_loader = DataLoader(MNIST_train_BNN, batch_size=500,
#                         shuffle=True)

# MNIST_test_BNN=GroupMNIST(X_test_BNN_tensor,Y_test_BNN_tensor,group=3)
# MNIST_test_BNN_loader = DataLoader(MNIST_test_BNN, batch_size=500,
#                         shuffle=True)



################################################################################

MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=10,act_func='ReLU')

length=len(File_Path_psgld_list)
avg_time_psgld=12000/12000
Acc_psgld_avg=[]
NLL_psgld_avg=[]
for ind in range(length):
    print('Current ind:%s'%(ind+1))
    FilePath_psgld=File_Path_psgld_list[ind]
    state_list_psgld=ReadStoredSamples(FilePath_psgld)
    
    len_psgld=len(state_list_psgld)
    Acc_psgld_list=np.zeros(len_psgld)
    NLL_psgld_list=np.zeros(len_psgld)
    time_list_psgld=avg_time_psgld*100/120*np.linspace(0,len_psgld,len_psgld)
    
    for ind,state_psgld in zip(range(len_psgld),state_list_psgld):
        if (ind+1)%20==0:
            print('%s'%(ind+1))
        Acc_psgld,NLL_psgld=Test_accuracy(MNIST_test_BNN_loader,MLP_mnist,state_psgld,data_number=float(len(MNIST_test_BNN)))
        Acc_psgld_list[ind]=1.-Acc_psgld
        NLL_psgld_list[ind]=-NLL_psgld.data.cpu().numpy()
        
    Acc_psgld_avg.append(Acc_psgld_list)
    NLL_psgld_avg.append(NLL_psgld_list)

    
All_Acc_psgld=np.stack(tuple(Acc_psgld_avg),axis=0)
All_NLL_psgld=np.stack(tuple(NLL_psgld_avg),axis=0)
Acc_psgld=np.mean(np.stack(tuple(Acc_psgld_avg),axis=0),axis=0)
NLL_psgld=np.mean(np.stack(tuple(NLL_psgld_avg),axis=0),axis=0)
np.savetxt('./ReLU_Generalization_Long_Run/Acc_Avg_psgld_step_1.4_TEST',Acc_psgld)
np.savetxt('./ReLU_Generalization_Long_Run/NLL_Avg_psgld_step_1.4_TEST',NLL_psgld)
np.savetxt('./ReLU_Generalization_Long_Run/All_Acc_psgld_step_1.4_TEST',All_Acc_psgld)
np.savetxt('./ReLU_Generalization_Long_Run/All_NLL_psgld_step_1.4_TEST',All_NLL_psgld)
Acc_psgld_std=np.std(np.stack(tuple(Acc_psgld_avg),axis=0),axis=0)
NLL_psgld_std=np.std(np.stack(tuple(NLL_psgld_avg),axis=0),axis=0)
np.savetxt('./ReLU_Generalization_Long_Run/Acc_Std_psgld_step_1.4_TEST',Acc_psgld_std)
np.savetxt('./ReLU_Generalization_Long_Run/NLL_Std_psgld_step_1.4_TEST',NLL_psgld_std)
    

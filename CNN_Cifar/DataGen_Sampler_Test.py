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
########################### Define the initial Data Loader #############################################################
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
########################## Define the Parameter ########################################################################
Type='NNSGHMC Test'
#15768 and 61478
data_N=25000.
eps1=float(np.sqrt(0.003/data_N))
eps2=float(np.sqrt(0.003/data_N))
Scale_Q_D=float(0.005/eps2)
coef=15768./61478.
Param_NNSGHMC=GenerateParameters('NNSGHMC Test',
                                Random_Seed=[10,11,12,13,14,15,16,17,18,19],
                                Step_Size_1=eps1,
Step_Size_2=eps2,
Offset_Q=0,
Scale_Q_D=Scale_Q_D,
Scale_D=30,
Scale_G=100,
Offset_D=0,
Epoch=200,
Sample_Interval=5000,
Mom_Resample=10000000,
Num_CNN=20,
Noise_Estimation=0,
Sigma=22.,
Clamp_Q=7.,
Clamp_D_Min=0.,
Clamp_D_Max=1000,
Batch_Size=500,
Coef=coef,
                                 Test_Interval=50
                            )

Param_SGHMC=GenerateParameters(Type='SGHMC',Step_Size=0.003,Beta=0.,Batch_Size=500,Epoch=200,Random_Seed=[10,11,12,13,14,15,16,17,18,19],Num_Run=10,Sigma=22.,Num_CNN=20,
                               Mom_Resample=10000000,Alpha=0.01,Interval=5000,Test_Interval=50,Test_Mode='Cross Chain')
Param_SGLD=GenerateParameters(Type='SGLD',Step_Size=0.15,Beta=0.,Batch_Size=500,Epoch=200,Random_Seed=[10,11,12,13,14,15,16,17,18,19],Num_Run=10,Sigma=22.,Num_CNN=20,Mom_Resample=1000000000,Interval=5000,
                              Test_Interval=50,Test_Mode='Cross Chain')
Param_PSGLD=GenerateParameters(Type='PSGLD',Step_Size=1.4e-3,Beta=0.,Batch_Size=500,Epoch=200,Random_Seed=[10,11,12,13,14,15,16,17,18,19],Num_Run=10,Sigma=22,Num_CNN=20,Interval=5000,Test_Interval=50,Exp_Term=0.99)


######################## Define own data-loader ####################
X_train_sampler_tensor,X_train_CNN_tensor,Y_train_sampler_tensor,Y_train_CNN_tensor,X_test_CNN_tensor,Y_test_CNN_tensor,X_test_sampler_tensor,\
Y_test_sampler_tensor=SelectImage_DataGen(trainset,testset,train_image=[0,4],test_image=[5,9])
# Define Splited dataset
Sampler_train_data=GroupCifar(X_train_sampler_tensor,Y_train_sampler_tensor,group=1)
Sampler_train_loader=DataLoader(Sampler_train_data, batch_size=500,
                        shuffle=True)
Sampler_test_data=GroupCifar(X_test_sampler_tensor,Y_test_sampler_tensor,group=4)
Sampler_test_loader=DataLoader(Sampler_test_data, batch_size=500,
                        shuffle=True)
CNN_train_data=GroupCifar(X_train_CNN_tensor,Y_train_CNN_tensor,group=2)
CNN_train_loader=DataLoader(CNN_train_data, batch_size=500,
                        shuffle=True)
CNN_test_data=GroupCifar(X_test_CNN_tensor,Y_test_CNN_tensor,group=3)
CNN_test_loader=DataLoader(CNN_test_data, batch_size=500,
                        shuffle=True)
##################################################################
############################################### Start Testing #####################################################
# For each run
for n_r in range(10):
    timestr = time.strftime("%Y%m%d-%H%M%S")
# Run each Sampler
    for i in range(1):
        print('Num Run:%s'%(n_r))
        i=0
        if i==0:
            Param=Param_NNSGHMC
        elif i==1:
            Param=Param_SGHMC
        elif i==2:
            Param=Param_PSGLD
        elif i==3:
            Param=Param_SGLD
        ######################## Define own data-loader ####################
        X_train_sampler_tensor, X_train_CNN_tensor, Y_train_sampler_tensor, Y_train_CNN_tensor, X_test_CNN_tensor, Y_test_CNN_tensor, X_test_sampler_tensor, \
        Y_test_sampler_tensor = SelectImage_DataGen(trainset, testset, train_image=[0, 4], test_image=[5, 9])
        # Define Splited dataset
        Sampler_train_data = GroupCifar(X_train_sampler_tensor, Y_train_sampler_tensor, group=1)
        Sampler_train_loader = DataLoader(Sampler_train_data, batch_size=500,
                                          shuffle=True)
        Sampler_test_data = GroupCifar(X_test_sampler_tensor, Y_test_sampler_tensor, group=4)
        Sampler_test_loader = DataLoader(Sampler_test_data, batch_size=500,
                                         shuffle=True)
        CNN_train_data = GroupCifar(X_train_CNN_tensor, Y_train_CNN_tensor, group=2)
        CNN_train_loader = DataLoader(CNN_train_data, batch_size=500,
                                      shuffle=True)
        CNN_test_data = GroupCifar(X_test_CNN_tensor, Y_test_CNN_tensor, group=3)
        CNN_test_loader = DataLoader(CNN_test_data, batch_size=500,
                                     shuffle=True)
        if Param['Type']=='NNSGHMC Test':
        ##################### Define parameters #####################
            print('Sampling From NNSGHMC')
            model_str='20180923-074531_100'
            model_load_path='./tmp_model_save/'
            torch.manual_seed(Param['Random Seed'][n_r])
            num_CNN=Param['Num CNN']
            CNN=Parallel_CNN(num_CNN=num_CNN,fc_size=100,out_channel=16,flat_size=16*6*6,out_dim=5)
            Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
            D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
            total_dim=CNN.get_dimension()
            B=Param['Noise Estimation']
            Q=parallel_Q_eff(CNN,Q_MLP,clamp=Param['Clamp Q'],offset=Param['Offset Q'])
            D=parallel_D_eff(CNN,D_MLP,Param['Scale G'],Param['Scale D'],Param['Offset D'],Param['Clamp D Min'],Param['Clamp D Max'],
                             Param['Scale Q D'])
            Gamma=parallel_Gamma_eff(flag_D=True)

            # Load stored model
            Q_MLP.load_state_dict(torch.load('%s/DataGen_Q_MLP_%s'%(model_load_path,model_str)))
            D_MLP.load_state_dict(torch.load('%s/DataGen_D_MLP_%s'%(model_load_path,model_str)))

            # Define Sampler
            NNSGHMC_obj=NNSGHMC(CNN,Q,D,Gamma)
            # Define Training parameters
            epoch=Param['Epoch']
            eps1=Param['Step Size 1']
            eps2=Param['Step Size 2']
            sigma=Param['Sigma']
            coef=Param['Coef']


            # Training
            weight_init = 0.05 * torch.randn(num_CNN, total_dim,requires_grad=True)
            state_mom_init=0.* torch.randn(num_CNN,total_dim,requires_grad=True)

            state_list,state_mom_list,_,Acc_list,NLL_list=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,CNN_train_loader,data_N,
                                                                    sigma,num_CNN,epoch,1000000000,eps1,eps2,
                                                                    1000000,coef,Param['Sample Interval'],0,Param['Mom Resample'],
                                                                    False,CNN_test_loader,5000.,test_interval=Param['Test Interval'],CNN_out_dim=5)
            Acc_list_np=np.asarray(Acc_list)
            NLL_list_np=np.asarray(NLL_list)
            # Save Results
            np.savetxt('./Results/DG_NNSGHMC_ep_%s_%s_%s_Acc'%(Param['Epoch'],model_str,timestr),Acc_list_np)
            np.savetxt('./Results/DG_NNSGHMC_ep_%s_%s_%s_NLL'%(Param['Epoch'],model_str,timestr),NLL_list_np)
            write_Dict('./Results/DG_Param_NNSGHMC_ep_%s_%s_%s'%(Param['Epoch'],model_str,timestr), Param)

        elif Param['Type']=='SGHMC':
            print('SGHMC Test')
            torch.manual_seed(Param['Random Seed'][n_r])
            ##Define initial parameters
            eps = Param['Step Size'] / 50000.
            alpha = Param['Alpha']
            beta = Param['Beta']
            epoch = Param['Epoch']
            num_runs = Param['Num Run']
            sigma = Param['Sigma']
            num_CNN = Param['Num CNN']
            mom_resample = Param['Mom Resample']
            interval = Param['Interval']

            # Define CNN network
            CNN = Parallel_CNN(num_CNN=num_CNN, fc_size=100,out_dim=5)
            total_dim = CNN.get_dimension()
            # Define Sampler
            SGHMC_obj = SGHMC(total_dim, CNN)
            # Initialize the weight and momentum
            state_pos = 0.05 * torch.randn(num_CNN, total_dim, requires_grad=True)
            state_mom = 0. * torch.randn(num_CNN, total_dim)
            #Sequential_Accuracy_obj = Sequential_Accuracy(test_loader_seq, CNN)
            state_list,Acc_list,NLL_list = SGHMC_obj.parallel_sample(state_pos, state_mom, CNN_train_loader, data_N=data_N, num_CNN=num_CNN,
                                                   mom_resample=mom_resample,
                                                   total_step=epoch, eps=eps, alpha=alpha, beta=beta, sigma=sigma,
                                                   interval=interval, flag_SGLD=False,
                                                   test_loader=CNN_test_loader, data_len=5000.,
                                                   Sequential_Accuracy=None,test_interval=Param['Test Interval'],test_mode=Param['Test Mode'],CNN_out_dim=5)
            Acc_list_np = np.asarray(Acc_list)
            NLL_list_np = np.asarray(NLL_list)
            # Save Results
            np.savetxt('./Results/DG_SGHMC_ep_%s_%s_Acc' % (Param['Epoch'],timestr), Acc_list_np)
            np.savetxt('./Results/DG_SGHMC_ep_%s_%s_NLL' % (Param['Epoch'],timestr), NLL_list_np)
            write_Dict('./Results/DG_Param_SGHMC_ep_%s_%s' % (Param['Epoch'],timestr), Param)
        elif Param['Type']=='SGLD':
            print('SGLD Sampling')
            torch.manual_seed(Param['Random Seed'][n_r])
            ##Define initial parameters
            eps = Param['Step Size'] / 50000.
            beta = Param['Beta']
            epoch = Param['Epoch']
            num_runs = Param['Num Run']
            sigma = Param['Sigma']
            num_CNN = Param['Num CNN']
            interval = Param['Interval']

            # Define CNN network
            CNN = Parallel_CNN(num_CNN=num_CNN, fc_size=100,out_dim=5)
            total_dim = CNN.get_dimension()
            # Define Sampler
            SGHMC_obj = SGHMC(total_dim, CNN)
            # Initialize the weight and momentum
            state_pos = 0.05 * torch.randn(num_CNN, total_dim, requires_grad=True)
            state_mom = 0. * torch.randn(num_CNN, total_dim)
            #Sequential_Accuracy_obj = Sequential_Accuracy(test_loader_seq, CNN)
            state_list, Acc_list, NLL_list = SGHMC_obj.parallel_sample(state_pos, state_mom, CNN_train_loader, data_N=data_N,
                                                                       num_CNN=num_CNN,
                                                                       mom_resample=100000000,
                                                                       total_step=epoch, eps=eps, alpha=1., beta=beta,
                                                                       sigma=sigma,
                                                                       interval=interval, flag_SGLD=True,
                                                                       test_loader=CNN_test_loader, data_len=5000.,
                                                                       Sequential_Accuracy=None,
                                                                       test_interval=Param['Test Interval'],
                                                                       test_mode=Param['Test Mode'])
            Acc_list_np = np.asarray(Acc_list)
            NLL_list_np = np.asarray(NLL_list)
            # Save Results
            np.savetxt('./Results/DG_SGLD_ep_%s_%s_Acc_10' % (Param['Epoch'], timestr), Acc_list_np)
            np.savetxt('./Results/DG_SGLD_ep_%s_%s_NLL_10' % (Param['Epoch'], timestr), NLL_list_np)
            write_Dict('./Results/DG_Param_SGLD_ep_%s_%s_10' % (Param['Epoch'], timestr), Param)
        elif Param['Type']=='PSGLD':
            print('PSGLD Sampling')
            torch.manual_seed(Param['Random Seed'][n_r])
            ##Define initial parameters
            eps = Param['Step Size'] / 50000.
            beta = Param['Beta']
            epoch = Param['Epoch']
            num_runs = Param['Num Run']
            sigma = Param['Sigma']
            num_CNN = Param['Num CNN']
            interval = Param['Interval']

            # Define CNN network
            CNN = Parallel_CNN(num_CNN=num_CNN, fc_size=100,out_dim=5)
            total_dim = CNN.get_dimension()
            # Define Sampler
            PSGLD_obj = PSGLD(CNN,total_dim)
            # Initialize the weight and momentum
            state_pos = 0.05 * torch.randn(num_CNN, total_dim, requires_grad=True)
            #Sequential_Accuracy_obj = Sequential_Accuracy(test_loader_seq, CNN)
            state_list, Acc_list, NLL_list = PSGLD_obj.parallel_sample(state_pos,CNN_train_loader,data_N,num_chain=num_CNN,total_step=epoch,eps=eps,
                                                                       exp_term=Param['Exp Term'],lamb=1e-5,sigma=sigma,interval=interval,
                                                                       test_loader=CNN_test_loader,data_len=5000.,test_interval=Param['Test Interval'])
            Acc_list_np = np.asarray(Acc_list)
            NLL_list_np = np.asarray(NLL_list)
            # Save Results
            np.savetxt('./Results/PSGLD_ep_%s_%s_Acc' % (Param['Epoch'], timestr), Acc_list_np)
            np.savetxt('./Results/PSGLD_ep_%s_%s_NLL' % (Param['Epoch'], timestr), NLL_list_np)
            write_Dict('./Results/Param_PSGLD_ep_%s_%s' % (Param['Epoch'], timestr), Param)



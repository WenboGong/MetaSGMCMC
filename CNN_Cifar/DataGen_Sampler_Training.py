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
###################### Define parameters ################################
data_N=25000.
eps1=float(np.sqrt(0.0007/data_N))
eps2=float(np.sqrt(0.0007/data_N))
Scale_Q_D=float(0.005/eps2)
CNN_out_dim=5
Param_NNSGHMC_Training=GenerateParameters('NNSGHMC Training',
                                          Random_Seed=10,
                                          Optimizer_Step_Size=0.002,
                                          Optimizer_Betas=(0.9,0.99),
                                          Step_Size_1=eps1,
                                          Step_Size_2=eps2,
                                          Offset_Q=0.,
                                          Scale_Q_D=Scale_Q_D,
                                          Scale_G=100,
                                          Scale_D=30,
                                          Offset_D=0,
                                          Training_Epoch=100,
                                          Sub_Epoch=10,
                                          Limit_Step=50,
                                          TBPTT=20,
                                          Sample_Interval=3,
                                          Mom_Resample=1000000,
                                          Num_CNN=50,
                                          Noise_Estimation=0.,
                                          Sigma=22.,
                                          Clamp_Q=5,
                                          Clamp_D_Min=0.,
                                          Clamp_D_Max=10000,
                                          Batch_Size=500,
                                          Saving_Interval=10,
                                          Roll_Out=0.15,
                                          Flag_Single_Roll_Out=True,
                                          Sub_Sample_Num=5, # if 0, need to turn off the in Chain loss in CNN_Sampler.py
                                          Roll_Out_Mom_Resample=False,
                                          Flag_In_Chain=True,
                                          Scale_Entropy=1.,
                                          Eps_Training=False,
                                          Test_Interval=10



                                          )
Param=Param_NNSGHMC_Training
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
##################### Define parameters #####################
# Set random seed
torch.manual_seed(Param['Random Seed'])
np.random.seed(Param['Random Seed'])

num_CNN=Param['Num CNN']
CNN=Parallel_CNN(num_CNN=num_CNN,fc_size=50,out_channel=8,flat_size=8*6*6,out_dim=CNN_out_dim)
Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
total_dim=CNN.get_dimension()
B=Param['Noise Estimation']
Q=parallel_Q_eff(CNN,Q_MLP,clamp=Param['Clamp Q'],offset=Param['Offset Q'])
D=parallel_D_eff(CNN,D_MLP,Param['Scale G'],Param['Scale D'],Param['Offset D'],Param['Clamp D Min'],Param['Clamp D Max'],
                 Param['Scale Q D'])
Gamma=parallel_Gamma_eff(flag_D=True)

# Define Optimizer
Adam_Q=torch.optim.Adam(list(Q_MLP.parameters()),lr=Param['Optimizer Step Size'],betas=Param['Optimizer Betas'])
Adam_D=torch.optim.Adam(list(D_MLP.parameters()),lr=Param['Optimizer Step Size'],betas=Param['Optimizer Betas'])
# Define Sampler
NNSGHMC_obj=NNSGHMC(CNN,Q,D,Gamma)
# Define Training parameters
epoch=Param['Training Epoch']
sub_epoch=Param['Sub Epoch']
eps1=Param['Step Size 1']
eps2=Param['Step Size 2']
sigma=Param['Sigma']
coef=1.
p_roll_out=Param['Roll Out']

state_pos_rep=collections.deque(maxlen=50)
state_mom_rep=collections.deque(maxlen=50)
Acc_rep=collections.deque(maxlen=50)
############################## Training Algorithm ####################################
for ep in range(epoch):
    print('Initial DataGen Training of epoch %s'%(ep+1))
    Adam_Q.zero_grad()
    Adam_D.zero_grad()
    #flag_roll_out=roll_out(p_roll_out)
    # Initialization
    # if flag_roll_out and len(state_pos_rep)!=0:
    #     # Roll-out
    #
    #     ind=np.random.randint(0,len(state_pos_rep))
    #     print('Roll Out with Starting Acc:%s and NLL:%s'%(Acc_rep[ind][0],Acc_rep[ind][1]))
    #     weight_init=torch.tensor(state_pos_rep[ind],requires_grad=True)
    #     if Param['Roll Out Mom Resample']==False:
    #         state_mom_init=torch.tensor(state_mom_rep[ind],requires_grad=True)
    #     else:
    #         state_mom_init=0*torch.randn(num_CNN,total_dim,requires_grad=True)
    # else:
    #     weight_init = 0.1 * torch.randn(num_CNN, total_dim,requires_grad=True)
    #     state_mom_init=0.* torch.randn(num_CNN,total_dim,requires_grad=True)



    # Multi Roll Out
    if len(state_pos_rep)!=0:
        weight_init,state_mom_init,total_num_replay,state_ind,ind_chain=roll_out_multi(p_roll_out,state_pos_rep,state_mom_rep,num_CNN,total_dim)
        print('Replay Num:%s' % (total_num_replay))
        state_ind=[int(i) for i in state_ind]
        #print('Roll Out Acc:%s'%([x for i,x in enumerate(Acc_rep) if i in state_ind]))
        print('Roll Out Acc:%s'%([Acc_rep[i] for i in state_ind]))
        ind_chain=[int(i) for i in ind_chain]
    else:
        weight_init = 0.1 * torch.randn(num_CNN, total_dim, requires_grad=True)
        state_mom_init=0.* torch.randn(num_CNN,total_dim,requires_grad=True)
        ind_chain=None
    if Param['Roll Out Mom Resample']:
        state_mom_init=0*torch.randn(num_CNN,total_dim,requires_grad=True)

    # Drawing samples
    state_list,state_mom_list,counter_ELBO,Acc_list,NLL_list=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,Sampler_train_loader,data_N,
                                                        sigma,num_CNN,sub_epoch,Param['Limit Step'],eps1,eps2,
                                                        Param['TBPTT'],coef,Param['Sample Interval'],Param['Sub Sample Num'],Param['Mom Resample'],
                                                        True,None,5000.,Param['Flag In Chain'],show_ind=ind_chain,scale_entropy=Param['Scale Entropy'],test_interval=Param['Test Interval'],CNN_out_dim=CNN_out_dim)

    # Store in the rep
    state_pos_rep.append(torch.tensor(state_list[-1].data))
    state_mom_rep.append(torch.tensor(state_mom_list[-1].data))
    # Modify gradient
    modify_grad(counter_ELBO, Q_MLP, D_MLP)
    Err, NLL = Test_Accuracy(CNN,Sampler_test_loader,state_list[-1],5000.,False)
    print('Sampler Test Err:%s Test NLL:%s' % (Err, NLL.cpu().data.numpy()))
    Acc_rep.append((Err,NLL.cpu().data.numpy()))
    # Update Parameter
    Adam_Q.step()
    Adam_D.step()

    # Sub Epoch Training

    for ep_se in range(sub_epoch):
        print('         DataGen Continuous Training:%s' % (ep_se + 1))
        Adam_Q.zero_grad()
        Adam_D.zero_grad()
        # ONe Coin Roll Out
        # flag_roll_out = roll_out(p_roll_out)
        #
        # if Param['Flag Single Roll Out']:
        #     # Single Roll Out
        #     flag_roll_out=False
        # if flag_roll_out:
        #     ind = np.random.randint(0, len(state_pos_rep))
        #     print('Roll Out with starting Acc:%s and NLL:%s'%(Acc_rep[ind][0],Acc_rep[ind][1]))
        #     weight_init = torch.tensor(state_pos_rep[ind], requires_grad=True)
        #     if Param['Roll Out Mom Resample']==False:
        #         state_mom_init = torch.tensor(state_mom_rep[ind], requires_grad=True)
        #     else:
        #         state_mom_init=0*torch.randn(num_CNN,total_dim,requires_grad=True)
        #
        # else:
        #
        #     state_mom_init = Variable(state_mom_list[-1].data, requires_grad=True)
        #     weight_init = Variable(state_list[-1].data, requires_grad=True)


        if Param['Flag Single Roll Out']:
            state_mom_init = Variable(state_mom_list[-1].data, requires_grad=True)
            weight_init = Variable(state_list[-1].data, requires_grad=True)
        else:

            weight_init, state_mom_init,total_num_replay,state_ind,ind_chain = roll_out_multi(p_roll_out, state_pos_rep, state_mom_rep, num_CNN,
                                                            total_dim)
            print('Replay Num:%s' % (total_num_replay))
            state_ind = [int(i) for i in state_ind]
            #print('Roll Out Acc:%s' % ([x for i, x in enumerate(Acc_rep) if i in state_ind]))
            print('Roll Out Acc:%s' % ([Acc_rep[i] for i in state_ind]))
            ind_chain = [int(i) for i in ind_chain]
            if Param['Roll Out Mom Resample']:
                state_mom_init = 0 * torch.randn(num_CNN, total_dim, requires_grad=True)

        state_list, state_mom_list, counter_ELBO,Acc_list,NLL_list = NNSGHMC_obj.parallel_sample(weight_init, state_mom_init, B,
                                                                               Sampler_train_loader, data_N,
                                                                               sigma, num_CNN, sub_epoch,
                                                                               Param['Limit Step'], eps1, eps2,
                                                                               Param['TBPTT'], coef,
                                                                               Param['Sample Interval'],
                                                                               Param['Sub Sample Num'],
                                                                               Param['Mom Resample'],
                                                                               True, None, 5000.,Param['Flag In Chain'],show_ind=ind_chain,scale_entropy=Param['Scale Entropy'],test_interval=Param['Test Interval'])

        # Store in the rep
        state_pos_rep.append(torch.tensor(state_list[-1].data))
        state_mom_rep.append(torch.tensor(state_mom_list[-1].data))
        # Modify Gradient
        modify_grad(counter_ELBO, Q_MLP, D_MLP)
        Err, NLL = Test_Accuracy(CNN, Sampler_test_loader, state_list[-1], 5000., False)
        print('Test Err:%s Test NLL:%s' % (Err, NLL.cpu().data.numpy()))
        Acc_rep.append((Err, NLL.cpu().data.numpy()))
        # Update Parameter
        Adam_Q.step()
        Adam_D.step()
    # Save trained model
    if (ep+1)%Param['Saving Interval']==0:
        torch.save(Q_MLP.state_dict(),'./tmp_model_save/DataGen_Q_MLP_%s_%s'%(timestr,(ep+1)))
        torch.save(D_MLP.state_dict(),'./tmp_model_save/DataGen_D_MLP_%s_%s'%(timestr,(ep+1)))
        if (ep+1)==Param['Saving Interval']:
            write_Dict('./tmp_model_save/DataGen_Param_%s'%(timestr), Param)

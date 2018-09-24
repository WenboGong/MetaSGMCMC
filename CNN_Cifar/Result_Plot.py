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

def DQContour(Q_MLP, D_MLP, dim=[0, 1], range1=[-2.5, 2.5], range2=[-2.5, 2.5], range3=-0.1, num=200, flag_D=False,
          flag_energy=False):
    if flag_D == False:
        x1 = np.linspace(range1[0], range1[1], num)
        x2 = np.linspace(range2[0], range2[1], num)
        X1, X2 = np.meshgrid(x1, x2)
        X_grid = np.stack((-X1, X2), axis=2)
        X_tensor = torch.tensor(torch.from_numpy(X_grid).float().data).cuda()
        out = np.squeeze(np.abs(Q_MLP.forward(X_tensor).cpu().data.numpy()))

        return X1, X2, out
    if flag_D == True:
        if flag_energy == True:
            x1 = np.linspace(range1[0], range1[1], num)
        else:
            x1 = np.linspace(range1[0], range1[1], num)

        x2 = np.linspace(range2[0], range2[1], num)
        X1, X2 = np.meshgrid(x1, x2)
        X3 = np.tile(range3, X1.shape)
        # Tracer()()
        if dim == [0, 1]:
            X_grid = np.stack((-X1, X2, X3), axis=2)
        elif dim == [0, 2]:
            X_grid = np.stack((-X1, X3, X2), axis=2)
        elif dim == [1, 2]:
            X_grid = np.stack((X3, X1, X2), axis=2)

        X_tensor = Variable(torch.from_numpy(X_grid).float().cuda())

        out = np.squeeze(np.abs(10 * D_MLP.forward(X_tensor).cpu().data.numpy()))
        return X1, X2, out
def Load_Results(Results_Path,**kwargs):
    Path_Dict={}
    Path_Dict['NNSGHMC']=kwargs['NNSGHMC']
    Path_Dict['SGHMC']=kwargs['SGHMC']
    Path_Dict['SGLD']=kwargs['SGLD']
    Path_Dict['PSGLD']=kwargs['PSGLD']
    Path_Dict['Adam']=kwargs['Adam']
    Path_Dict['SGDM']=kwargs['SGDM']
    Results_Dict={}
    for key,value in Path_Dict.items():
        if type(value)==type(None):
            continue
        elif key=='SGHMC' or key=='NNSGHMC':
            length=len(value)
            for i in range(length):
                if i==0:
                    Acc=np.expand_dims(np.loadtxt('%s%s%s'%(Results_Path,value['%s'%(i)],'Acc')),axis=0)
                    NLL=np.expand_dims(np.loadtxt('%s%s%s'%(Results_Path,value['%s'%(i)],'NLL')),axis=0)
                else:
                    Acc_item=np.expand_dims(np.loadtxt('%s%s%s'%(Results_Path,value['%s'%(i)],'Acc')),axis=0)
                    NLL_item=np.expand_dims(np.loadtxt('%s%s%s'%(Results_Path,value['%s'%(i)],'NLL')),axis=0)
                    Acc=np.concatenate((Acc,Acc_item),axis=0)
                    NLL=np.concatenate((NLL,NLL_item),axis=0)
            Results_Dict[key]={}
            Results_Dict[key]['Acc']=Acc
            Results_Dict[key]['NLL']=NLL
        elif key=='SGLD' or key=='PSGLD':
            length = len(value)
            for i in range(length):
                if i == 0:
                    Acc = np.expand_dims(np.loadtxt('%s%s%s' % (Results_Path, value['%s' % (i)], 'Acc_10')), axis=0)
                    NLL = np.expand_dims(np.loadtxt('%s%s%s' % (Results_Path, value['%s' % (i)], 'NLL_10')), axis=0)
                else:
                    Acc_item = np.expand_dims(np.loadtxt('%s%s%s' % (Results_Path, value['%s' % (i)], 'Acc_10')),axis=0)
                    NLL_item = np.expand_dims(np.loadtxt('%s%s%s' % (Results_Path, value['%s' % (i)], 'NLL_10')),axis=0)
                    Acc = np.concatenate((Acc, Acc_item), axis=0)
                    NLL = np.concatenate((NLL, NLL_item), axis=0)
            Results_Dict[key] = {}
            Results_Dict[key]['Acc'] = Acc
            Results_Dict[key]['NLL'] = NLL
    return Results_Dict
def Avg(Results_item):
    return np.mean(Results_item,axis=0)
def Std(Results_item): # Std Error
    return np.std(Results_item,axis=0)/np.sqrt(float(Results_item.shape[0]))

######################################### Define Plot Parameters #####################################
Flag_Contour=False # This is to illustrate the strategy learned by D and Q (for debug purpose)

model_str='20180920-143635_100'
model_load_path='./tmp_model_save/'
Results_Path='./Results/'

NNSGHMC_Path={}
SGHMC_Path={}
SGLD_Path={}
PSGLD_Path={}

interval=10
Interval=10*np.arange(200*interval)
Short_Range=30
Long_Start=50
Long_End=200
# Style Parameter #
plt.style.use('ggplot')


#### Define Path ########################
NNSGHMC_Path['0']='NNSGHMC_ep_200_20180920-143635_100_'
NNSGHMC_Path['1']='NNSGHMC_ep_200_20180920-143635_100_time_20180922-030815_'
NNSGHMC_Path['2']='NNSGHMC_ep_200_20180920-143635_100_time_20180922-085208_'
NNSGHMC_Path['3']='NNSGHMC_ep_200_20180920-143635_100_time_20180922-134108_'
NNSGHMC_Path['4']='NNSGHMC_ep_200_20180920-143635_100_20180923-152534_'

SGHMC_Path['0']='SGHMC_ep_200_20180921-065323_'
SGHMC_Path['1']='SGHMC_ep_200_20180921-142730_'
SGHMC_Path['2']='SGHMC_ep_200_20180922-030815_'
SGHMC_Path['3']='SGHMC_ep_200_20180922-085208_'
SGHMC_Path['4']='SGHMC_ep_200_20180922-134108_'

# For SGLD add _Acc_10 or _NLL_10
SGLD_Path['0']='SGLD_ep_200_20180923-213439_'
SGLD_Path['1']='SGLD_ep_200_20180923-223000_'
SGLD_Path['2']='SGLD_ep_200_20180923-232522_'
SGLD_Path['3']='SGLD_ep_200_20180924-002050_'
SGLD_Path['4']='SGLD_ep_200_20180924-011621_'






################################# Generate DQ Contour ##############################

if Flag_Contour==True:
    # Load Model
    Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
    D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)

    Q_MLP.load_state_dict(torch.load('%sQ_MLP_%s'%(model_load_path,model_str)))
    D_MLP.load_state_dict(torch.load('%sD_MLP_%s'%(model_load_path,model_str)))

    # Generate Contour
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(18,5))
    # Ax1 with Q Contour
    X1,X2,out=DQContour(Q_MLP,D_MLP,range1=[0,20],range2=[-2,2])
    CS1=ax1.contourf(X1,X2,out,50)
    fig.colorbar(CS1,ax=ax1)
    # Ax2 with D Contour 0,1 dim
    X1_D,X2_D,out_D=DQContour(Q_MLP,D_MLP,dim=[0,1],range1=[0,15],range2=[-2,2],range3=0.5,flag_D=True)
    CS2=ax2.contourf(X1_D,X2_D,out_D,50)
    fig.colorbar(CS2,ax=ax2)
    # Ax2 with D Contour 1,2 dim
    X1_D,X2_D,out_D=DQContour(Q_MLP,D_MLP,dim=[1,2],range1=[-2,2],range2=[-2,2],range3=-5,flag_D=True)
    CS3=ax3.contourf(X1_D,X2_D,out_D,50)
    fig.colorbar(CS3,ax=ax3)
    #plt.savefig('./Results/Countour_')


    # #X1,X2,out=DQContour(Q_MLP,D_MLP,range1=[0,15],range2=[-5,5])
    # X1_D,X2_D,out_D=DQContour(Q_MLP,D_MLP,dim=[0,1],range1=[4,15],range2=[-0.5,0.5],range3=0.3,flag_D=True)
    #
    # fig1,ax1=plt.subplots()
    #
    # #CS=ax1.contourf(X1,X2,out,50)
    # CS=ax1.contourf(X1_D,X2_D,out_D,50)
    # cbar=fig1.colorbar(CS)
    plt.savefig('./Results/Countour_%s.png'%(model_str))
else:
    ################## Loadd the data ########################
    Results_Dict=Load_Results(Results_Path,NNSGHMC=NNSGHMC_Path,SGHMC=SGHMC_Path,SGLD=SGLD_Path,PSGLD=None,Adam=None,SGDM=None)


    ###### Network Gen Short Term  #####
    f, ax = plt.subplots(2, 3, sharex='col', figsize=(20, 7))
    ax[0, 0].errorbar(Interval, 1-Avg(Results_Dict['NNSGHMC']['Acc']), color='b', yerr=Std(Results_Dict['NNSGHMC']['Acc']), linewidth=2,
                      label='NNSGHMC', errorevery=25, capsize=6, elinewidth=2.)
    ax[0, 0].errorbar(Interval, 1-Avg(Results_Dict['SGHMC']['Acc']), color='y', yerr=Std(Results_Dict['SGHMC']['Acc']), linewidth=2,
                      label='SGHMC', errorevery=25, capsize=6, elinewidth=2.)
    ax[0, 0].errorbar(Interval, 1-Avg(Results_Dict['SGLD']['Acc']), color='m', yerr=Std(Results_Dict['SGLD']['Acc']), linewidth=2, label='SGLD',
                      errorevery=25, capsize=6, elinewidth=2.)
    #ax[0, 0].errorbar(time_list_sgld, Acc_PSGLD_n, color='r', yerr=Std_PSGLD_n / np.sqrt(10), linewidth=2,
                      #label='PSGLD', errorevery=25, capsize=6, elinewidth=2.)
    ax[0, 0].legend(prop={'size': 17})
    ax[0,0].set_xlim(0,Short_Range*100)
    ax[0,0].set_yscale('log')
    ax[0, 0].set_ylim(0.237, 0.9)
    ax[0, 0].tick_params(labelsize=15)
    ax[0, 0].set_ylabel('Error', fontsize=17)
    ax[0, 0].set_title('Short Term Network Generalization', fontsize=15)

    ax[1, 0].errorbar(Interval, -Avg(Results_Dict['NNSGHMC']['NLL'])/100., color='b', yerr=Std(Results_Dict['NNSGHMC']['NLL'])/100., linewidth=2,
                      label='NNSGHMC', errorevery=25, capsize=6, elinewidth=2.)
    ax[1, 0].errorbar(Interval, -Avg(Results_Dict['SGHMC']['NLL'])/100., color='y', yerr=Std(Results_Dict['SGHMC']['NLL'])/100., linewidth=2,
                      label='SGHMC', errorevery=25, capsize=6, elinewidth=2.)
    ax[1, 0].errorbar(Interval, -Avg(Results_Dict['SGLD']['NLL'])/100., color='m', yerr=Std(Results_Dict['SGLD']['NLL'])/100., linewidth=2,
                      label='SGLD', errorevery=25, capsize=6, elinewidth=2.)
    # ax[1, 0].errorbar(time_list_sgld, NLL_PSGLD_n, color='r', yerr=Std_PSGLD_n_nl / np.sqrt(10), linewidth=2,
    #                   label='PSGLD', errorevery=25, capsize=6, elinewidth=2.)
    # ax[1,0].legend(prop={'size':17})

    ax[1, 0].set_ylim(67, 240)
    ax[1, 0].tick_params(labelsize=15)
    ax[1, 0].set_xlim(0, Short_Range*100)
    ax[1, 0].set_xlabel('Iter.', fontsize=15)
    ax[1, 0].set_ylabel('Neg. LL', fontsize=17)



    plt.savefig('./Results/Cifar-10_Results_Short.png')
################################################################################################################################################
    ###### Network Gen Long Term  #####
    fl, axl = plt.subplots(2, 3, sharex='col', figsize=(20, 7))
    axl[0, 0].errorbar(Interval, 1 - Avg(Results_Dict['NNSGHMC']['Acc']), color='b',
                      yerr=Std(Results_Dict['NNSGHMC']['Acc']), linewidth=2,
                      label='NNSGHMC', errorevery=25, capsize=6, elinewidth=2.)
    axl[0, 0].errorbar(Interval, 1 - Avg(Results_Dict['SGHMC']['Acc']), color='y',
                      yerr=Std(Results_Dict['SGHMC']['Acc']), linewidth=2,
                      label='SGHMC', errorevery=25, capsize=6, elinewidth=2.)
    axl[0, 0].errorbar(Interval, 1 - Avg(Results_Dict['SGLD']['Acc']), color='m', yerr=Std(Results_Dict['SGLD']['Acc']),
                      linewidth=2, label='SGLD',
                      errorevery=25, capsize=6, elinewidth=2.)
    # axl[0, 0].errorbar(time_list_sgld, Acc_PSGLD_n, color='r', yerr=Std_PSGLD_n / np.sqrt(10), linewidth=2,
    # label='PSGLD', errorevery=25, capsize=6, elinewidth=2.)
    axl[0, 0].legend(prop={'size': 17})
    axl[0, 0].set_xlim(Long_Start*100, Long_End * 100)
    axl[0, 0].set_yscale('log')
    axl[0, 0].set_ylim(0.217, 0.3)
    axl[0, 0].tick_params(labelsize=15)
    axl[0, 0].set_ylabel('Error', fontsize=17)
    axl[0, 0].set_title('Short Term Network Generalization', fontsize=15)

    axl[1, 0].errorbar(Interval, -Avg(Results_Dict['NNSGHMC']['NLL']) / 100., color='b',
                      yerr=Std(Results_Dict['NNSGHMC']['NLL']) / 100., linewidth=2,
                      label='NNSGHMC', errorevery=25, capsize=6, elinewidth=2.)
    axl[1, 0].errorbar(Interval, -Avg(Results_Dict['SGHMC']['NLL']) / 100., color='y',
                      yerr=Std(Results_Dict['SGHMC']['NLL']) / 100., linewidth=2,
                      label='SGHMC', errorevery=25, capsize=6, elinewidth=2.)
    axl[1, 0].errorbar(Interval, -Avg(Results_Dict['SGLD']['NLL']) / 100., color='m',
                      yerr=Std(Results_Dict['SGLD']['NLL']) / 100., linewidth=2,
                      label='SGLD', errorevery=25, capsize=6, elinewidth=2.)
    # axl[1, 0].errorbar(time_list_sgld, NLL_PSGLD_n, color='r', yerr=Std_PSGLD_n_nl / np.sqrt(10), linewidth=2,
    #                   label='PSGLD', errorevery=25, capsize=6, elinewidth=2.)
    # axl[1,0].legend(prop={'size':17})

    axl[1, 0].set_ylim(67, 90)
    axl[1, 0].tick_params(labelsize=15)
    axl[1, 0].set_xlim(Long_Start*100, Long_End * 100)
    axl[1, 0].set_xlabel('Iter.', fontsize=15)
    axl[1, 0].set_ylabel('Neg. LL', fontsize=17)



    plt.savefig('./Results/Cifar-10_Results_Long.png')
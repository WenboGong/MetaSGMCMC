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
import json
# Generate Parameters
def GenerateParameters(Type,**kwargs):
    '''
    Automatically generate dict containing experiment parameters

    :param Type: Str the model you want to run
    :param kwargs: The experiment parameters, at least include Batch_Size, Epoch, Random_Seed (list), Num_Run
    :return: Dict, include Parameters
    '''
    Param={}
    if Type=='Adam':
        Param['Type']=Type
        Param['Step Size']=kwargs['Step_Size']
        Param['Betas']=kwargs['Betas']
        Param['Batch Size']=kwargs['Batch_Size']
        Param['Epoch']=kwargs['Epoch']
        Param['Random Seed']=kwargs['Random_Seed']
        Param['Num Run']=kwargs['Num_Run']
        Param['Precision']=kwargs['Precision']
        Param['Test Interval']=kwargs['Test_Interval']
        assert len(Param['Random Seed'])==Param['Num Run'],'Number of runs are not consistent with length of random seeds'
    if Type=='SGDM':
        Param['Type']=Type
        Param['Step Size']=kwargs['Step_Size']
        Param['Momentum']=kwargs['Momentum']
        Param['Batch Size']=kwargs['Batch_Size']
        Param['Epoch']=kwargs['Epoch']
        Param['Random Seed']=kwargs['Random_Seed']
        Param['Num Run']=kwargs['Num_Run']
        Param['Precision']=kwargs['Precision']
        Param['Test Interval']=kwargs['Test_Interval']
        assert len(Param['Random Seed'])==Param['Num Run'],'Number of runs are not consistent with length of random seeds'
    if Type=='SGHMC':
        Param['Type'] = Type
        Param['Step Size'] = kwargs['Step_Size']
        Param['Beta'] = kwargs['Beta']
        Param['Batch Size'] = kwargs['Batch_Size']
        Param['Epoch'] = kwargs['Epoch']
        Param['Random Seed'] = kwargs['Random_Seed']
        Param['Num Run'] = kwargs['Num_Run']
        Param['Sigma'] = kwargs['Sigma']
        Param['Num CNN']=kwargs['Num_CNN']
        Param['Mom Resample']=kwargs['Mom_Resample']
        Param['Alpha']=kwargs['Alpha']
        Param['Interval']=kwargs['Interval']
        Param['Test Interval']=kwargs['Test_Interval']
        Param['Test Mode']=kwargs['Test_Mode']
        assert len(Param['Random Seed']) == Param['Num Run'], 'Number of runs are not consistent with length of random seeds'
    if Type=='NNSGHMC Training':
        Param['Type']=Type
        Param['Random Seed']=kwargs['Random_Seed']
        Param['Optimizer Step Size']=kwargs['Optimizer_Step_Size']
        Param['Optimizer Betas']=kwargs['Optimizer_Betas']
        Param['Step Size 1']=kwargs['Step_Size_1']
        Param['Step Size 2']=kwargs['Step_Size_2']
        Param['Offset Q']=kwargs['Offset_Q']
        Param['Scale Q D']=kwargs['Scale_Q_D']
        Param['Scale D']=kwargs['Scale_D']
        Param['Scale G']=kwargs['Scale_G']
        Param['Offset D']=kwargs['Offset_D']
        Param['Training Epoch']=kwargs['Training_Epoch']
        Param['Sub Epoch']=kwargs['Sub_Epoch']
        Param['Limit Step']=kwargs['Limit_Step']
        Param['TBPTT']=kwargs['TBPTT']
        Param['Sample Interval']=kwargs['Sample_Interval']
        Param['Mom Resample']=kwargs['Mom_Resample']
        Param['Num CNN']=kwargs['Num_CNN']
        Param['Noise Estimation']=kwargs['Noise_Estimation']
        Param['Sigma']=kwargs['Sigma']
        Param['Clamp Q']=kwargs['Clamp_Q']
        Param['Clamp D Min']=kwargs['Clamp_D_Min']
        Param['Clamp D Max']=kwargs['Clamp_D_Max']
        Param['Batch Size']=kwargs['Batch_Size']
        Param['Saving Interval']=kwargs['Saving_Interval']
        Param['Roll Out']=kwargs['Roll_Out']
        Param['Flag Single Roll Out']=kwargs['Flag_Single_Roll_Out']
        Param['Sub Sample Num']=kwargs['Sub_Sample_Num']
        Param['Roll Out Mom Resample']=kwargs['Roll_Out_Mom_Resample']
        Param['Flag In Chain']=kwargs['Flag_In_Chain']
        Param['Scale Entropy']=kwargs['Scale_Entropy']
        Param['Eps Training']=kwargs['Eps_Training']
        Param['Test Interval']=kwargs['Test_Interval']
    if Type=='NNSGHMC Test':
        Param['Type'] = Type
        Param['Random Seed'] = kwargs['Random_Seed']
        Param['Step Size 1'] = kwargs['Step_Size_1']
        Param['Step Size 2'] = kwargs['Step_Size_2']
        Param['Offset Q'] = kwargs['Offset_Q']
        Param['Scale Q D'] = kwargs['Scale_Q_D']
        Param['Scale D'] = kwargs['Scale_D']
        Param['Scale G'] = kwargs['Scale_G']
        Param['Offset D'] = kwargs['Offset_D']
        Param['Epoch'] = kwargs['Epoch']
        Param['Sample Interval'] = kwargs['Sample_Interval']
        Param['Mom Resample'] = kwargs['Mom_Resample']
        Param['Num CNN'] = kwargs['Num_CNN']
        Param['Noise Estimation'] = kwargs['Noise_Estimation']
        Param['Sigma'] = kwargs['Sigma']
        Param['Clamp Q'] = kwargs['Clamp_Q']
        Param['Clamp D Min'] = kwargs['Clamp_D_Min']
        Param['Clamp D Max'] = kwargs['Clamp_D_Max']
        Param['Batch Size'] = kwargs['Batch_Size']
        Param['Coef']=kwargs['Coef']
        Param['Test Interval']=kwargs['Test_Interval']
    if Type=='SGLD':
        Param['Type']='SGLD'
        Param['Step Size'] = kwargs['Step_Size']
        Param['Beta'] = kwargs['Beta']
        Param['Batch Size'] = kwargs['Batch_Size']
        Param['Epoch'] = kwargs['Epoch']
        Param['Random Seed'] = kwargs['Random_Seed']
        Param['Num Run'] = kwargs['Num_Run']
        Param['Sigma'] = kwargs['Sigma']
        Param['Num CNN'] = kwargs['Num_CNN']
        Param['Interval'] = kwargs['Interval']
        Param['Test Interval'] = kwargs['Test_Interval']
        Param['Test Mode'] = kwargs['Test_Mode']
    if Type=='PSGLD':
        Param['Type'] = 'PSGLD'
        Param['Step Size'] = kwargs['Step_Size']
        Param['Beta'] = kwargs['Beta']
        Param['Batch Size'] = kwargs['Batch_Size']
        Param['Epoch'] = kwargs['Epoch']
        Param['Random Seed'] = kwargs['Random_Seed']
        Param['Num Run'] = kwargs['Num_Run']
        Param['Sigma'] = kwargs['Sigma']
        Param['Num CNN'] = kwargs['Num_CNN']
        Param['Interval'] = kwargs['Interval']
        Param['Test Interval'] = kwargs['Test_Interval']
        Param['Exp Term']=kwargs['Exp_Term']


    return Param


def write_Dict(Filename,Dict):
    '''
    To store Parameter dict to txt
    :param Filename: Str : The name of the file to store parameters
    :param Dict: dict: The parameter dictionary
    :return: None
    '''
    with open(Filename,'w') as file:
        file.write(json.dumps(Dict))


def roll_out(p):
    '''
    This determines whether to use roll-out with probability p
    :param p: The probability of roll-out
    :return: flag_roll_out (Bool)
    '''
    u=np.random.uniform(0,1)
    if u<=p:
        flag_roll_out=True
    else:
        flag_roll_out=False
    return flag_roll_out
def roll_out_multi(p,state_pos_rep,state_mom_rep,num_CNN,total_dim):
    '''
    THis implement the multiple roll-out
    :param p:
    :return: state_pos_init,state_mom_init
    '''
    # Generate which chains need to be initialized with repo
    u_list=torch.rand(num_CNN)
    ind_list=(u_list<=p)
    total_num_replay=torch.sum(ind_list)
    ind_ind=[i for i,p in enumerate(ind_list) if bool(p)]
    if torch.abs(total_num_replay-0.).data.cpu().numpy()<0.1:
        state_pos_init=0.1*torch.randn(num_CNN,total_dim,requires_grad=True)
        state_mom_init=0*torch.randn(num_CNN,total_dim,requires_grad=True)
        ind_state=[0]
        ind_ind=[0]
        return state_pos_init,state_mom_init,total_num_replay,ind_state,ind_ind
    # Draw from repo
    size_repo=len(state_pos_rep)

    if size_repo<num_CNN:
        ind_state=torch.randint(0,size_repo,(total_num_replay,))
        ind_chain=torch.multinomial(torch.ones(num_CNN),total_num_replay)
    else:

        ind_state=torch.multinomial(torch.ones(size_repo),total_num_replay)
        ind_chain=torch.randint(0,num_CNN,(total_num_replay,))
    # Extraction
    for i,x in enumerate(ind_state):
        if i==0:
            repo_pos=state_pos_rep[int(x)][int(ind_chain[i]):int(ind_chain[i])+1,:]
            repo_mom=state_mom_rep[int(x)][int(ind_chain[i]):int(ind_chain[i])+1,:]
        else:
            repo_pos=torch.cat((repo_pos,state_pos_rep[int(x)][int(ind_chain[i]):int(ind_chain[i])+1,:]),dim=0)
            repo_mom=torch.cat((repo_mom,state_mom_rep[int(x)][int(ind_chain[i]):int(ind_chain[i])+1,:]),dim=0)


    # Generate initialization
    state_pos_init=0.1*torch.randn(num_CNN,total_dim)
    state_pos_init[ind_list]=repo_pos
    state_mom_init=0*torch.randn(num_CNN,total_dim)
    state_mom_init[ind_list]=repo_mom
    # Set Requires grad
    state_pos_init.requires_grad=True
    state_mom_init.requires_grad=True
    return state_pos_init,state_mom_init,total_num_replay,ind_state,ind_ind





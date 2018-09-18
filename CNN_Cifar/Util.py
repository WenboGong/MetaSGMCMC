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

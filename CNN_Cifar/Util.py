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
        assert len(Param['Random Seed'])==Param['Num Run'],'Number of runs are not consistent with length of random seeds'
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




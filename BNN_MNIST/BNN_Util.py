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


from BNN_Q_D import *
from BNN_Model_def import *

def pairwise_distance(weight):
    num_chain=int(weight.data.shape[0])
    total_number=int((num_chain)*(num_chain-1.)/2.)
    p_dist=np.zeros(total_number)
    counter=0
    for i in range(num_chain-1):
        for j in range(i+1,num_chain):
            base=weight.data[i,:]
            pair=weight.data[j,:]
            #Tracer()()
            p_dist[counter]=torch.norm(base-pair,2)
            counter+=1
    return p_dist
def draw_samples(state_list,burn_in=0,interval=1,end=1000,draw_method='Cross'):
    state_list=state_list[burn_in:end:interval]
    all_samples=torch.stack(tuple(state_list),dim=1) # chain x time x total_dim
    if draw_method=='Cross':
        draw_list=state_list
    elif draw_method=='Within':
        draw_list=[]
        all_sample_list=list(torch.split(all_samples,1,dim=0))
        for p in all_sample_list:
            draw_list.append(torch.squeeze(p))
    elif draw_method=='Mix':
        draw_list=[torch.cat(tuple(state_list),dim=0)] # chain time x total_dim
    return draw_list

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
def generate_accuracy(test_loader,MLP_mnist,state_list_sghmc,state_list_nnsghmc,state_list_sgld,interval=2,limit=100,avg_time_nnsghmc=1.,avg_time_sghmc=1.,avg_time_sgld=1.,data_len=10000.):
    state_list_sghmc=state_list_sghmc[0:limit]
    state_list_nnsghmc=state_list_nnsghmc[0:limit]
    state_list_sgld=state_list_sgld[0:limit]
    len_sghmc=len(state_list_sghmc)
    len_nnsghmc=len(state_list_nnsghmc)
    Acc_sghmc_list=np.zeros(len_sghmc)
    Acc_nnsghmc_list=np.zeros(len_nnsghmc)
    Acc_sgld_list=np.zeros(len_nnsghmc)
    NLL_sghmc_list=np.zeros(len_sghmc)
    NLL_nnsghmc_list=np.zeros(len_nnsghmc)
    NLL_sgld_list=np.zeros(len_nnsghmc)
    time_list_sghmc=avg_time_sghmc*interval*np.linspace(0,len_sghmc,len_sghmc)
    time_list_nnsghmc=avg_time_nnsghmc*interval*np.linspace(0,len_nnsghmc,len_nnsghmc)
    time_list_sgld=avg_time_sgld*interval*np.linspace(0,len_nnsghmc,len_nnsghmc)
    for ind,state_sghmc in zip(range(len_sghmc),state_list_sghmc):
        #draw_list_sghmc=draw_samples(state_list_sghmc,burn_in=0,interval=1,draw_method='Cross')
        if (ind+1)%20==0:
            print('%s'%(ind+1))
        state_nnsghmc=state_list_nnsghmc[ind]
        state_sgld=state_list_sgld[ind]
        Acc_sghmc,NLL_sghmc=Test_accuracy(test_loader,MLP_mnist,state_sghmc,data_number=data_len)
        Acc_nnsghmc,NLL_nnsghmc=Test_accuracy(test_loader,MLP_mnist,state_nnsghmc,data_number=data_len)
        Acc_sgld,NLL_sgld=Test_accuracy(test_loader,MLP_mnist,state_sgld,data_number=data_len)
        
        Acc_sghmc_list[ind]=1.-Acc_sghmc
        Acc_nnsghmc_list[ind]=1.-Acc_nnsghmc
        Acc_sgld_list[ind]=1.-Acc_sgld
        NLL_sghmc_list[ind]=-NLL_sghmc.data.cpu().numpy()
        NLL_nnsghmc_list[ind]=-NLL_nnsghmc.data.cpu().numpy()
        NLL_sgld_list[ind]=-NLL_sgld.data.cpu().numpy()
    return Acc_sghmc_list,Acc_nnsghmc_list,Acc_sgld_list,NLL_sghmc_list,NLL_nnsghmc_list,NLL_sgld_list,time_list_sghmc,time_list_nnsghmc,time_list_sgld

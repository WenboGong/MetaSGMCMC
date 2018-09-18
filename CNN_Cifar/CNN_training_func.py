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
from torch.utils.data import Dataset, DataLoader
# Import Custom packages
from Cifar_Dataloader import *
from CNN_Module import *
from Test_Module import *
from Util import *
from CNN_Q_D import *
from Test_Module import *
########################################################################################################################
def rbf_kernel_matrix_eff(x,y,bandwidth):
    num_samples=int(x.data.shape[0])
    dim=int(x.data.shape[1])
    x=torch.tensor(x.data)
    y=torch.tensor(y.data)
    x_batch=torch.unsqueeze(x,dim=0).repeat(num_samples,1,1) # Nx(rep) x Nx x d
    y_batch=torch.unsqueeze(y,dim=1) # Ny x 1 x d
    # Kernel Matrix
    K=torch.tensor(torch.exp(-0.5*torch.sum(torch.abs(y_batch.data-x_batch.data)**2,dim=2)/(bandwidth.data))) # N x N
    # G_K
    K_batch=torch.unsqueeze(K,dim=2)
    G_K=torch.tensor(torch.sum(1./(bandwidth.data)*K_batch.data*(y_batch.data-x_batch.data),dim=1))
    return K,G_K
def rbf_kernel_matrix_eff_2(x,y,bandwidth):
    num_samples = int(x.data.shape[0])
    dim = int(x.data.shape[1])
    x = torch.tensor(x.data)
    y = torch.tensor(y.data)

    x_batch = torch.unsqueeze(x, dim=0).repeat(num_samples, 1, 1)  # Nx(rep) x Nx x d
    y_batch = torch.unsqueeze(y, dim=1)  # Ny x 1 x d

    Z=torch.cat((x,y),dim=0)
    ZZT=torch.mm(Z,Z.t())
    diag_ZZT=torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr=diag_ZZT.expand_as(ZZT)
    exponent=Z_norm_sqr-2*ZZT+Z_norm_sqr.t()
    K=torch.exp(-0.5/(bandwidth.data)*exponent)[:num_samples,num_samples:]
    K_batch = torch.unsqueeze(K, dim=2)
    # Trans to CPU computation
    x_batch_np=x_batch.cpu().data.numpy()
    y_batch_np=y_batch.cpu().data.numpy()
    K_batch_np=K_batch.cpu().data.numpy()
    G_K_np=np.sum(1./(bandwidth.data.cpu().numpy())*K_batch_np*(y_batch_np-x_batch_np),axis=1)
    G_K=torch.from_numpy(G_K_np).float().cuda()
    #G_K = torch.tensor(torch.sum(1. / (bandwidth.data) * K_batch.data * (y_batch.data - x_batch.data), dim=1))
    return K,G_K
def gradient_estimate_im(x,bandwidth,lam=0.5,cpu_flag=False):
    num_samples = int(x.data.shape[0])
    if cpu_flag==True:
        K_e, G_K_e = rbf_kernel_matrix_eff_2(x, x, bandwidth)
    else:
        K_e, G_K_e = rbf_kernel_matrix_eff(x, x, bandwidth)
    G_e = torch.tensor(-(K_e.data + lam * Variable(torch.eye(num_samples)).data).inverse().matmul(G_K_e.data).data)
    return G_e
def sample_median(samples):
    counter=0
    for x_d in samples.split(1):
        d=torch.unsqueeze(torch.norm(x_d.data-samples.data,2,dim=1)**2,0)
        if counter==0:
            M=d
        else:
            M=torch.cat((M,d),dim=0)
        counter+=1
    med=torch.tensor(torch.Tensor([torch.median(M)]))
    return med
def grad_ELBO(weight,CNN,x,y,data_N,counter,limit_step,sample_interval,sigma=1.):
    # Need to retain graph??????
    #
    r=float(np.floor((limit_step-1)/sample_interval))
    weight_copy=torch.tensor(weight.data,requires_grad=True)
    #neg_dlogP_dW = CNN.part_free_grad_CNN(x, y, weight_copy, data_N, coef=1., sigma=sigma, flag_retain=False)
    neg_dlogP_dW, _, _, _ = CNN.grad_CNN(x, y, weight_copy, data_N, coef=1., sigma=sigma, flag_retain=False)  # num_CNN x dim

    dlogP_dW = torch.tensor((-neg_dlogP_dW).data)  # chain x dim
    loss_1 = 1./r*torch.mean(dlogP_dW * weight, dim=0, keepdim=True)  # 1 x dim
    # Backpropagation for cross chain U term
    (-loss_1).backward(torch.ones(loss_1.data.shape), retain_graph=True)
    ######### Evaluate entropy gradient
    bandwidth = torch.tensor((sample_median(weight) / 2.).data)
    dlogQ_dW = gradient_estimate_im(weight, bandwidth, lam=0.05)
    loss_2 = 1./r*torch.mean(dlogQ_dW * weight, dim=0, keepdim=True) # 1 x dim
    loss_2.backward(torch.ones(loss_2.data.shape), retain_graph=True)
    counter=1

    return counter
def grad_ELBO_In_Chain(CNN,x,y,data_N,state_list_in_chain,sub_sample_number=8,sigma=1.):
    # Number of chains and total time
    num_CNN=state_list_in_chain[0].shape[0]
    T=float(len(state_list_in_chain))
    # Subsample index
    ind=np.random.choice(num_CNN,sub_sample_number,replace=False)
    # Reshape and concat to single vector
    for i,p in zip(range(int(T)),state_list_in_chain):
        p_ind=p[ind,:]
        if i==0:
            concat_vector=p_ind
        else:
            concat_vector=torch.cat((concat_vector,p_ind),dim=0)

    # ELBO Loss First Term
    concat_vector_copy=torch.tensor(concat_vector.data,requires_grad=True)
    #neg_dlogP_dW, _, _, _ = CNN.grad_CNN(x, y, concat_vector_copy, data_N, coef=1., sigma=sigma, flag_retain=False)  # concat_length x dim
    neg_dlogP_dW=CNN.part_free_grad_CNN(x, y, concat_vector_copy, data_N, coef=1., sigma=sigma, flag_retain=False)

    dlogP_dW = torch.tensor((-neg_dlogP_dW).data)  # cat length x dim
    loss_1 = 1./sub_sample_number*torch.mean(dlogP_dW * concat_vector, dim=0, keepdim=True)  # 1 x dim
    (-loss_1).backward(torch.ones(loss_1.data.shape), retain_graph=True)
    ########## Evaluate Entropy gradient ###############
    bandwidth = torch.tensor((sample_median(concat_vector) / 2.).data)
    dlogQ_dW = gradient_estimate_im(concat_vector, bandwidth, lam=0.05,cpu_flag=True)



    loss_2 = 1./sub_sample_number*torch.mean(dlogQ_dW * concat_vector, dim=0, keepdim=True)  # 1 x dim
    loss_2.backward(torch.ones(loss_2.data.shape), retain_graph=False)
    return None


def modify_grad(counter_ELBO,Q_MLP,D_MLP):
    for layer,p in enumerate(Q_MLP.parameters()):
        if type(p.grad)!=type(None):
            p.grad=torch.tensor(1./counter_ELBO*p.grad.data)
    for layer,p in enumerate(D_MLP.parameters()):
        if type(p.grad)!=type(None):
            p.grad=torch.tensor(1./counter_ELBO*p.grad.data)
    return None




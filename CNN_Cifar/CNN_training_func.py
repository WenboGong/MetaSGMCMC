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
def gradient_estimate_im(x,bandwidth,lam=0.5):
    num_samples = int(x.data.shape[0])
    K_e, G_K_e = rbf_kernel_matrix_eff(x, x, bandwidth)
    G_e = torch.tensor(-(K_e.data + lam * Variable(torch.eye(num_samples)).data).inverse().matmul(G_K_e.data))
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
def grad_ELBO(weight,CNN,x,y,data_N,counter,sigma=1.):
    # Need to retain graph??????
    neg_dlogP_dW, _, _, _ = CNN.grad_CNN(x,y,weight,data_N,coef=1.,sigma=sigma,flag_retain=True)# num_CNN x dim
    dlogP_dW = torch.tensor((-neg_dlogP_dW).data)  # chain x dim
    loss_1 = torch.mean(dlogP_dW * weight, dim=0, keepdim=True)  # 1 x dim
    # Backpropagation for cross chain U term
    (-loss_1).backward(torch.ones(loss_1.data.shape), retain_graph=True)
    ######### Evaluate entropy gradient
    bandwidth = torch.tensor((sample_median(weight) / 2.).data)
    dlogQ_dW = gradient_estimate_im(weight, bandwidth, lam=0.05)
    loss_2 = torch.mean(dlogQ_dW * weight, dim=0, keepdim=True) # 1 x dim
    loss_2.backward(torch.ones(loss_2.data.shape), retain_graph=True)
    counter+=1
    return counter
def modify_grad(counter_ELBO,Q_MLP,D_MLP):
    for layer,p in enumerate(Q_MLP.parameters()):
        if type(p.grad)!=type(None):
            p.grad=torch.tensor(1./counter_ELBO*p.grad.data)
    for layer,p in enumerate(D_MLP.parameters()):
        if type(p.grad)!=type(None):
            p.grad=torch.tensor(1./counter_ELBO*p.grad.data)
    return None




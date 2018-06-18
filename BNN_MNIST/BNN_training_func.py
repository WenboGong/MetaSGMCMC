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

from BNN_Model_def import *
from BNN_Util import *
from BNN_Q_D import *
from BNN_Sampler import *

def grad_ELBO(weight,BNN_obj,X,y,Q_MLP,D_MLP,data_N,sigma=1.):
    num_chain=int(weight.data.shape[0])
    ######## E_q[log p]
    neg_dlogP_dW,_,_,_=BNN_obj.grad_BNN(X,y,weight,data_N=data_N,sigma=sigma,flag_retain=True)
    #detach
    dlogP_dW=Variable((-neg_dlogP_dW).data) # chain x dim
    loss_1=torch.mean(dlogP_dW*weight,dim=0,keepdim=True) # 1 x dim
    #grad_loss_1_Q=grad(loss_1,Q_MLP.parameters(),torch.ones(loss_1.data.shape),retain_graph=True,allow_unused=True)
    #grad_loss_1_D=grad(loss_1,D_MLP.parameters(),torch.ones(loss_1.data.shape),retain_graph=True,allow_unused=True)


    (-loss_1).backward(torch.ones(loss_1.data.shape),retain_graph=True)
    
    ######## E_q[log q]
    bandwidth=Variable((sample_median(weight)/2.).data)
    dlogQ_dW=gradient_estimate_im(weight,bandwidth,lam=0.05) # chain x dim
    loss_2=torch.mean(dlogQ_dW*weight,dim=0,keepdim=True) # 1 x dim
    #grad_loss_2_Q=grad(loss_2,Q_MLP.parameters(),torch.ones(loss_2.data.shape),retain_graph=True,allow_unused=True)
    #grad_loss_2_D=grad(loss_2,D_MLP.parameters(),torch.ones(loss_2.data.shape),retain_graph=True,allow_unused=True)
    loss_2.backward(torch.ones(loss_2.data.shape),retain_graph=True)
    return 1,1,1,1
    #return grad_loss_1_Q,grad_loss_1_D,grad_loss_2_Q,grad_loss_2_D
def modify_grad(total_grad,Q_MLP,D_MLP):
    num_time=len(total_grad) # num of KL computed
    for layer,p in enumerate(Q_MLP.parameters()):
        if type(p.grad)!=type(None):
            p.grad=Variable(1./num_time*p.grad.data)
    for layer,p in enumerate(D_MLP.parameters()):
        if type(p.grad)!=type(None):
            p.grad=Variable(1./num_time*p.grad.data)
    return None        
def gradient_estimate_im(x,bandwidth,lam=0.5):
    #x=convert_to_Variable(state_list)
    num_samples=int(x.data.shape[0])
    K_e,G_K_e=rbf_kernel_matrix_eff(x,x,bandwidth)
    G_e=Variable(-(K_e.data+lam*Variable(torch.eye(num_samples)).data).inverse().matmul(G_K_e.data))
    return G_e
def rbf_kernel_matrix_eff(x,y,bandwidth):
    num_samples=int(x.data.shape[0])
    dim=int(x.data.shape[1])
    x=Variable(x.data.clone())
    y=Variable(y.data.clone())   
    x_batch=torch.unsqueeze(x,dim=0).repeat(num_samples,1,1) # Nx(rep) x Nx x d
    y_batch=torch.unsqueeze(y,dim=1) # Ny x 1 x d   
    # Kernel Matrix
    K=Variable(torch.exp(-0.5*torch.sum(torch.abs(y_batch.data-x_batch.data)**2,dim=2)/(bandwidth.data))) # N x N
    # G_K
    K_batch=torch.unsqueeze(K,dim=2)
    G_K=Variable(torch.sum(1./(bandwidth.data)*K_batch.data*(y_batch.data-x_batch.data),dim=1))
    return K,G_K
def sample_median(samples):
    counter=0
    for x_d in samples.split(1):
        d=torch.unsqueeze(torch.norm(x_d.data-samples.data,2,dim=1)**2,0)
        if counter==0:
            M=d
        else:
            M=torch.cat((M,d),dim=0)
        counter+=1
    med=Variable(torch.Tensor([torch.median(M)]))
    return med
def Param_Gradient_Estimate(x,G,param,flag_detach=True):
    '''
        Args: x
              G: estimation of grad_x(log q) with dimension N x D/2
              param: Network Parameters
        Output: Gradient estimate: grad_{param}(E_q[log q])
    '''
    #x=convert_to_Variable(state_list)
    dim=int(x.data.shape[1])
    num_samples=int(x.data.shape[0])
    G_param=list(grad(1./num_samples*G*x,param,torch.ones(x.data.shape),create_graph=True,allow_unused=True))
    for p in G_param:
        if type(p)!=type(None):
            p=Variable(p.data)
    return G_param
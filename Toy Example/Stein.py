import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad


def gradient_estimate_im(x,bandwidth,lam=0.5):
    #x=convert_to_Variable(state_list)
    num_samples=int(x.data.shape[0])
    K_e,G_K_e=rbf_kernel_matrix_eff(x,x,bandwidth)
    G_e=-(K_e+lam*Variable(torch.eye(num_samples))).inverse().matmul(G_K_e)
    return G_e
def rbf_kernel_matrix_eff(x,y,bandwidth):
    num_samples=int(x.data.shape[0])
    dim=int(x.data.shape[1])
    x=Variable(x.data.clone(),requires_grad=True)
    y=Variable(y.data.clone(),requires_grad=True)   
    x_batch=torch.unsqueeze(x,dim=0).repeat(num_samples,1,1) # Nx(rep) x Nx x d
    y_batch=torch.unsqueeze(y,dim=1) # Ny x 1 x d   
    # Kernel Matrix
    K=torch.exp(-0.5*torch.sum(torch.abs(y_batch-x_batch)**2,dim=2)/(bandwidth)) # N x N
    # G_K
    K_batch=torch.unsqueeze(K,dim=2)
    G_K=torch.sum(1./(bandwidth)*K_batch*(y_batch-x_batch),dim=1)
    return K,G_K
def sample_median(samples):
    counter=0
    for x_d in samples.split(1):
        d=torch.unsqueeze(torch.norm(x_d-samples,2,dim=1)**2,0)
        if counter==0:
            M=d
        else:
            M=torch.cat((M,d),dim=0)
        counter+=1
    med=torch.median(M)
    return med
def Param_Gradient_Estimate(x,G,param):
    '''
        Args: x
              G: estimation of grad_x(log q) with dimension N x D/2
              param: Network Parameters
        Output: Gradient estimate: grad_{param}(E_q[log q])
    '''
    #x=convert_to_Variable(state_list)
    dim=int(x.data.shape[1])
    num_samples=int(x.data.shape[0])
    G_param=grad(1./num_samples*G*x,param,torch.ones(x.data.shape),create_graph=True,allow_unused=True)
    return G_param
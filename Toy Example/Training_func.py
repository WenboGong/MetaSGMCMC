import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from Util import *
from Stein import *

def Chain_grad(state_list_nnsg,draw_method,Q_MLP,D_MLP,U,end,burn_in=0,interval=9):
    ######### Draw Samples #############
    samples_list=parallel_draw_sample(state_list_nnsg,end,burn_in=burn_in,interval=interval,draw_method=draw_method)
    len_samples=len(samples_list)
    samples=torch.cat(tuple(samples_list),dim=0)
    num_samples=int(samples.data.shape[0])
    mean_log_p=1./num_samples*torch.sum(U.forward(samples,transp=False))
    ####### Compute Gradient E_q[log p] #########
    grad_log_p=grad(mean_log_p,Q_MLP.parameters(),create_graph=True,allow_unused=True)
    grad_log_p_D=grad(mean_log_p,D_MLP.parameters(),create_graph=True,allow_unused=True)
    
    
    ##### E_q[log q] ####
    for each_ind,each_sample in zip(range(len_samples),samples_list):
        #### Kernel bandwidth using median Heuristics ######
        med=Variable(sample_median(each_sample).data.clone())
        bandwidth=0.5*med
        #### Compute KL
        if each_ind==0:
            G_log=Variable(gradient_estimate_im(each_sample,bandwidth,lam=0.03).data.clone())
        else:
            grad_x_log_q=Variable(gradient_estimate_im(each_sample,bandwidth,lam=0.03).data.clone())
            G_log=torch.cat((G_log,grad_x_log_q),dim=0)
            
    grad_Qparam_log_q=Param_Gradient_Estimate(samples,G_log,Q_MLP.parameters())
    grad_Dparam_log_q=Param_Gradient_Estimate(samples,G_log,D_MLP.parameters())
    return grad_log_p,grad_log_p_D,grad_Qparam_log_q,grad_Dparam_log_q,mean_log_p
def All_Chain_grad(state_list_nnsg,draw_method,Q_MLP,D_MLP,U,end,burn_in=0,interval=9):
    ######## Draw Samples ############
    samples_list=parallel_draw_sample(state_list_nnsg,end,burn_in=burn_in,interval=interval,draw_method=draw_method)    
    ind_start=np.random.randint(10)
    ind_interval=np.random.randint(5,10)
    #### Subsampling chain #########
    samples_list=samples_list[ind_start::ind_interval]
    #samples_list=[samples_list[ind_start]]
    len_samples=len(samples_list)
    
    samples=torch.cat(tuple(samples_list),dim=0)
    
    num_samples=int(samples.data.shape[0])
    mean_log_p=1./num_samples*torch.sum(U.forward(samples,transp=False))
    ####### Compute Gradient E_q[log p] #########
    grad_log_p=grad(mean_log_p,Q_MLP.parameters(),create_graph=True,allow_unused=True)
    grad_log_p_D=grad(mean_log_p,D_MLP.parameters(),create_graph=True,allow_unused=True)
    ####### Compute entropy for all points ######
    med=Variable(sample_median(samples).data.clone())
    bandwidth=0.5*med
    G_log=Variable(gradient_estimate_im(samples,bandwidth,lam=0.03).data.clone())
    
    grad_Qparam_log_q=Param_Gradient_Estimate(samples,G_log,Q_MLP.parameters())
    grad_Dparam_log_q=Param_Gradient_Estimate(samples,G_log,D_MLP.parameters())
    return grad_log_p,grad_log_p_D,grad_Qparam_log_q,grad_Dparam_log_q,mean_log_p

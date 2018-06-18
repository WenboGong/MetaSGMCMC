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

from BNN_Util import *
from BNN_Model_def import *

class parallel_Q_eff:
    def __init__(self,total_dim,Q_MLP,BNN_obj,num_chain,clamp=100000,dim_pen=1.,dim_pen_p=1.,sqrt=False):
        self.Q_MLP=Q_MLP
        self.BNN_obj=BNN_obj
        self.clamp=clamp
        self.dim=total_dim
        self.dim_pen=dim_pen
        self.dim_pen_p=dim_pen_p
        self.num_chain=num_chain
        self.sqrt=sqrt
    def forward(self,state_mom,state_pos,energy,grad_U,const=0.,Q_pre=0.,state_mom_pre=0.,energy_pre=0.,flag_graph=True,flag_finite=False):
        '''
        Input is now U and momentum
        Notice: U is the mean energy function, not the total energy function for stability 
        
        The structure of state_pos and state_mom is num_chain x total_dim
        The structure of input to Q and D is num_chain x total_dim x 2/3
        
        '''
        all_dim=int(self.dim*self.num_chain)
        ##### NN ###########
        ##### Not flow through the Q net
        state_pos=Variable(state_pos.data,requires_grad=True) # num_chain x dim
        state_mom=Variable(state_mom.data,requires_grad=True)/self.dim_pen_p # num_chain x dim
        #####
        U_value=Variable(energy.data)/self.dim_pen #num_chain x 1 
        U_value=U_value.repeat(1,self.dim)# num_chain x dim 
        
        
        
        grad_U=Variable(grad_U.data)# num_chain x dim 
        
        if self.sqrt==False:
            input_NN=Variable(torch.stack((U_value,state_mom),dim=2).data,requires_grad=True) ### num_chain x dim x 2
        else:
            raise NotImplementedError
        out=self.Q_MLP.forward(input_NN) ## num_chain x dim x 1
        out=torch.clamp(out, min=-self.clamp,max=self.clamp)
        out_Q=const+torch.squeeze(out) #num_chain x dim
        
        
        if flag_finite==False:
            grad_Q=grad(out,input_NN,torch.ones(out.data.shape),allow_unused=True,create_graph=flag_graph)[0]
        if self.sqrt==False and flag_finite==False:
            grad_Q_mom=torch.squeeze(grad_Q[:,:,1:]/self.dim_pen_p) #### num_chain x dim
            grad_Q_pos=torch.squeeze(grad_Q[:,:,0:1])*grad_U/self.dim_pen ## num_chain x dim
        return out_Q,grad_Q_pos,grad_Q_mom,grad_Q
    def finite_diff_forward(self,state_mom,state_pos,energy,energy_rep,grad_U,state_mom_pre,energy_pre,Q_dash_pre,Q_pre,const=1.,flag_dash=False):
        state_pos=Variable(state_pos.data,requires_grad=True) # num_chain x dim
        state_mom=Variable(state_mom.data,requires_grad=True)/self.dim_pen_p # num_chain x dim

        U_value=Variable(energy_rep.data)/self.dim_pen #num_chain x 1 
        #U_value=U_value.repeat(1,self.dim)# num_chain x dim 
              
        
        grad_U=Variable(grad_U.data)# num_chain x dim 
        
        input_NN=Variable(torch.stack((U_value,state_mom),dim=2).data,requires_grad=True)
        
        out=self.Q_MLP.forward(input_NN) ## num_chain x dim x 1
        out=torch.clamp(out, min=-self.clamp,max=self.clamp)
        #out=torch.sign(out)*F.relu(torch.abs(out)-self.clamp_min)+self.clamp_min
        out_Q=torch.squeeze(out)# Q_t
        
        
        
        
        ############ Finite difference for gradient ###########
        grad_Q_pos=Variable((out_Q.data-Q_dash_pre.data)/(energy.data-energy_pre.data+1e-7)*grad_U.data)
        if flag_dash==False:
            grad_Q_mom=Variable((Q_dash_pre.data-Q_pre.data)/(state_mom.data-state_mom_pre.data+1e-7))
        else:
            ##### Q_pre is Q_t and state_mom is p_{t+1} and state_mom_pre is p_{t} ###
            grad_Q_mom=Variable((out_Q.data-Q_pre.data)/(state_mom.data-state_mom_pre.data+1e-7))
        return out_Q,grad_Q_pos,grad_Q_mom    
class parallel_D_eff:
    def __init__(self,total_dim,D_MLP,BNN_obj,num_chain,clamp_min=0.,clamp_max=10000000,dim_pen=1.,dim_pen_p=1.,dim_pen_g=1.,sqrt=False):
        self.D_MLP=D_MLP
        self.BNN_obj=BNN_obj
        self.dim=total_dim
        self.dim_pen=dim_pen
        self.dim_pen_p=dim_pen_p
        self.dim_pen_g=dim_pen_g
        self.clamp_min=clamp_min
        self.clamp_max=clamp_max
        self.num_chain=num_chain
        self.sqrt=sqrt
    def forward(self,state_mom,state_pos,energy,grad_U,Q_out,grad_Q_mom,const=0.,flag_graph=True,flag_finite=False):
        '''
        Input is now U and momentum and Gradient
        The strucutre of momentum/gradient is num_chain x dim and energy is num_chain x 1
        The strucutre of input_NN is num_chain x dim x 3 (energy,p,gradient)
        '''
        all_dim=int(self.dim*self.num_chain)
        ##### NN ###########
        ##### Not flow through the Q net
        state_pos=Variable(state_pos.data,requires_grad=True) # num_chain x total_dim
        state_mom=Variable(state_mom.data,requires_grad=True)/self.dim_pen_p # num_chain x total_dim
        
        Q_out=Variable(Q_out.data,requires_grad=True)
        grad_Q_mom=Variable(grad_Q_mom.data)
        #####
        U_value=Variable(energy.data)/self.dim_pen# num_chain x 1
        U_value=U_value.repeat(1,self.dim)# num_chain x total_dim
        
        grad_U=70*Variable(grad_U.data)/self.dim_pen_g### num_chain x total_dim
        
        if self.sqrt==False:
            input_NN=Variable(torch.stack((U_value,state_mom,grad_U),dim=2).data,requires_grad=True) # num_chain x dim x 3
        else:
            raise NotImplementedError
        out=50*self.D_MLP.forward(input_NN) ### num_chain x dim x 1
        out_D=0+(Q_out**2)*const+torch.squeeze(out)
        out_D=self.clamp_min+F.relu(out_D-self.clamp_min)
        out_D=torch.clamp(out_D, min=self.clamp_min,max=self.clamp_max)
        if flag_finite==True:
            grad_D_mom=torch.squeeze(grad(out,input_NN,torch.ones(out.data.shape),allow_unused=True,create_graph=flag_graph)[0][:,:,1:2])/self.dim_pen_p#Variable(torch.zeros(state_mom.data.shape))
            grad_D_Q=Variable(grad(out_D,Q_out,torch.ones(Q_out.data.shape),allow_unused=True,create_graph=flag_graph)[0].data)
        if self.sqrt==False and flag_finite==False:
            grad_D_mom=torch.squeeze(grad(out,input_NN,torch.ones(out.data.shape),allow_unused=True,create_graph=flag_graph)[0][:,:,1:2])/self.dim_pen_p#Variable(torch.zeros(state_mom.data.shape))## num_chain x dim 
            grad_D_Q=grad(out_D,Q_out,torch.ones(Q_out.data.shape),allow_unused=True,create_graph=flag_graph)[0]
        return out_D,grad_D_mom,grad_D_Q
    def finite_diff_forward(self,state_mom,energy,energy_rep,grad_U,state_mom_pre,Q_out,const=0.):
        state_mom=Variable(state_mom.data,requires_grad=True)/self.dim_pen_p
        Q_out=Variable(Q_out.data,requires_grad=True)
        U_value=Variable(energy_rep.data)/self.dim_pen# num_chain x 1
        
        grad_U=70*Variable(grad_U.data)/self.dim_pen_g### num_chain x total_dim
        input_NN=Variable(torch.stack((U_value,state_mom,grad_U),dim=2).data,requires_grad=True)
        input_NN_pre=Variable(torch.stack((U_value,state_mom_pre,grad_U),dim=2).data,requires_grad=True)
        
        out=50*self.D_MLP.forward(input_NN)
        out_pre=50*self.D_MLP.forward(input_NN_pre)
        out_pre=torch.squeeze(out_pre)
        
        out_D=0+(Q_out**2)*const+torch.squeeze(out)
        out_D=self.clamp_min+F.relu(out_D-self.clamp_min)
        out_D=torch.clamp(out_D, min=self.clamp_min,max=self.clamp_max)
        
        
        
        ########## finite diff compute gradient ######
        grad_D_mom=Variable((torch.squeeze(out).data-out_pre.data)/(state_mom.data-state_mom_pre.data+1e-10))
        grad_D_Q=Variable(2*Q_out.data*const)
        return out_D,grad_D_mom,grad_D_Q
        
        
        
class parallel_Gamma_eff:
    def __init__(self,total_dim,Q_NN,D_NN=None):
        self.dim=total_dim
        self.Q_NN=Q_NN
        self.D_NN=D_NN
    def forward(self,state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q,const_D,Q_out,clamp_min=30.):
        '''
        Gamma matrix 
        
        '''
        #Tracer()()
        out1=-grad_Q_mom # num_chain x dim 
        out2=grad_Q_pos # num_chain x dim
        if self.D_NN!=None:
            out2=grad_Q_pos+grad_D_mom+grad_D_Q*grad_Q_mom#-torch.sign(Q_out)*1./(Q_out**2)*const_D*grad_Q_mom # num_chain x dim 
        return out1,out2
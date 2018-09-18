############################ Import Necessary packages ##########################################
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

from CNN_Module import *
from Util import *
from CNN_Module import *
from Test_Module import *

###################################################################################################
class parallel_Q_eff:
    '''
    This class contains the implementation of Q matrix and finite difference approximation required for :math:`\Gamma` matrix
    '''
    def __init__(self,CNN,MLP,clamp=100000,offset=0.):
        '''
        Init
        :param CNN: CNN Obj
        :param MLP: NN for Q matrix
        :param clamp: The threshold for Q output (Clamp both positive and negative value)
        '''
        self.CNN=CNN
        self.MLP=MLP
        self.total_dim=self.CNN.get_dimension()
        self.num_CNN=self.CNN.num_CNN
        self.clamp=clamp
        self.offset=offset

    def forward(self,state_mom,energy,grad_U,flag_graph=True):
        '''
        The evaluation of Q matrix

        :param state_mom: The current momentum tensor
        :param energy: The energy for each chain: Size: :math:`num_CNN x 1`
        :param grad_U: The energy gradients: Size: :math:`num_CNN x total dim`
        :param const: The offset value for Q matrix
        :param flag_graph: Whether to create the computation graph, default to True for training the sampler because of the :math:`\Gamma` matrix
        :return: out_Q,grad_Q_pos,grad_Q_mom,grad_Q
        '''
        # To create separate copy for each input to stop back-propagate through these variable during training
        state_mom,U_value,grad_U=torch.tensor(state_mom.data,requires_grad=True),5*torch.tensor(energy.data),5*torch.tensor(
            grad_U.data)
        U_value=U_value.repeat(1,self.total_dim)# num_CNN x total_dim
        # Group input together
        input_NN=torch.tensor(torch.stack((U_value,state_mom),dim=2).data,requires_grad=True) #num_CNN x total_dim x 2
        # forward pass through the NN
        out = self.MLP.forward(input_NN)  ## num_chain x dim x 1
        out_Q= self.offset+torch.squeeze(torch.clamp(out, min=-self.clamp, max=self.clamp))# num_CNN x dim
        #out_Q = const + torch.squeeze(out)  # num_chain x dim
        # compute exact gradient
        grad_Q = grad(out, input_NN, torch.ones(out.data.shape), allow_unused=True,
                      create_graph=flag_graph)[0]# num_CNN x total dim x 2
        grad_Q_mom = torch.squeeze(grad_Q[:, :, 1:])  #### num_chain x dim
        grad_Q_pos = torch.squeeze(grad_Q[:, :, 0:1]) * grad_U  ## num_chain x dim
        return out_Q,grad_Q_pos,grad_Q_mom,grad_Q
    def finite_diff_forward(self,state_mom,energy,energy_rep,grad_U,state_mom_pre,energy_pre,Q_dash_pre,
                            Q_pre,flag_dash=False):
        '''
        This implements the finite different approximation to compute necessary statistics for :math:`\Gamma` matrix
        :param state_mom: The momentum variable with size: :math:`num_CNN x total dim`
        :param energy: The mean energy vector for all chains with size :math:`num_CNN x 1`
        :param energy_rep: The repeated copy of mean energy with size: :math:`num_CNN x total_dim`
        :param grad_U: The mean energy gradient
        :param state_mom_pre: The previous momentum variable
        :param energy_pre: The previous mean energy
        :param Q_dash_pre: The previous :math:`Q'` matrix with :math:`\theta_{t-1}` and :math:`p_{t}` at time t.
        :param Q_pre: The previous Q matrix, so at time t, it is :math:`Q_{t-1}`
        :param flag_dash: The flag whether to evaluate the :math:`Q'` matrix. At the beginning of each subsampled step, turn off this, otherwise, wrong gradient.
        :return: out_Q, grad_Q_pos,grad_Q_mom
        '''
        state_mom=torch.tensor(state_mom.data)
        U_value=torch.tensor(energy_rep.data)
        grad_U=torch.tensor(grad_U.data)
        input_NN = Variable(torch.stack((U_value, state_mom), dim=2).data, requires_grad=True)

        out = self.MLP.forward(input_NN)
        out = torch.clamp(out, min=-self.clamp, max=self.clamp)
        out_Q = self.offset+torch.squeeze(out)# num_CNN x dim

        # Finite Difference for gradient
        grad_Q_pos = torch.tensor((out_Q.data - Q_dash_pre.data) / (energy.data - energy_pre.data + 1e-7) * grad_U.data)
        if flag_dash==False:
            # only use this at the beginning of each subsampling, Q_dahs_pre will be Q'
            grad_Q_mom=torch.tensor((Q_dash_pre.data-Q_pre.data)/(state_mom.data-state_mom_pre.data+1e-7))
        else:
            # now out_Q is the Q'
            grad_Q_mom=torch.tensor((out_Q.data-Q_pre.data)/(state_mom.data-state_mom_pre.data+1e-7))

        return out_Q, grad_Q_pos,grad_Q_mom

class parallel_D_eff:
    def __init__(self,CNN,MLP,grad_U_scale=70.,D_scale=50.,offset=0.,clamp_min=0.,clamp_max=10000.,scale_Q_term=0.):
        self.CNN=CNN
        self.MLP=MLP
        self.total_dim=self.CNN.get_dimension()
        self.num_CNN=self.CNN.num_CNN
        # scale term
        self.grad_U_scale=grad_U_scale
        self.D_scale=D_scale
        self.offset=offset
        self.scale_Q_term=scale_Q_term
        # Clamp value
        self.clamp_min=clamp_min
        self.clamp_max=clamp_max
    def forward(self,state_mom,energy,grad_U,Q_out,flag_graph=True):
        '''
        This is to compute the necessary statistics for D matrix
        :param state_mom: The current momentum variable
        :param energy: The mean U value at current time
        :param grad_U: The mean energy gradient
        :param Q_out: The output from Q matrix
        :param const: The offset for :math:`Q^2` term
        :param flag_graph: Whether to create the graph for higher order gradient computation. Only set true if the sampler is at training mode
        :return: out_D, grad_D_mom, grad_D_Q
        '''
        # Clone the input to stop gradient propagation
        state_mom=torch.tensor(state_mom.data,requires_grad=True)

        # Should I decouple Q_out? Try not first
        #Q_out=torch.tensor(Q_out.data,requires_grad=True)



        U_value = 5*torch.tensor(energy.data) # num_chain x 1
        U_value = U_value.repeat(1, self.total_dim)  # num_chain x total_dim
        grad_U = self.grad_U_scale * torch.tensor(grad_U.data)  ### num_chain x total_dim
        # Now form the input to NN
        input_NN = torch.tensor(torch.stack((U_value, state_mom, grad_U), dim=2).data,
                            requires_grad=True)  # num_chain x dim x 3
        # Now compute the output
        out = self.D_scale * self.MLP.forward(input_NN) # num_CNN x dim x 1
        out_D = self.offset + (Q_out ** 2) * self.scale_Q_term + torch.squeeze(out)
        # Clamp the value
        out_D = torch.clamp(out_D, min=self.clamp_min, max=self.clamp_max)
        # Compute the gradient
        grad_D_mom = torch.squeeze(
            grad(out, input_NN, torch.ones(out.data.shape), allow_unused=True, create_graph=flag_graph)[0][:, :,
            1:2])  # Variable(torch.zeros(state_mom.data.shape))## num_chain x dim
        grad_D_Q = grad(out_D, Q_out, torch.ones(Q_out.data.shape), allow_unused=True, create_graph=flag_graph)[0]
        return out_D,grad_D_mom,grad_D_Q
    def finite_diff_forward(self,state_mom,energy_rep,grad_U,state_mom_pre,Q_out):
        '''
        This implements the finite difference methods for D matrix
        :param state_mom: The current momentum variable
        :param energy_rep: The repeated copy of mean energy with size :math:`num_CNN x dim`
        :param grad_U: The mean energy gradient
        :param state_mom_pre: The previous momentum variable
        :param Q_out: The output from Q matrix
        :param const: The offset for :math:`Q^2` term
        :return: out_D, grad_D_mom,grad_D_Q
        '''

        state_mom = torch.tensor(state_mom.data)
        Q_out = torch.tensor(Q_out.data)
        U_value = torch.tensor(energy_rep.data)  # num_chain x dim
        grad_U = self.grad_U_scale * torch.tensor(grad_U.data) ### num_chain x total_dim
        # Form the input and previous input
        input_NN = torch.tensor(torch.stack((U_value, state_mom, grad_U), dim=2).data)
        input_NN_pre = torch.tensor(torch.stack((U_value, state_mom_pre, grad_U), dim=2).data)

        # Compute the output
        out = self.D_scale * self.MLP.forward(input_NN)
        out_pre = self.D_scale * self.MLP.forward(input_NN_pre)# num_CNN x dim x 1
        out_pre = torch.squeeze(out_pre)

        out_D = self.offset + (Q_out ** 2) * self.scale_Q_term + torch.squeeze(out)
        out_D = torch.clamp(out_D, min=self.clamp_min, max=self.clamp_max)

        # Compute the finite approximation of gradient
        grad_D_mom = torch.tensor((torch.squeeze(out).data - out_pre.data) / (state_mom.data - state_mom_pre.data + 1e-7))
        grad_D_Q = torch.tensor(2 * Q_out.data * self.scale_Q_term)
        return out_D, grad_D_mom, grad_D_Q
class parallel_Gamma_eff:
    def __init__(self,flag_D=True):
        self.flag_D=flag_D
    def forward(self,grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q):
        '''
        This is to compute the :math:`\Gamma` matrix.
        :param grad_Q_mom: Gradient Q wrt momentum
        :param grad_Q_pos: Gradient Q wrt :math:`\theta`
        :param grad_D_mom: Gradient D wrt momentum
        :param grad_D_Q: Gradient D wrt Q
        :return: out1(for :math:`\theta` update ), out2(for momentum update)
        '''
        out1=-grad_Q_mom
        out2=grad_Q_pos
        if self.flag_D==True:
            out2 = grad_Q_pos + grad_D_mom + grad_D_Q * grad_Q_mom
        return out1,out2



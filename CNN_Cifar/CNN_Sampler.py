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
from CNN_Module import *
from CNN_Q_D import *
from Test_Module import *
from CNN_training_func import *
########################################################################################################################
# Sampler for SGHMC and SGLD
class SGHMC:
    def __init__(self,total_dim,CNN,test_mode='Cross Chain'):
        '''
        The SGHMC sampler class.
        :param total_dim: The total weight dimensionality for CNN
        :param CNN: Parallel CNN obj
        '''
        self.total_dim=total_dim
        self.CNN=CNN
        self.test_mode=test_mode
    def parallel_sample(self,state_pos,state_mom,loader,data_N,num_CNN=20,mom_resample=50,
                        total_step=1000,eps=0.001,alpha=0.1,beta=0.,sigma=1.,interval=100,flag_SGLD=False,
                        test_loader=None,data_len=10000.,Sequential_Accuracy=None):
        '''
        This implements the parallel sampling procedures of SGHMC for CNN Cifar-10

        Note: The learning rate are converted in the following way:

        .. math::

            \epsilon=\eta^2\;\;\;\;\; \\alpha=C\eta\;\;\;\;\; \eta=\sqrt{\\frac{lr}{N}}

        where lr is the per-batch learning rate, :math:`\epsilon` is the modified step size, :math:`\eta` is step size and :math:`\\alpha` is effective friction

        :param state_pos: The position variable (weight tensor) with size :math:`num_CNN x total_dim`
        :param state_mom: The momentum augmentation with size the same as state_pos
        :param loader: The training data loader
        :param data_N: The total training examples
        :param num_CNN: The number of parallel CNN to run
        :param mom_resample:
        :param total_step: The maximum running epochs for SGHMC
        :param eps: The modified step size: This equals to
        :param alpha: The friction term (not D matrix, but with )
        :param beta: The subsampling noise estimation default:0
        :param sigma: The variance of prior
        :param interval: Sampling interval
        :param flag_SGLD: Whether switch to SGLD (zero momentum SGHMC with correct noise)
        :param test_loader: The test data loader
        :param data_len: The number of test samples in test dataset
        :return: state_list: list containing the samples
        '''
        state_list=[]
        counter=0 #count total running steps
        for time_t in range(total_step):
            print('Epoch:%s'%(time_t+1))
            for data in enumerate(loader):
                if (counter+1)%1000==0:
                    print('Total Steps:%s'%(counter+1))
                x, y = data[1][0].cuda(), data[1][1].cuda()
                y_orig=torch.tensor(y.data.clone())
                y = torch.unsqueeze(y, dim=1)
                batch_y = int(y.shape[0])
                y = torch.tensor(torch.zeros(batch_y, 10).scatter_(1, y, 1)).float()# one hot vector



                # Momentum Resampling
                if (counter+1)%mom_resample==0:
                    state_mom=0.0*torch.randn(num_CNN,self.total_dim)
                # Position Update
                # Create Clone of state_pos
                state_pos=torch.tensor(state_pos.data+state_mom.data)
                state_pos_clone=torch.tensor(state_pos.data.clone(),requires_grad=True)
                # CNN Evaluation
                grad_U,_,energy,_=self.CNN.grad_CNN(x,y,state_pos_clone,data_N,sigma=sigma)
                #grad_U=grad_U.detach()
                energy=energy.detach()
                # Momentum update
                noise = torch.randn(num_CNN, self.total_dim)
                add_noise = float(np.sqrt(2 * alpha * eps - beta * eps)) * noise
                if flag_SGLD==True:
                    state_mom=torch.zeros(num_CNN,self.total_dim)
                state_mom=torch.tensor(state_mom.data-eps*grad_U.data-alpha*state_mom.data+add_noise.data)
                # Store the sample
                if (counter+1)%interval==0:
                    state_list.append(state_pos.detach())
                counter+=1
            if self.test_mode=='Cross Chain' and type(test_loader) != type(None) and (time_t + 1) % 1 == 0:
                # Print the accuracy across chains
                print(Test_Accuracy(self.CNN, test_loader, state_pos.data, test_number=data_len))
            elif test_mode=='Cross Time' and (time_t)%2 ==0 and type(Sequential_Accuracy)!=type(None):
                # Print the Sequential Accuracy
                print(Sequential_Accuracy.Test_Accuracy(state_pos.data))


        return state_list

########################################################################################################################
# Sampler for NNSGHMC
class NNSGHMC:
    def __init__(self,CNN,Q,D,Gamma):
        self.CNN=CNN
        self.Q=Q
        self.D=D
        self.Gamma=Gamma
        self.total_dim=self.CNN.get_dimension()
    def parallel_sample(self,state_pos,state_mom,B,loader,data_N,sigma=1.,num_CNN=20,total_step=10,limit_step=10000,eps=0.1,eps2=0.1,
                        TBPTT_step=20,coef=1.,sample_interval=10,sub_sample_number=8,mom_resample=100000,mode_train=True,
                        test_loader=None,data_len=10000.):
        '''
        This is to run the sampler dynamics.

        Note the discretizations in here are different to SGHMC. To convert them, use the following (eps is the one used in SGHMC):

        .. math::

            discretization=\sqrt{eps}\;\;\;\;\;\;\;\;\; discretization=\sqrt{\\frac{lr}{N}}

        :param state_pos: The :math:`\theta` variable
        :param state_mom: The momentum variable
        :param B: The noise estimation matrix, detault set to 0
        :param loader: Training data loader
        :param data_N: The size of training data
        :param sigma: The variance of prior
        :param num_CNN: Number of parallel CNN to run
        :param total_step: The maximum number of epoch to run
        :param limit_step: The maximum of steps to run
        :param eps: The initial discretization
        :param eps2: The discretization after burn-in
        :param TBPTT_step: The max step before stopping the gradient
        :param coef: The coef for dimensionality mismatch
        :param sample_interval: The thinning interval
        :param mom_resample: The number of steps before re-sampleing momentum
        :param mode_train: The mode of the sampler
        :param test_loader: The data loader for test set
        :param data_len: The data size of test set
        :return: state_list
        '''

        state_list = []
        state_mom_list=[]
        counter = 0
        counter_ELBO=0
        state_list_in_chain=[] # Store samples for In-Chain Loss
        # Run sampler
        for time_t in range(total_step):
            if (time_t+1)%1==0:
                print('epoch:%s'%(time_t+1))
            # Switching step size
            if (time_t+1)>=2:
                eps=eps2
            # Run sampler
            for data in enumerate(loader):
                if (counter+1)%50==0:
                    print('Step:%s'%(counter+1))
                # Stop if exceeds the step limits for training mode
                if mode_train==True and (counter+1)%limit_step==0:
                    break
                # Subsample the data
                x, y = torch.tensor(data[1][0].cuda()), data[1][1].cuda()
                y = torch.unsqueeze(y, dim=1)
                batch_y = int(y.shape[0])
                y = torch.tensor(torch.zeros(batch_y, 10).scatter_(1, y, 1)).float()

                # Clone the theta variable
                state_pos_clone=torch.tensor(state_pos.data,requires_grad=True)
                # Compute CNN statistics
                grad_U, grad_U_mod,energy , energy_mod = self.CNN.grad_CNN(x, y, state_pos_clone, data_N,coef=coef, sigma=sigma)

                energy = torch.tensor(energy.data)  # num_chain x 1
                grad_U = torch.tensor(grad_U.data)  # num_chain x dim

                mean_grad_U = torch.tensor(1. / data_N * grad_U.data)
                mean_grad_U_mod = torch.tensor(1. / data_N * grad_U_mod.data)
                # Mean Energy
                mean_energy_mod = Variable(1. / data_N * energy_mod.data)
                # Stop Gradient
                if (counter + 1) % TBPTT_step == 0:
                    state_pos = torch.tensor(state_pos.data, requires_grad=True)
                    state_mom = torch.tensor(state_mom.data, requires_grad=True)
                    # Compute In Chain Loss
                    if mode_train == True:
                        print('In Chain Loss')
                        grad_ELBO_In_Chain(self.CNN, x, y, data_N, state_list_in_chain, sub_sample_number=sub_sample_number, sigma=sigma)

                        # Clear In Chain Sample Storage
                        state_list_in_chain = []



                # Momentum Resampling
                if (counter + 1) % mom_resample == 0:
                    state_mom=0.001*torch.randn(num_CNN,self.total_dim,requires_grad=True)
                # Sampler Dynamics
                    # Compute preconditioned matrix
                Q_out, grad_Q_pos, grad_Q_mom, _ = self.Q.forward(state_mom, mean_energy_mod,
                                                                      mean_grad_U_mod,
                                                                      flag_graph=mode_train)
                D_out, grad_D_mom, grad_D_Q = self.D.forward(state_mom, mean_energy_mod, mean_grad_U,
                                                                 Q_out, flag_graph=mode_train)
                tau_out1, tau_out2 = self.Gamma.forward(grad_Q_mom, grad_Q_pos, grad_D_mom,
                                                        grad_D_Q)

                    # Compute dynamics
                G_noise = torch.randn(num_CNN, self.total_dim)
                noise = torch.sqrt(2. * eps * D_out - (eps ** 2) * B) * G_noise  ## chain x dim
                    # Momentum Update
                if mode_train==True:

                    state_mom=state_mom-eps*Q_out*grad_U-eps*D_out*state_mom+eps*tau_out2+noise # chain x dim
                else:
                    state_mom=torch.tensor(state_mom.data-eps*Q_out.data*grad_U.data-eps*D_out.data*state_mom.data+eps*tau_out2.data+noise.data,requires_grad=True) # chain x dim

                    # Compute Preconditioned matrix for position update

                Q_out_dash, grad_Q_pos, grad_Q_mom, _ = self.Q.forward(state_mom, mean_energy_mod,
                                                                      mean_grad_U_mod,
                                                                      flag_graph=mode_train)
                tau_out1, tau_out2 = self.Gamma.forward(grad_Q_mom, grad_Q_pos, grad_D_mom,
                                                            grad_D_Q)
                    # Update Position
                if mode_train == True:
                    state_pos = state_pos + eps * Q_out * state_mom + eps * tau_out1  # chain x dim
                    # Store for in chain loss
                    state_list_in_chain.append(state_pos)
                else:
                    state_pos = torch.tensor(state_pos.data + eps * Q_out_dash.data * state_mom.data + eps * tau_out1.data,
                                         requires_grad=True)
                # Accumulate the gradients
                if mode_train==True and (counter+1)%sample_interval==0:
                    # Cross Chain Loss
                    counter_ELBO = grad_ELBO(state_pos,self.CNN,x,y,data_N,counter_ELBO,limit_step,sample_interval,sigma=sigma)

                elif mode_train==False and (counter+1)%sample_interval==0:
                    state_list.append(torch.tensor(state_pos.data))
                # Add 1 to counter
                counter+=1
            # Test the accuracy
            if (time_t+1)%2==0 and type(test_loader)!=type(None):
                print(Test_Accuracy(self.CNN, test_loader, state_pos.data, test_number=data_len))
            # Stop the dynamics
            if mode_train==True and (counter+1)%limit_step==0:
                state_list.append(torch.tensor(state_pos.data))
                state_mom_list.append(torch.tensor(state_mom.data))
                break
        if mode_train==True:
            state_list.append(torch.tensor(state_pos.data))
            state_mom_list.append(torch.tensor(state_mom.data))
        return state_list,state_mom_list,counter_ELBO
    def parallel_sample_FD(self,state_pos,state_mom,B,loader,data_N,sigma=1.,num_CNN=50,total_step=10,limit_step=10000,eps=0.1,
                           eps2=0.1,coef=1.,sample_interval=10,mom_resample=1000000,flag_nan=True,
                           test_loader=None,data_len=10000.):
        state_list=[]
        counter=0
        flag_reset=False
        for time_t in range(total_step):
            if (time_t+1)%1==0:
                print('epoch:%s'%(time_t+1))
            if (time_t+1)>=2:
                eps=eps2
            for data in enumerate(loader):
                if (counter+1)%1000==0:
                    print('Step:%s'%(counter+1))
                x, y = torch.tensor(data[1][0].cuda()), data[1][1]
                y = torch.unsqueeze(y, dim=1).cuda()
                batch_y = int(y.shape[0])
                y = torch.tensor(torch.zeros(batch_y, 10).scatter_(1, y, 1)).float()
                state_pos_clone = torch.tensor(state_pos.data, requires_grad=True)
                # Store previous U
                if counter>0:
                    energy_pre=torch.tensor(mean_energy_mod.data)
                grad_U, grad_U_mod, energy, energy_mod=self.CNN.grad_CNN(x, y, state_pos_clone, data_N,coef=coef, sigma=sigma)
                # Pre-process the necessary statistics
                energy = torch.tensor(energy.data)  # num_chain x 1
                grad_U = torch.tensor(grad_U.data)  # num_chain x dim
                mean_grad_U = torch.tensor(1. / data_N * grad_U.data)
                mean_grad_U_mod = torch.tensor(1. / data_N * grad_U_mod.data)
                mean_energy_mod = torch.tensor(1. / data_N * energy_mod.data)
                mean_energy_mod_rep = torch.tensor(mean_energy_mod.repeat(1, self.total_dim).data)
                # Momentum re-sampling
                if (counter+1)%mom_resample==0:
                    flag_reset=True
                    state_mom=0.01*torch.randn(num_CNN,self.total_dim)


                # Evaluate preconditioned Q matrix
                if counter>0 and flag_reset==False:
                    Q_out_pre=torch.tensor(Q_out.data)
                    Q_out,grad_Q_pos,grad_Q_mom=self.Q.finite_diff_forward(state_mom,mean_energy_mod,mean_energy_mod_rep,mean_grad_U_mod,state_mom_pre,energy_pre,
                                                                           Q_dash_pre,Q_out_pre,flag_dash=False)

                else:
                    Q_out, grad_Q_pos, grad_Q_mom, _ = self.Q.forward(state_mom, mean_energy_mod,
                                                                      mean_grad_U_mod,
                                                                      flag_graph=False)
                # Debug if nan happens
                if flag_nan==True and (counter+1)%100==0:
                    assert np.sum(np.isnan(grad_Q_mom.cpu().data.numpy()))==0,'NaN occurs at counter %s'%(counter+1)
                # Evaluate preconditioned D matrix
                if counter>0:
                    D_out,grad_D_mom,grad_D_Q=self.D.finite_diff_forward(state_mom,mean_energy_mod_rep,mean_grad_U_mod,state_mom_pre,Q_out)
                else:
                    D_out,grad_D_mom,grad_D_Q=self.D.forward(state_mom,mean_energy_mod,mean_grad_U,Q_out,flag_graph=False)
                # Update Momentum
                tau_out1, tau_out2 = self.Gamma.forward(grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q)
                G_noise =torch.tensor(torch.randn(num_CNN, self.total_dim))

                noise = torch.tensor(torch.sqrt(2. * eps * D_out.data - (eps ** 2) * B.data)) * G_noise  ## chain x dim

                state_mom_pre = torch.tensor(state_mom.data)
                state_mom = torch.tensor(
                    state_mom.data - eps * Q_out.data * grad_U.data - eps * D_out.data * state_mom.data + eps * tau_out2.data + noise.data)  # chain x dim
                # Q' evaluation
                if counter>0 and flag_reset==False:
                    Q_out_dash,_,grad_Q_mom=self.Q.finite_diff_forward(state_mom,mean_energy_mod,mean_energy_mod_rep,mean_grad_U_mod,state_mom_pre,energy_pre,
                                                                           Q_dash_pre,Q_out,flag_dash=True)
                    Q_dash_pre=torch.tensor(Q_out_dash.data)
                else:
                    Q_out_dash,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom, mean_energy_mod,
                                                                      mean_grad_U_mod,
                                                                      flag_graph=False)
                    Q_dash_pre=torch.tensor(Q_out_dash.data)
                    flag_reset=False
                tau_out1,tau_out2=self.Gamma.forward(grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q)
                # Position update
                state_pos = torch.tensor(state_pos.data + eps * Q_out_dash.data * state_mom.data + eps * tau_out1.data,
                                     requires_grad=True)
                # Store samples
                if (counter+1)%sample_interval==0:
                    state_list.append(torch.tensor(state_pos.data))
                counter+=1
            if (time_t+1)%2==0 and type(test_loader)!=type(None):
                print(Test_Accuracy(self.CNN, test_loader, state_pos.data, test_number=data_len))
        return state_list









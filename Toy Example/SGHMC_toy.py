import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad


class SGHMC:    
    def __init__(self,dim,U,C,B):
        '''
        C is the positive definite friction matrix, currently support diagonal C 
        B is the positive definite subsampling noise covariance matrix, currently support diagonal B
        U is the subsampled energy functon 
        dim is the dimension of position variable
        '''
        self.dim=dim
        self.U=U
        self.C=C
        self.B=B
    def sample(self,state_pos,state_mom,total_step=1000,eps=0.1,flag_manual_noise=False):
        state_list=[]
        state_mom_list=[]
        for time in range(total_step):
            if (time+1)%200==0:
                print('Step:%s'%(time+1))
            #### Position update ####
            state_pos=Variable(state_pos.data+eps*state_mom.data)
            state_list.append(state_pos)
            #### Evaluate grad U ####
            state_pos_clone=Variable(state_pos.data.clone(),requires_grad=True)
            energy=self.U.forward(state_pos_clone)
            grad_U=Variable(grad(energy,state_pos_clone)[0].data.clone())
            if flag_manual_noise==True:
                noise=Variable(torch.randn(self.dim,1))
                grad_U=grad_U+1*noise
            #### Update mom ####
            noise=Variable(torch.randn(self.dim,1))
            add_noise=torch.sqrt(eps*2*self.C-(eps**2)*self.B).matmul(noise)
            state_mom=Variable(state_mom.data-eps*grad_U.data-eps*C.data.matmul(state_mom.data)+add_noise.data)
            
            state_mom_list.append(state_mom)
        return state_list,state_mom_list
    def parallel_sample(self,state_pos,state_mom,num_chain=50,total_step=1000,eps=0.1,flag_manual_noise=False,inject_scale=1.):
        '''
        !!!only support diagonal C and B!!!!
        state_pos is dim x chain
        state_mom is dim x chain
        '''
        state_list=[]
        state_mom_list=[]
        for time in range(total_step):
            if (time+1)%200==0:
                print('Step:%s'%(time+1))
            #### Position update #####
            state_pos=state_pos+eps*state_mom
            state_list.append(state_pos)
            #### Evaluate grad U ####
            state_pos_clone=Variable(state_pos.data.clone(),requires_grad=True)
            energy=self.U.forward(state_pos_clone)
            grad_U=Variable(grad(energy,state_pos_clone,torch.ones(energy.data.shape))[0].data.clone())
            if flag_manual_noise==True:
                noise=Variable(torch.randn(self.dim,num_chain))
                grad_U=grad_U+inject_scale*noise
            #### Update mom ######
            noise=Variable(torch.randn(self.dim,num_chain))
            add_noise=torch.unsqueeze(torch.diag(torch.sqrt(eps*2*self.C-(eps**2)*self.B)),dim=1)*noise#.matmul(noise)
            state_mom=state_mom-eps*grad_U-eps*torch.unsqueeze(torch.diag(self.C),dim=1)*state_mom+add_noise
            state_mom_list.append(state_mom)
        return state_list, state_mom_list
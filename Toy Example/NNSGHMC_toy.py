import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad


class MLP(nn.Module):
    def __init__(self,input_dim,hidden,out_size=1):
        super(MLP,self).__init__()
        self.input_dim=input_dim
        self.hidden=hidden
        self.out_func=nn.Linear(hidden,out_size)
        self.features=nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_size)
        )
    def forward(self,x):
        out=self.features(x)
        return out    
class Positive_MLP(nn.Module):
    def __init__(self,input_dim,hidden,out_size=1):
        super(Positive_MLP,self).__init__()
        self.input_dim=input_dim
        self.hidden=hidden
        self.out_func=nn.Linear(hidden,out_size)
        self.features=nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_size), 
        )
    def forward(self,x):
        out=torch.abs(self.features(x))
        return out
class parallel_Q_eff:
    def __init__(self,dim,Q_MLP,U,num_chain,clamp=100000,dim_pen=1.):
        self.Q_MLP=Q_MLP
        self.U=U
        self.clamp=clamp
        self.dim=dim
        self.dim_pen=dim_pen
        self.num_chain=num_chain
        self.U_value=None
        self.grad_U=None
        self.out=None
        self.input_NN=None
        
    def forward(self,state_mom,state_pos,energy,grad_U):
        '''
        Input is now U and momentum
        
        '''
        total_dim=int(self.dim*self.num_chain)
        ##### NN ###########
        ##### Not flow through the Q net
        state_pos=Variable(state_pos.data,requires_grad=True)
        state_pos=state_pos.t().contiguous().view(total_dim,1) ##### total dim x 1
        state_mom=Variable(state_mom.data,requires_grad=True)
        state_mom=state_mom.t().contiguous().view(total_dim,1) ##### total dim x 1
        #####
        U_value=Variable(torch.unsqueeze(energy,dim=0).data)/self.dim_pen#1 x chain 
        U_value=U_value.repeat(self.dim,1)# dim x chain 
        U_value=U_value.t().contiguous().view(total_dim,1) #### total dim x 1
        
        
        grad_U=Variable(grad_U.data)# dim x chain 
        
        input_NN=Variable(torch.cat((U_value,state_mom),dim=1).data,requires_grad=True) ### total dim x 2 
        out=self.Q_MLP.forward(input_NN) ## total dim x 1
        out=torch.clamp(out, min=-self.clamp,max=self.clamp)
        out_Q=out.view(self.num_chain,self.dim).t() #### dim x chain 
        
        
        
        grad_Q=grad(out,input_NN,torch.ones(out.data.shape),allow_unused=True,create_graph=True)[0]
        grad_Q_mom=grad_Q[:,1:].contiguous().view(self.num_chain,self.dim).t() #### dim x chain 
        grad_Q_pos=grad_Q[:,0:1].contiguous().view(self.num_chain,self.dim).t()*grad_U/self.dim_pen
        return out_Q,grad_Q_pos,grad_Q_mom,grad_Q
class parallel_D_eff:
    def __init__(self,dim,D_MLP,U,num_chain,clamp=10000000,dim_pen=1.):
        self.D_MLP=D_MLP
        self.U=U
        self.dim=dim
        self.dim_pen=dim_pen
        self.clamp=clamp
        self.num_chain=num_chain
        self.U_value=None
        self.grad_U=None
        self.out=None
        self.input_NN=None
    def forward(self,state_mom,state_pos,energy,grad_U):
        '''
        Input is now U and momentum and Gradient
        
        '''
        total_dim=int(self.dim*self.num_chain)
        ##### NN ###########
        ##### Not flow through the Q net
        state_pos=Variable(state_pos.data,requires_grad=True)
        state_mom=Variable(state_mom.data,requires_grad=True)
        state_pos=state_pos.t().contiguous().view(total_dim,1) ### total x 1
        state_mom=state_mom.t().contiguous().view(total_dim,1) ### total x 1
        #####
        U_value=Variable(torch.unsqueeze(energy,dim=0).data)/self.dim_pen#1 x chain 
        U_value=U_value.repeat(self.dim,1)# dim x chain 
        U_value=U_value.t().contiguous().view(total_dim,1) #### total dim x 1
        
        grad_U=Variable(grad_U.data).t().contiguous().view(total_dim,1)### total x 1
        
        
        input_NN=Variable(torch.cat((U_value,state_mom,grad_U),dim=1).data.clone(),requires_grad=True)
        out=self.D_MLP.forward(input_NN) ### total x 1
        out=torch.clamp(out, min=-self.clamp,max=self.clamp)
        out_D=out.view(self.num_chain,self.dim).t() #### dim x chain 
       
        
        grad_D_mom=grad(out,input_NN,torch.ones(out.data.shape),allow_unused=True,create_graph=True)[0][:,1:2]#### total x 1
        grad_D_mom=grad_D_mom.contiguous().view(self.num_chain,self.dim).t()
        return out_D,grad_D_mom 
class parallel_Gamma_eff:
    def __init__(self,dim,Q_NN,D_NN=None):
        self.dim=dim
        self.Q_NN=Q_NN
        self.D_NN=D_NN
    def forward(self,state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom):
        '''
        This is for Q input is U and momentum
        '''
        out1=-grad_Q_mom
        out2=grad_Q_pos
        if self.D_NN!=None:
            out2=grad_Q_pos+grad_D_mom
        return out1,out2
class NN_SGHMC:
    def __init__(self,dim,U,D,Q,Gamma):
        self.dim=dim
        self.U=U
        self.D=D
        self.Q=Q
        self.Gamma=Gamma
    def parallel_sample(self,state_pos,state_mom,B,num_chain=50,total_step=1000,eps=0.1,flag_manual_noise=False,TBPTT_step=10,mom_resample=20,mom_scale=1.,inject_scale=1.,mode_train=True):
        '''
        TBPTT_step: iterations before stop gradient
        mode_train: Set to True during training but False during evaluation
        
        '''
        state_list=[]
        state_mom_list=[]
        for time in range(total_step):
            if (time+1)%200==0:
                print('Step:%s'%(time+1))
            ###### Evaluate Energy function and Grad energy ####
            state_pos_clone=Variable(state_pos.data,requires_grad=True)
            # Energy has the form [chain] 
            energy=self.U.forward(state_pos_clone)
            # Now grad_U has the form dim x chain 
            grad_U=Variable(grad(energy,state_pos_clone,torch.ones(energy.data.shape))[0].data)
            energy=Variable(energy.data)
            if flag_manual_noise==True:
                noise=Variable(torch.randn(self.dim,num_chain))
                grad_U=grad_U+inject_scale*noise
            ######### Stop Gradient ###########
            if (time+1)%TBPTT_step==0:
                state_pos=Variable(state_pos.data,requires_grad=True)
                state_mom=Variable(state_mom.data,requires_grad=True)
                
            if (time+1)%mom_resample==0:
                state_mom=Variable(mom_scale*torch.randn(self.dim,num_chain),requires_grad=True)
            ###### mom update ########
            Q_out,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,energy,grad_U)
            D_out,grad_D_mom=self.D.forward(state_mom,state_pos,energy,grad_U)
            tau_out1,tau_out2=self.Gamma.forward(state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom)
            
            G_noise=Variable(torch.randn(self.dim,num_chain))
            
            noise=torch.sqrt(2.*eps*D_out-(eps**2)*torch.unsqueeze(torch.diag(B),dim=1))*G_noise ## dim x chain
            
            if mode_train==True:
                state_mom=state_mom-eps*Q_out*grad_U-eps*D_out*state_mom+eps*tau_out2+noise
            else:
                state_mom=Variable(state_mom.data-eps*Q_out.data*grad_U.data-eps*D_out.data*state_mom.data+eps*tau_out2.data+noise.data,requires_grad=True)
            state_mom_list.append(state_mom)
            ###### position update ######
            Q_out,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,energy,grad_U)
            tau_out1,tau_out2=self.Gamma.forward(state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom)
            if mode_train==True:
                state_pos=state_pos+eps*Q_out*state_mom+eps*tau_out1
            else:
                state_pos=Variable(state_pos.data+eps*Q_out.data*state_mom.data+eps*tau_out1.data,requires_grad=True)
            if mode_train==False and (time+1)%200==0:
                ##### Check if nan happens #####
                assert torch.sum(torch.isnan(state_pos))<1,'Nan occurs, try to re-run, reduce step size or reduce clamp value'
            state_list.append(state_pos)
        return state_list,state_mom_list

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
import pdb

from BNN_Model_def import *
from BNN_Util import *
from BNN_Q_D import *
from BNN_training_func import *

class SGHMC:    
    def __init__(self,total_dim,BNN_obj):
        '''
        C is the positive definite friction matrix, currently support diagonal C 
        B is the positive definite subsampling noise covariance matrix, currently support diagonal B
        BNN is BNN Model 
        dim is the dimension of position variable
        '''
        self.dim=total_dim
        self.U=BNN_obj
        #self.C=C
        #self.B=B
    def parallel_sample(self,state_pos,state_mom,loader,data_N,mom_resample=50,num_chain=20,total_step=1000,eps=0.01,alpha=0.01,beta=0.,sigma=1.,interval=100,flag_SGLD=False,test_loader=None,data_len=10000.):
        '''
        
        !!!only support diagonal C and B!!!!
        state_pos is chain x total_dim
        state_mom is chain x total_dim
        X is the batched data (dataloader)
        y is corresponding label (dataloader)
        
        '''
        state_list=[]
        state_mom_list=[]
        energy_list=[]
        counter=0
        time_list=[]
        
        alpha=alpha
        
        for time_t in range(total_step):
            #Tracer()()
            st=time.time()
            print('ep:%s'%(time_t+1))
            for data in enumerate(loader):
                
                if (counter+1)%1000==0:
                    #Tracer()()
                    pass
                    print('Step:%s'%(counter+1))
                X,y=Variable(data[1][0].cuda()),data[1][1]
                #pdb.set_trace()
                y=torch.unsqueeze(y,dim=1).cuda()
                batch_y=int(y.shape[0])
                y = Variable(torch.zeros(batch_y, self.U.dim_out).scatter_(1, y, 1)).float()
                
                if (counter+1)%mom_resample==0:
                    state_mom=Variable(0.0001*torch.randn(num_chain,self.dim),requires_grad=True)
                
                
                #### Position update #####
                state_pos=Variable(state_pos.data+state_mom.data)
                if (counter+1)%interval==0:
                    pass
                    #state_list.append(state_pos)
                #### Evaluate grad U ####
                state_pos_clone=Variable(state_pos.data.clone(),requires_grad=True)
                grad_U,energy,_,_=self.U.grad_BNN(X,y,state_pos_clone,data_N,sigma=sigma) ####
                grad_U=Variable(grad_U.data)### chain x total_dim
                #Tracer()()
                energy=Variable(energy.data)### chain x 1
                energy_list.append(energy.data)
                if (counter+1)%interval==0:
                    pass
                    #energy_list.append(energy.data)
                #### Update mom ######
                noise=Variable(torch.randn(num_chain,self.dim))
                #add_noise=torch.unsqueeze(torch.diag(torch.sqrt(eps*2*self.C-(eps**2)*self.B)),dim=1)*noise#num_chain x total_dim
                #state_mom=Variable(0*torch.randn(num_chain,self.dim),requires_grad=True)
                add_noise=float(np.sqrt(2*alpha*eps-beta*eps))*noise
                if flag_SGLD==True:
                    state_mom=Variable(torch.zeros(num_chain,self.dim))
                state_mom=Variable(state_mom.data-eps*grad_U.data-alpha*state_mom.data+add_noise.data)
                #Tracer()
                if (counter+1)%interval==0:
                    state_list.append(state_pos)
                counter+=1
                
            ed=time.time()-st
            
            time_list.append(ed)
            if (time_t+1)%10==0 and type(test_loader)!=type(None):
                print(Test_accuracy(test_loader,self.U,state_pos,data_number=data_len))
            #state_list.append(state_pos)
        return state_list, state_mom_list,energy_list,time_list
    
############################## NNSGHMC ###########################################################
class NN_SGHMC:
    def __init__(self,total_dim,BNN_obj,D,Q,Gamma):
        '''
        if run parallel chain, currently support only diagonal D and Q
        '''
        self.dim=total_dim
        self.BNN_obj=BNN_obj
        self.D=D
        self.Q=Q
        self.Gamma=Gamma
    def parallel_sample(self,state_pos,state_mom,B,loader,data_N,sigma=1.,num_chain=50,total_step=10,limit_step=100,eps=0.1,eps2=0.1,TBPTT_step=10,coef=1.,sample_interval=3,mom_resample=2000000,mom_scale=1.,mode_train=True,const_Q=0.,const_D=0.,flag_finite=False,test_loader=None,data_len=10000.):
        '''


        B is the estimated noise matrix 
        state_pos is chain x dim
        state_mom is chain x dim
        !!!! Only support diagonal form Q and D !!!!
        
        '''
        state_list=[]
        state_mom_list=[]
        out_Q_list=[]
        energy_list=[]
        grad_energy_list=[]
        A_list=[]
        time_list=[]
        counter=0
        
        for time_t in range(total_step):
            if (time_t+1)%1==0:
                print('epoch:%s'%(time_t+1))
            if (time_t+1)>=2:
                eps=eps2
            st=time.time()
            for data in enumerate(loader):
                
                if (counter+1)%50==0:
                    print('Step:%s'%(counter+1))
                if mode_train==True:
                    if (counter+1)%limit_step==0:
                        break
                
                X,y=Variable(data[1][0].cuda()),data[1][1]
                y=torch.unsqueeze(y,dim=1).cuda()
                batch_y=int(y.shape[0])
                y = Variable(torch.zeros(batch_y, self.BNN_obj.dim_out).scatter_(1, y, 1)).float()
                
                
                
                ###### Evaluate Energy function and Grad energy ####
                state_pos_clone=Variable(state_pos.data,requires_grad=True)
                # Energy has the form [chain] 
                
                ##### set Q_pre Value ####
                
                
                    
                            
                if counter>0:
                    energy_pre=Variable(mean_energy_mod.data)
                    
                if flag_finite==True and counter>0:
                    Q_out,_,_,_=self.Q.forward(state_mom_pre,state_pos_pre,energy_pre,mean_grad_U_mod,const=const_Q,flag_graph=mode_train,flag_finite=True,state_mom_pre=state_mom_pre,energy_pre=Variable(torch.zeros(1)),Q_pre=Variable(torch.zeros(1)))
                    Q_pre=Variable(Q_out.data)
                    
                grad_U,energy_mod,grad_U_mod,energy=self.BNN_obj.grad_BNN(X,y,state_pos_clone,coef=coef,data_N=data_N,sigma=sigma)
                
                # Now grad_U has the form dim x chain 
                energy=Variable(energy.data) # num_chain x 1
                grad_U=Variable(grad_U.data) # num_chain x dim 
                
                mean_grad_U=Variable(1./data_N*grad_U.data)
                mean_grad_U_mod=Variable(1./data_N*grad_U_mod.data)
                mean_energy=Variable(1./data_N*energy.data)
                
                ##### Debug 
                if counter>0:
                    mean_energy_pre=Variable(mean_energy_mod.data)
                ######
                
                mean_energy_mod=Variable(1./data_N*energy_mod.data)
                #Tracer()()
                #energy_list.append(energy)
                #grad_energy_list.append(grad_U)
                if (counter+1)%TBPTT_step==0:
                    
                    state_pos=Variable(state_pos.data,requires_grad=True)
                    state_mom=Variable(state_mom.data,requires_grad=True)

                if (counter+1)%mom_resample==0:
                    raise NotImplementedError
                    #state_mom=Variable(mom_scale*torch.randn(self.dim,num_chain),requires_grad=True)
                ###### mom update ########
                if flag_finite==False or mode_train==True:
#                     ''' DEBUG'''
#                     if counter>0:
#                         Q_out_pre=Variable(Q_out.data)
#                     ######################
                    Q_out,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=mode_train)
#                     '''
#                                     Debug
#                     '''
#                     if counter>0:
#                         grad_Q_pos_true=Variable(grad_Q_pos.data)
#                         grad_Q_mom_true=Variable(grad_Q_mom.data)
#                         grad_Q_pos=Variable(((Q_out-Q_out_dash)/(mean_energy_mod-energy_pre+1e-7)*mean_grad_U_mod).data)
#                         grad_Q_mom=Variable(((Q_out_dash-Q_out_pre)/(state_mom-state_mom_pre+1e-7)).data)
#                         if np.sum(np.isnan(grad_Q_mom.cpu().data.numpy()))>0:
#                             print('NAN occur')
#                             Tracer()()
#                     if counter>100:
#                         pass
#                         #Tracer()()
#                     ######################
#                     ''' Debug'''
#                     if counter>0:
#                         D_out_pre=Variable(D_out.data)
                    ######################
                    D_out,grad_D_mom,grad_D_Q=self.D.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U,Q_out,grad_Q_mom,const=const_D,flag_graph=mode_train)
#                     '''
#                                 Debug
                    
#                     '''
#                     if counter>0:
#                         grad_D_mom_true=Variable(grad_D_mom.data)
#                         D_out_pseudo,_,_=self.D.forward(state_mom_pre,state_pos,mean_energy_mod,mean_grad_U,Q_out,grad_Q_mom,const=const_D,flag_graph=mode_train)
#                         grad_D_mom=Variable((D_out.data-D_out_pseudo.data)/(state_mom.data-state_mom_pre.data+1e-7))
#                     if counter>100:
#                         pass
#                         #Tracer()()
                        
#                     #######################
                else:
                    if counter==0:
                        Q_out,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=mode_train)
                        D_out,grad_D_mom,grad_D_Q=self.D.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U,Q_out,grad_Q_mom,const=const_D,flag_graph=mode_train)
                        #Q_pre=Variable(Q_out.data)
                    else:
                        Q_out,grad_Q_pos,grad_Q_mom=self.Q.finite_diff_forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,state_mom_pre,state_pos_pre,energy_pre,Q_pre,const=2.)
                        D_out,grad_D_mom,grad_D_Q=self.D.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U,Q_out,grad_Q_mom,const=const_D,flag_graph=mode_train,flag_finite=True)
                        #Q_pre=Variable(Q_out.data)
                    
                if (counter)>=1800 and (counter)%10==0:
                    pass
                    #Tracer()()
                if (counter)%40==0 and counter>=900:
                    pass
                    #Tracer()()
                ####### Debug #########
#                 Q_out=Variable(torch.zeros(grad_U.data.shape))
#                 grad_Q_pos=Variable(torch.zeros(state_pos.data.shape))
#                 grad_Q_mom=Variable(torch.zeros(state_pos.data.shape))
#                 D_out=Variable(torch.zeros(state_pos.data.shape))
#                 grad_D_mom=Variable(torch.zeros(state_pos.data.shape))
                
                #######################
                
                tau_out1,tau_out2=self.Gamma.forward(state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q,const_D,Q_out,clamp_min=self.D.clamp_min)
                
                G_noise=Variable(torch.randn(num_chain,self.dim))

                noise=torch.sqrt(2.*eps*D_out-(eps**2)*B)*G_noise ## chain x dim
                
                if mode_train==True:
                    state_mom=state_mom-eps*Q_out*grad_U-eps*D_out*state_mom+eps*tau_out2+noise # chain x dim
                else:
                    state_mom_pre=Variable(state_mom.data)
                    state_mom=Variable(state_mom.data-eps*Q_out.data*grad_U.data-eps*D_out.data*state_mom.data+eps*tau_out2.data+noise.data,requires_grad=True) # chain x dim
                
                #state_mom_list.append(state_mom)
                
                ###### position update ######
                if flag_finite==False or mode_train==True:
                    Q_out_dash,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=mode_train)
#                     '''      
                    
                    
#                                     Debug
                    
#                     '''
#                     Q_out_dash=Variable(Q_out_dash.data)
#                     if counter>0:
#                         grad_Q_mom_true=Variable(grad_Q_mom.data)
#                         grad_Q_mom=Variable((Q_out_dash.data-Q_out.data)/(state_mom.data-state_mom_pre.data+1e-7))
#                     if counter>100:
#                         pass
#                         #Tracer()()
                        
#                     ########################
                    
                    tau_out1,tau_out2=self.Gamma.forward(state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q,const_D,Q_out_dash,clamp_min=self.D.clamp_min)
                else:
                    if counter==0:
                        Q_out,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=mode_train,flag_finite=False,state_mom_pre=state_mom_pre,energy_pre=0.,Q_pre=0)
                        #Q_pre=Variable(Q_out.data)
                    else:
                        Q_out,grad_Q_pos,grad_Q_mom=self.Q.finite_diff_forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,state_mom_pre,state_pos_pre,energy_pre,Q_pre)#Q_out,grad_Q_pos,_,_=self.Q.forward(state_mom_pre,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=mode_train,flag_finite=False,state_mom_pre=state_mom_pre,energy_pre=energy_pre,Q_pre=Q_pre)
                        #Q_pre=Variable(Q_out.data)
                #out_Q_list.append(Q_out)
                if mode_train==True:
                    state_pos=state_pos+eps*Q_out*state_mom+eps*tau_out1 #chain x dim
                else:
                    state_pos_pre=Variable(state_pos.data)
                    state_pos=Variable(state_pos.data+eps*Q_out_dash.data*state_mom.data+eps*tau_out1.data,requires_grad=True)
                
                ######### Compute the gradient of loss ######
                if mode_train==True:
                    if (counter+1)%sample_interval==0:
                        grad_total=grad_ELBO(state_pos,self.BNN_obj,X,y,self.Q.Q_MLP,self.D.D_MLP,data_N=data_N,sigma=sigma)
                        A_list.append(grad_total)
                if mode_train==False:
                    if (counter+1)%sample_interval==0:
                        state_list.append(Variable(state_pos.data))
                    #state_mom_list.append(Variable(state_mom.data))
                counter+=1
                
                
            ed=time.time()-st
            time_list.append(ed)
            if (time_t+1)%10==0 and type(test_loader)!=type(None):
                print(Test_accuracy(test_loader,self.BNN_obj,state_pos,data_number=data_len))
            if mode_train==True:
                if (counter+1)%limit_step==0:
                    state_list.append(Variable(state_pos.data))
                    state_mom_list.append(Variable(state_mom.data))
                    break
                    
                        
        return state_list,state_mom_list,out_Q_list,energy_list,grad_energy_list,A_list,time_list
    
    def parallel_sample_FD(self,state_pos,state_mom,B,loader,data_N,sigma=1.,num_chain=50,total_step=10,limit_step=100,eps=0.1,eps2=0.1,coef=1.,sample_interval=10,const_Q=0.,const_D=0.,flag_nan=True,test_loader=None,data_len=10000.):
        state_list=[]
        time_list=[]
        counter=0
        for time_t in range(total_step):
            if (time_t+1)%1==0:
                print('epoch:%s'%(time_t+1))
            if (time_t+1)>=2:
                eps=eps2
            st=time.time()
            for data in enumerate(loader):
                
                if (counter+1)%1000==0:
                    print('Step:%s'%(counter+1))
                
                X,y=Variable(data[1][0].cuda()),data[1][1]
                y=torch.unsqueeze(y,dim=1).cuda()
                batch_y=int(y.shape[0])
                y = Variable(torch.zeros(batch_y, self.BNN_obj.dim_out).scatter_(1, y, 1)).float()

                state_pos_clone=Variable(state_pos.data,requires_grad=True)
                if counter>0:
                    energy_pre=Variable(mean_energy_mod.data)                   
                grad_U,energy_mod,grad_U_mod,energy=self.BNN_obj.grad_BNN(X,y,state_pos_clone,coef=coef,data_N=data_N,sigma=sigma)
                # Now grad_U has the form dim x chain 
                energy=Variable(energy.data) # num_chain x 1
                grad_U=Variable(grad_U.data) # num_chain x dim 
                mean_grad_U=Variable(1./data_N*grad_U.data)
                mean_grad_U_mod=Variable(1./data_N*grad_U_mod.data)
                mean_energy=Variable(1./data_N*energy.data)
                mean_energy_mod=Variable(1./data_N*energy_mod.data)
                mean_energy_mod_rep=Variable(mean_energy_mod.repeat(1,self.dim).data)
                ########### FD on Q ##############
                if counter>0:
                    Q_out_pre=Variable(Q_out.data)
                    Q_out,grad_Q_pos,grad_Q_mom=self.Q.finite_diff_forward(state_mom,state_pos,mean_energy_mod,mean_energy_mod_rep,mean_grad_U_mod,state_mom_pre,energy_pre,Q_dash_pre,Q_out_pre,const=const_Q,flag_dash=False)
                else:
                    Q_out,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=False)
                    
                    
                if flag_nan==True and (counter+1)%100==0:
                    assert np.sum(np.isnan(grad_Q_mom.cpu().data.numpy()))==0,'NaN occurs at counter %s'%(counter+1)
                        
                ########### FD on D ##############
                if counter>0:
                    D_out,grad_D_mom,grad_D_Q=self.D.finite_diff_forward(state_mom,mean_energy_mod,mean_energy_mod_rep,mean_grad_U_mod,state_mom_pre,Q_out,const=const_D)
                else:
                    D_out,grad_D_mom,grad_D_Q=self.D.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U,Q_out,grad_Q_mom,const=const_D,flag_graph=False)
                ########## Update Mometnum ######
                tau_out1,tau_out2=self.Gamma.forward(state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q,const_D,Q_out,clamp_min=self.D.clamp_min)
                
                G_noise=Variable(torch.randn(num_chain,self.dim))

                noise=Variable(torch.sqrt(2.*eps*D_out.data-(eps**2)*B.data))*G_noise ## chain x dim
                
                state_mom_pre=Variable(state_mom.data)
                state_mom=Variable(state_mom.data-eps*Q_out.data*grad_U.data-eps*D_out.data*state_mom.data+eps*tau_out2.data+noise.data) # chain x dim
                ########## Q dash ##############
                if counter>0:
                    Q_out_dash,_,grad_Q_mom=self.Q.finite_diff_forward(state_mom,state_pos,mean_energy_mod,mean_energy_mod_rep,mean_grad_U_mod,state_mom_pre,energy_pre,Variable(torch.zeros(1)),Q_out,const=const_Q,flag_dash=True)
                    Q_dash_pre=Variable(Q_out_dash.data)
                else:
                    Q_out_dash,grad_Q_pos,grad_Q_mom,_=self.Q.forward(state_mom,state_pos,mean_energy_mod,mean_grad_U_mod,const=const_Q,flag_graph=False)
                    Q_dash_pre=Variable(Q_out_dash.data)
                    
                tau_out1,tau_out2=self.Gamma.forward(state_mom,state_pos,grad_Q_mom,grad_Q_pos,grad_D_mom,grad_D_Q,const_D,Q_out_dash,clamp_min=self.D.clamp_min)
                ######### Update Position ########
                state_pos_pre=Variable(state_pos.data)
                state_pos=Variable(state_pos.data+eps*Q_out_dash.data*state_mom.data+eps*tau_out1.data,requires_grad=True)
                if (counter+1)%sample_interval==0:
                    state_list.append(state_pos)
                counter+=1
            ed=time.time()-st
            if (time_t+1)%10==0 and type(test_loader)!=type(None):
                print(Test_accuracy(test_loader,self.BNN_obj,state_pos,data_number=data_len))
            time_list.append(ed)
        return state_list,time_list
    
class PSGLD:    
    def __init__(self,total_dim,BNN_obj):
        '''
        BNN is BNN Model 
        dim is the dimension of position variable
        '''
        self.dim=total_dim
        self.U=BNN_obj
        #self.C=C
        #self.B=B
    def parallel_sample(self,state_pos,loader,data_N,num_chain=50,total_step=1000,eps=0.01,exp_term=0.99,lamb=1e-5,sigma=1.,interval=100,test_loader=None,data_len=10000.):
        '''
        
        Similar setting as SGHMC but with exp_term controls the exponential decay
        
        '''
        state_list=[]
        energy_list=[]
        counter=0
        time_list=[]
        
        
        for time_t in range(total_step):
            #Tracer()()
            st=time.time()
            print(time_t)
            for data in enumerate(loader):
                
                if (counter+1)%1000==0:
                    #Tracer()()
                    pass
                    print('Step:%s'%(counter+1))
                X,y=Variable(data[1][0].cuda()),data[1][1]

                y=torch.unsqueeze(y,dim=1).cuda()
                batch_y=int(y.shape[0])
                y = Variable(torch.zeros(batch_y, self.U.dim_out).scatter_(1, y, 1)).float()
                
                #### Evaluate grad U ####
                state_pos_clone=Variable(state_pos.data.clone(),requires_grad=True)
                grad_U,energy,_,_=self.U.grad_BNN(X,y,state_pos_clone,data_N,sigma=sigma) ####
                grad_U=Variable(grad_U.data)### chain x total_dim
                mean_grad_U=Variable(1./data_N*(-grad_U.data-(-state_pos.data/(sigma**2))))
                #Tracer()()
                energy=Variable(energy.data)### chain x 1
                mean_energy=Variable(1./data_N*energy.data)
                
                energy_list.append(energy.data)
                
                ####### compute V matrix #####
                if counter==0:
                    V=Variable(mean_grad_U.data*mean_grad_U.data)
                else:
                    V=exp_term*V+(1-exp_term)*Variable(mean_grad_U.data*mean_grad_U.data)# num_chain x dim
                ####### Compute G term #######
                G=1./(lamb+torch.sqrt(V)) #### num_chain x dim
                #Tracer()()
                
                noise=Variable(torch.randn(num_chain,self.dim))
                #add_noise=torch.unsqueeze(torch.diag(torch.sqrt(eps*2*self.C-(eps**2)*self.B)),dim=1)*noise#num_chain x total_dim
                #state_mom=Variable(0*torch.randn(num_chain,self.dim),requires_grad=True)
                add_noise=torch.sqrt(2*eps*G)*noise
                state_pos=Variable(state_pos.data-eps*G.data*grad_U.data+add_noise.data)
                if (counter+1)%200==0:
                    pass
                    #Tracer()()
                if (counter+1)%interval==0:
                    state_list.append(state_pos)
                counter+=1
            ed=time.time()-st
            if (time_t+1)%10==0:
                print(Test_accuracy(test_loader,self.U,state_pos,data_number=data_len))
            time_list.append(ed)
                
            #state_list.append(state_pos)
        return state_list,energy_list,time_list
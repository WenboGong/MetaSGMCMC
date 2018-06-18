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
from BNN_Q_D import *
from BNN_Model_def import *
from BNN_Sampler import *
from BNN_training_func import *
from BNN_Dataloader import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
total_run=10
random_seed=[15,25,35,40,45,50,55,60,65,70]



train_loader = datasets.MNIST('./BNN_MNIST/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_loader = datasets.MNIST('./BNN_MNIST/data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_X,train_Y,test_X,test_Y=SelectImage_All(train_loader,test_loader)
train_class=NewMNISTLoader(train_X,train_Y,flag_train=True)
test_class=NewMNISTLoader(test_X,test_Y,flag_train=False)

train_loader=DataLoader(train_class, batch_size=500,
                        shuffle=True)
test_loader=DataLoader(test_class,batch_size=500,shuffle=True)



# #############################################################################################################################
##### Set which sampler to run #######
flag_nnsghmc=True
flag_SGLD=False


if flag_nnsghmc==True:
    print('NNSGHMC')
elif flag_SGLD==True:
    print('SGLD')
else:
    print('SGHMC')
if flag_nnsghmc==False:
    

    for total_idx in range(total_run):
        print('Current total:%s'%(total_idx+1))
        torch.manual_seed(random_seed[total_idx])

        num_chain=20
        MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=10)
        total_dim=MLP_mnist.get_total_dim()
        data_N=60000.
        C=50
        B=0.

        SGHMC_obj=SGHMC(total_dim,MLP_mnist)
        A=Variable(torch.Tensor([1.1]))

        weight_init=Variable(0.01*torch.randn(num_chain,total_dim),requires_grad=True)
        if flag_SGLD==True:
            eps=0.2/data_N
        else:
            eps=0.01/data_N # this is the learning rate, in original paper this = eps_true**2=gamma/data_N (gamma is the per sampler lr)
        if flag_SGLD==False:
            alpha=0.01 # this is eps_true*C 
        else:
            alpha=1.
            
        total_step=100
        sigma=1.

        state_mom_init=Variable(0*torch.randn(num_chain,total_dim),requires_grad=True)
        st=time.time()
        if flag_SGLD==True:
            state_list_SGLD,_,energy_list,time_list_SGLD=SGHMC_obj.parallel_sample(weight_init,state_mom_init,train_loader,data_N,num_chain=num_chain,eps=eps,alpha=alpha,beta=0,sigma=sigma,interval=100,mom_resample=100000,total_step=total_step,flag_SGLD=True,test_loader=test_loader)
        else:
            state_list_sghmc,_,energy_list,time_list_sghmc=SGHMC_obj.parallel_sample(weight_init,state_mom_init,train_loader,data_N,num_chain=num_chain,eps=eps,alpha=alpha,beta=0,sigma=sigma,interval=100,mom_resample=100000,total_step=total_step,flag_SGLD=False,test_loader=test_loader)
        ed=time.time()-st
        # Store the samples
        if flag_SGLD==True:
            tensor_state_list_sgld=torch.stack(tuple(state_list_SGLD),dim=0)
            torch.save(tensor_state_list_sgld,'./ReLU_Generalization_Long_Run/Test_long_run_sgld_%s_0.2'%(total_idx+1))

            draw_list_SGLD=draw_samples(state_list_SGLD,burn_in=0,interval=1,draw_method='Cross')
            Acc,NLL=Test_accuracy(test_loader,MLP_mnist,draw_list_SGLD[-1],data_number=10000.)
            Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()],[ed]))
            np.savetxt('./ReLU_Generalization_Long_Run/Test_long_run_sgld_%s_0.2'%(total_idx+1),Result)
        else:
            tensor_state_list_sghmc=torch.stack(tuple(state_list_sghmc),dim=0)
            torch.save(tensor_state_list_sghmc,'./ReLU_Generalization_Long_Run/Test_long_run_sghmc_%s'%(total_idx+1))

            draw_list_sghmc=draw_samples(state_list_sghmc,burn_in=0,interval=1,draw_method='Cross')
            Acc,NLL=Test_accuracy(test_loader,MLP_mnist,draw_list_sghmc[-1],data_number=10000.)
            Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()],[ed]))
            np.savetxt('./ReLU_Generalization_Long_Run/Test_long_run_sghmc_%s_result'%(total_idx+1),Result)
else:
    for total_idx in range(total_run):
        
        print('Current total:%s'%(total_idx+1))
    
        torch.manual_seed(random_seed[total_idx])
        
        num_chain=20
        MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=10)
        Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
        D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
        
        
        # load the trained sampler 
        Q_MLP.load_state_dict(torch.load('./Q_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2'))
        D_MLP.load_state_dict(torch.load('./D_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2'))
        
        total_dim=MLP_mnist.get_total_dim()
        data_N=60000.
        #Tracer()()
        B=Variable(torch.Tensor([0]))
        Q=parallel_Q_eff(total_dim,Q_MLP,MLP_mnist,num_chain,clamp=5,dim_pen=1.,dim_pen_p=1,sqrt=False)
        D=parallel_D_eff(total_dim,D_MLP,MLP_mnist,num_chain,clamp_min=0,clamp_max=1000,dim_pen=1.,dim_pen_p=1,dim_pen_g=1,sqrt=False)
        Gamma=parallel_Gamma_eff(total_dim,Q_NN=Q_MLP,D_NN=D_MLP)


        NNSGHMC_obj=NN_SGHMC(total_dim,MLP_mnist,D,Q,Gamma)


        eps=float(np.sqrt(0.0085/data_N))
        eps2=float(np.sqrt(0.018/data_N))
        sigma=1.
        const_Q=0.
        const_D=float(0.01/eps)

        coef=15910./total_dim

        total_step=100


        weight_init=Variable(0.01*torch.randn(num_chain,total_dim),requires_grad=True)

        state_mom_init=Variable(0*torch.randn(num_chain,total_dim),requires_grad=True)


        st=time.time()
        
        ##### This is to run meta without finite difference ######
#         state_list_nnsghmc,state_mom_list,energy_list,_,_,A_list,time_list_nnsghmc=NNSGHMC_obj.parallel_sample(weight_init,
#                                                                                                               state_mom_init,B,train_loader,coef=coef,num_chain=num_chain,data_N=data_N,sigma=sigma,
#                                                                                                               total_step=total_step,limit_step=100,eps=eps,eps2=eps2,TBPTT_step=10,sample_interval=100,mom_resample=2000000,mom_scale=1.,mode_train=False,const_Q=const_Q,const_D=const_D,flag_finite=False,test_loader=test_loader)
################ Finite difference 
        state_list_nnsghmc,time_list_nnsghmc=NNSGHMC_obj.parallel_sample_FD(weight_init,state_mom_init,B,train_loader,data_N=data_N,sigma=sigma,num_chain=num_chain,total_step=total_step,eps=eps,eps2=eps2,coef=coef,sample_interval=100,const_Q=const_Q,const_D=const_D,test_loader=test_loader)
        ed=time.time()-st
        
        tensor_state_list_nnsghmc=torch.stack(tuple(state_list_nnsghmc),dim=0)
        torch.save(tensor_state_list_nnsghmc,'./ReLU_Generalization_Long_Run/Test_long_run_nnsghmc_%s_fd_0.18'%(total_idx+1))

        draw_list_nnsghmc=draw_samples(state_list_nnsghmc,burn_in=0,interval=1,draw_method='Cross')
        Acc,NLL=Test_accuracy(test_loader,MLP_mnist,draw_list_nnsghmc[-1],data_number=10000.)
        Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()],[ed]))
        np.savetxt('./ReLU_Generalization_Long_Run/Test_long_run_nnsghmc_%s_result_fd_0.018'%(total_idx+1),Result)
        

        

        
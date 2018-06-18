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


'''
Loader Definition

'''

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./BNN_MNIST/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./BNN_MNIST/data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=500, shuffle=True)

Init_MNIST_train=datasets.MNIST('./BNN_MNIST/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
Init_MNIST_test=datasets.MNIST('./BNN_MNIST/data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
X_train_sampler_tensor,X_train_BNN_tensor,Y_train_sampler_tensor,Y_train_BNN_tensor,X_test_BNN_tensor,Y_test_BNN_tensor,X_test_sampler_tensor,Y_test_sampler_tensor=SelectImage(Init_MNIST_train,Init_MNIST_test,train_image=[0,4],test_image=[5,9])

MNIST_train_sampler=GroupMNIST(X_train_sampler_tensor,Y_train_sampler_tensor,group=1)
MNIST_train_sampler_loader = DataLoader(MNIST_train_sampler, batch_size=500,
                        shuffle=True)
MNIST_test_sampler=GroupMNIST(X_test_sampler_tensor,Y_test_sampler_tensor,group=4)
MNIST_test_sampler_loader = DataLoader(MNIST_test_sampler, batch_size=500,
                        shuffle=True)

MNIST_train_BNN=GroupMNIST(X_train_BNN_tensor,Y_train_BNN_tensor,group=2)
MNIST_train_BNN_loader = DataLoader(MNIST_train_BNN, batch_size=500,
                        shuffle=True)

MNIST_test_BNN=GroupMNIST(X_test_BNN_tensor,Y_test_BNN_tensor,group=3)
MNIST_test_BNN_loader = DataLoader(MNIST_test_BNN, batch_size=500,
                        shuffle=True)

'''
Sampler 

'''


flag_nnsghmc=True
flag_SGLD=True
flag_PSGLD=False
if flag_nnsghmc==True:
    print('NNSGHMC')
elif flag_SGLD==True:
    print('SGLD')
elif flag_SGLD==False:
    print('SGHMC')
elif flag_PSGLD==True:
    print('PSGLD')
    
if flag_nnsghmc==False and flag_PSGLD==False:
    

    for total_idx in range(total_run):
        print('Current total:%s'%(total_idx+1))
        torch.manual_seed(random_seed[total_idx])

        num_chain=20
        MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=5)
        total_dim=MLP_mnist.get_total_dim()
        data_N=float(len(MNIST_train_BNN))
        C=50
        B=0.
        A=Variable(torch.Tensor([1]).cuda())
        SGHMC_obj=SGHMC(total_dim,MLP_mnist)


        weight_init=Variable(0.01*torch.randn(num_chain,total_dim),requires_grad=True)

        if flag_SGLD==True:
            eps=0.2/data_N
        else:
            eps=0.01/data_N # this is the learning rate, in original paper this = eps_true**2=gamma/data_N (gamma is the per sampler lr)
        if flag_SGLD==False:
            alpha=0.01 # this is eps_true*C 
        else:
            alpha=1
            
        total_step=100
        sigma=1.

        state_mom_init=Variable(0*torch.randn(num_chain,total_dim),requires_grad=True)
        st=time.time()
        if flag_SGLD==True:
            state_list_SGLD,_,energy_list,time_list_SGLD=SGHMC_obj.parallel_sample(weight_init,state_mom_init,MNIST_train_BNN_loader,data_N,num_chain=num_chain,eps=eps,alpha=alpha,beta=0,sigma=sigma,interval=20,mom_resample=100000,total_step=total_step,flag_SGLD=True,test_loader=MNIST_test_BNN_loader,data_len=float(len(MNIST_test_BNN)))
        else:
            state_list_sghmc,_,energy_list,time_list_sghmc=SGHMC_obj.parallel_sample(weight_init,state_mom_init,MNIST_train_BNN_loader,data_N,num_chain=num_chain,eps=eps,alpha=alpha,beta=0,sigma=sigma,interval=20,mom_resample=100000,total_step=total_step,flag_SGLD=False,test_loader=MNIST_test_BNN_loader,data_len=float(len(MNIST_test_BNN)))
        ed=time.time()-st

        if flag_SGLD==True:
            tensor_state_list_sgld=torch.stack(tuple(state_list_SGLD),dim=0)
            torch.save(tensor_state_list_sgld,'./Dataset_Gen_Long_Run/Test_long_run_sgld_%s'%(total_idx+1))

            draw_list_SGLD=draw_samples(state_list_SGLD,burn_in=0,interval=1,draw_method='Cross')
            Acc,NLL=Test_accuracy(MNIST_test_BNN_loader,MLP_mnist,draw_list_SGLD[-1],data_number=float(len(MNIST_test_BNN)))
            Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()],[ed]))
            np.savetxt('./Dataset_Gen_Long_Run/Test_long_run_sgld_%s_result'%(total_idx+1),Result)
        else:
            tensor_state_list_sghmc=torch.stack(tuple(state_list_sghmc),dim=0)
            torch.save(tensor_state_list_sghmc,'./Dataset_Gen_Long_Run/Test_long_run_sghmc_%s'%(total_idx+1))

            draw_list_sghmc=draw_samples(state_list_sghmc,burn_in=0,interval=1,draw_method='Cross')
            Acc,NLL=Test_accuracy(MNIST_test_BNN_loader,MLP_mnist,draw_list_sghmc[-1],data_number=float(len(MNIST_test_BNN)))
            Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()],[ed]))
            np.savetxt('./Dataset_Gen_Long_Run/Test_long_run_sghmc_%s_result'%(total_idx+1),Result)
elif flag_nnsghmc==True and flag_PSGLD==False:
    for total_idx in range(total_run):
        
        print('Current total:%s'%(total_idx+1))
    
        torch.manual_seed(random_seed[total_idx])
        
        num_chain=20
        MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=5)
        Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
        D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
        #Q_MLP.load_state_dict(torch.load('./Q_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2_datasetGen'))
        #D_MLP.load_state_dict(torch.load('./D_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2_datasetGen'))
        Q_MLP.load_state_dict(torch.load('./tmp_model_save/Q_state_batch_500_baseline_50D_70G_step_0.007_100ep_broad_0.2_ep40'))
        D_MLP.load_state_dict(torch.load('./tmp_model_save/D_state_batch_500_baseline_50D_70G_step_0.007_100ep_broad_0.2_ep40'))
        total_dim=MLP_mnist.get_total_dim()
        data_N=float(len(MNIST_train_BNN))
        #Tracer()()
        B=Variable(torch.Tensor([0]))
        Q=parallel_Q_eff(total_dim,Q_MLP,MLP_mnist,num_chain,clamp=4,dim_pen=1.,dim_pen_p=1,sqrt=False)
        D=parallel_D_eff(total_dim,D_MLP,MLP_mnist,num_chain,clamp_min=0.,clamp_max=400,dim_pen=1.,dim_pen_p=1,dim_pen_g=1,sqrt=False)
        Gamma=parallel_Gamma_eff(total_dim,Q_NN=Q_MLP,D_NN=D_MLP)


        NNSGHMC_obj=NN_SGHMC(total_dim,MLP_mnist,D,Q,Gamma)


        eps=float(np.sqrt(0.008/data_N))
        eps2=float(np.sqrt(0.018/data_N))
        sigma=1.
        const_Q=0.
        const_D=float(0.01/eps)

        coef=15905./total_dim

        total_step=100


        weight_init=Variable(0.01*torch.randn(num_chain,total_dim),requires_grad=True)

        state_mom_init=Variable(0*torch.randn(num_chain,total_dim),requires_grad=True)


        st=time.time()
        #state_list_nnsghmc,state_mom_list,energy_list,_,_,A_list,time_list_nnsghmc=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,train_loader,coef=coef,num_chain=num_chain,data_N=data_N,sigma=sigma,total_step=total_step,limit_step=100,eps=eps,eps2=eps2,TBPTT_step=10,sample_interval=100,mom_resample=2000000,mom_scale=1.,mode_train=False,const_Q=const_Q,const_D=const_D,flag_finite=False,test_loader=MNIST_teset_BNN_loader,data_len=float(len(MNIST_test_BNN)))
        state_list_nnsghmc,time_list_nnsghmc=NNSGHMC_obj.parallel_sample_FD(weight_init,state_mom_init,B,MNIST_train_BNN_loader,data_N=data_N,sigma=sigma,num_chain=num_chain,total_step=total_step,eps=eps,eps2=eps2,coef=coef,sample_interval=20,const_Q=const_Q,const_D=const_D,test_loader=MNIST_test_BNN_loader,data_len=float(len(MNIST_test_BNN)))
        ed=time.time()-st
        
        tensor_state_list_nnsghmc=torch.stack(tuple(state_list_nnsghmc),dim=0)
        torch.save(tensor_state_list_nnsghmc,'./Dataset_Gen_Long_Run/Test_long_run_nnsghmc_%s_fd_0.18'%(total_idx+1))

        draw_list_nnsghmc=draw_samples(state_list_nnsghmc,burn_in=0,interval=1,draw_method='Cross')
        Acc,NLL=Test_accuracy(MNIST_test_BNN_loader,MLP_mnist,draw_list_nnsghmc[-1],data_number=float(len(MNIST_test_BNN)))
        Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()],[ed]))
        np.savetxt('./Dataset_Gen_Long_Run/Test_long_run_nnsghmc_result_%s_fd_0.18'%(total_idx+1),Result)
        
######################################### PSGLD ###########################################
elif flag_PSGLD==True:
    for ind in range(total_run):
        print('Current total:%s'%(ind+1))
        torch.manual_seed(random_seed[ind])
        num_chain=20
        MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=10)
        total_dim=MLP_mnist.get_total_dim()
        data_N=float(len(MNIST_train_BNN))
        A=Variable(torch.Tensor([0]).cuda())
        PSGLD_obj=PSGLD(total_dim,MLP_mnist)


        weight_init=Variable(0.01*torch.randn(num_chain,total_dim),requires_grad=True)

        eps=1.3e-3/data_N#2.5e-4 # this is the learning rate, in original paper this = eps_true**2=gamma/data_N (gamma is the per sampler lr)
        exp_term=0.99 # this is exponential decay term
        lamb=1e-5

        total_step=100
        sigma=1.




        state_list_PSGLD,energy_list,time_list_PSGLD=PSGLD_obj.parallel_sample(weight_init,MNIST_train_BNN_loader,
                                                                               data_N,num_chain=num_chain,total_step=total_step,eps=eps,exp_term=exp_term,lamb=lamb,
                                                                               sigma=sigma,interval=20,test_loader=MNIST_test_BNN_loader,data_len=float(len(MNIST_test_BNN)))

        tensor_state_list_psgld=torch.stack(tuple(state_list_PSGLD),dim=0)
        torch.save(tensor_state_list_psgld,'./Dataset_Gen_Long_Run/Test_long_run_psgld_%s_step_1.3'%(ind+1))

        draw_list_PSGLD=draw_samples(state_list_PSGLD,burn_in=0,interval=1,draw_method='Cross')
        Acc,NLL=Test_accuracy(MNIST_test_BNN_loader,MLP_mnist,draw_list_PSGLD[-1],data_number=10000.)
        #Result=np.concatenate(([Acc.cpu().data.numpy()],NLL.cpu().data.numpy()))
        #np.savetxt('./Dataset_Gen_Long_Run/long_run_psgld_%s_result_step_1.3'%(ind+1),Result)






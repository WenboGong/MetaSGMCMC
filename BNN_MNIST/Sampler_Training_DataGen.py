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
from torch.utils.data import Dataset, DataLoader
from BNN_Util import *
from BNN_Q_D import *
from BNN_Model_def import *
from BNN_Sampler import *
from BNN_training_func import *
from BNN_Dataloader import *




torch.set_default_tensor_type('torch.cuda.FloatTensor')

############# Loader Definition #############

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

#####################################################

print('Training')

torch.manual_seed(7)
num_chain=20
MLP_mnist=BNN(dim=784,hidden=20,layer_num=2,dim_out=5,act_func='ReLU')
Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
total_dim=MLP_mnist.get_total_dim()
data_N=float(len(MNIST_train_sampler))

B=Variable(torch.Tensor([0]))
Q=parallel_Q_eff(total_dim,Q_MLP,MLP_mnist,num_chain,clamp=5,dim_pen=1.,dim_pen_p=1.,sqrt=False)
D=parallel_D_eff(total_dim,D_MLP,MLP_mnist,num_chain,clamp_min=0.,clamp_max=400,dim_pen=1.,dim_pen_p=1.,dim_pen_g=1.,sqrt=False)
Gamma=parallel_Gamma_eff(total_dim,Q_NN=Q_MLP,D_NN=D_MLP)

Adam_Q=torch.optim.Adam(list(Q_MLP.parameters()),lr=0.003,betas=(0.5,0.9))
Adam_D=torch.optim.Adam(list(D_MLP.parameters()),lr=0.003,betas=(0.5,0.9))
NNSGHMC_obj=NN_SGHMC(total_dim,MLP_mnist,D,Q,Gamma)
epoch=40
ep_second=7
eps=float(np.sqrt(0.007/data_N))
eps2=float(np.sqrt(0.007/data_N))
sigma=1.
const_Q=0.
const_D=float(0.01/eps)
print(eps)
for ep in range(epoch):
    print('Initial Training:%s'%(ep+1))
    Adam_Q.zero_grad()
    Adam_D.zero_grad()
    weight_init=Variable(0.2*torch.randn(num_chain,total_dim),requires_grad=True)

    state_mom_init=Variable(0*torch.randn(num_chain,total_dim),requires_grad=True)

    state_list,state_mom_list,energy_list,_,_,A_list,_=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,MNIST_train_sampler_loader,data_N=data_N,sigma=sigma,num_chain=num_chain,total_step=2,limit_step=100,eps=eps,eps2=eps2,TBPTT_step=10,sample_interval=5,mom_resample=2000000,mom_scale=1.,mode_train=True,const_Q=const_Q,const_D=const_D,test_loader=MNIST_test_sampler_loader,data_len=float(len(MNIST_test_sampler)))
    
    modify_grad(A_list,Q_MLP,D_MLP)
    Err,NLL=Test_accuracy(MNIST_test_sampler_loader,MLP_mnist,state_list[-1],data_number=float(len(MNIST_test_sampler)))
    print('Test Err:%s Test NLL:%s'%(Err.cpu().data.numpy(),NLL.cpu().data.numpy()))
    Adam_Q.step()
    Adam_D.step()
    for ep_se in range(ep_second):
        print('          Continuous Training:%s'%(ep_se+1))
        Adam_Q.zero_grad()
        Adam_D.zero_grad()
        weight_init=Variable(state_list[-1].data,requires_grad=True)

        state_mom_init=Variable(1*state_mom_list[-1].data,requires_grad=True)

        state_list,_,energy_list,_,_,A_list,_=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,MNIST_train_sampler_loader,num_chain=num_chain,data_N=data_N,sigma=sigma,total_step=2,limit_step=100,eps=eps,eps2=eps2,TBPTT_step=10,sample_interval=5,mom_resample=2000000,mom_scale=1.,mode_train=True,const_Q=const_Q,const_D=const_D,test_loader=MNIST_test_sampler_loader,data_len=float(len(MNIST_test_sampler)))
    
        modify_grad(A_list,Q_MLP,D_MLP)
        Err,NLL=Test_accuracy(MNIST_test_sampler_loader,MLP_mnist,state_list[-1],data_number=float(len(MNIST_test_sampler)))
        print('Test Err:%s Test NLL:%s'%(Err.cpu().data.numpy(),NLL.cpu().data.numpy()))
        Adam_Q.step()
        Adam_D.step()
    if (ep+1)%10==0:
        torch.save(Q_MLP.state_dict(), './tmp_model_save/Q_state_batch_500_baseline_50D_70G_step_0.007_100ep_broad_0.2_ep%s'%(ep+1))
        torch.save(D_MLP.state_dict(), './tmp_model_save/D_state_batch_500_baseline_50D_70G_step_0.007_100ep_broad_0.2_ep%s'%(ep+1))
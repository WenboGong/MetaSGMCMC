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



print('Training')

torch.manual_seed(15)
num_chain=20
MLP_mnist=BNN(dim=784,hidden=20,layer_num=2,dim_out=10,act_func='ReLU')
Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)
total_dim=MLP_mnist.get_total_dim()
data_N=60000.

B=Variable(torch.Tensor([0]))
Q=parallel_Q_eff(total_dim,Q_MLP,MLP_mnist,num_chain,clamp=5,dim_pen=1.,dim_pen_p=1.,sqrt=False)
D=parallel_D_eff(total_dim,D_MLP,MLP_mnist,num_chain,clamp_min=0.,clamp_max=1000,dim_pen=1.,dim_pen_p=1.,dim_pen_g=1.,sqrt=False)
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

for ep in range(epoch):
    print('Initial Training:%s'%(ep+1))
    Adam_Q.zero_grad()
    Adam_D.zero_grad()
    weight_init=Variable(0.2*torch.randn(num_chain,total_dim),requires_grad=True)

    state_mom_init=Variable(0*torch.randn(num_chain,total_dim),requires_grad=True)

    state_list,state_mom_list,energy_list,_,_,A_list,_=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,train_loader,data_N=data_N,sigma=sigma,num_chain=num_chain,total_step=2,limit_step=100,eps=eps,eps2=eps2,TBPTT_step=15,sample_interval=3,mom_resample=2000000,mom_scale=1.,mode_train=True,const_Q=const_Q,const_D=const_D,test_loader=test_loader)
    
    modify_grad(A_list,Q_MLP,D_MLP)
    Err,NLL=Test_accuracy(test_loader,MLP_mnist,state_list[-1],data_number=10000.)
    print('Test Err:%s Test NLL:%s'%(Err.cpu().data.numpy(),NLL.cpu().data.numpy()))
    Adam_Q.step()
    Adam_D.step()
    for ep_se in range(ep_second):
        print('          Continuous Training:%s'%(ep_se+1))
        Adam_Q.zero_grad()
        Adam_D.zero_grad()
        weight_init=Variable(state_list[-1].data,requires_grad=True)

        state_mom_init=Variable(1*state_mom_list[-1].data,requires_grad=True)

        state_list,_,energy_list,_,_,A_list,_=NNSGHMC_obj.parallel_sample(weight_init,state_mom_init,B,train_loader,num_chain=num_chain,data_N=data_N,sigma=sigma,total_step=2,limit_step=100,eps=eps,eps2=eps2,TBPTT_step=15,sample_interval=3,mom_resample=2000000,mom_scale=1.,mode_train=True,const_Q=const_Q,const_D=const_D,test_loader=test_loader)
    
        modify_grad(A_list,Q_MLP,D_MLP)
        Err,NLL=Test_accuracy(test_loader,MLP_mnist,state_list[-1],data_number=10000.)
        print('Test Err:%s Test NLL:%s'%(Err.cpu().data.numpy(),NLL.cpu().data.numpy()))
        Adam_Q.step()
        Adam_D.step()
    if (ep+1)%10==0:
        torch.save(Q_MLP.state_dict(), './tmp_model_save/Q_state_batch_500_baseline_50D_70G_step_0.007_100ep_broad_0.2_ep%s_Adam_0.0015'%(ep+1))
        torch.save(D_MLP.state_dict(), './tmp_model_save/D_state_batch_500_baseline_50D_70G_step_0.007_100ep_broad_0.2_ep%s_Adam_0.0015'%(ep+1))

# torch.save(Q_MLP.state_dict(), './Q_state_batch_500_baseline_10D_70G_step_0.01_40ep_broad_0.2_70epoch')
# torch.save(D_MLP.state_dict(), './D_state_batch_500_baseline_10D_70G_step_0.01_40ep_broad_0.2_70epoch')

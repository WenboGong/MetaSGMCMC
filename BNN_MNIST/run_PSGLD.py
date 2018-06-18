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


'''
If run Sigmoid generalization, change folder path, act_func and eps
'''


for ind in range(total_run):
    print('Current total:%s'%(ind+1))
    torch.manual_seed(random_seed[ind])
    num_chain=20
    MLP_mnist=BNN(dim=784,hidden=40,layer_num=3,dim_out=10,act_func='ReLU')
    total_dim=MLP_mnist.get_total_dim()
    data_N=60000.
    A=Variable(torch.Tensor([0]).cuda())
    PSGLD_obj=PSGLD(total_dim,MLP_mnist)


    weight_init=Variable(0.01*torch.randn(num_chain,total_dim),requires_grad=True)

    eps=1.4e-3/data_N#2.5e-4 # this is the learning rate, in original paper this = eps_true**2=gamma/data_N (gamma is the per sampler lr)
    exp_term=0.99 # this is exponential decay term
    lamb=1e-5

    total_step=100
    sigma=1.


    

    state_list_PSGLD,energy_list,time_list_PSGLD=PSGLD_obj.parallel_sample(weight_init,train_loader,
                                                                           data_N,num_chain=num_chain,total_step=total_step,eps=eps,exp_term=exp_term,lamb=lamb,
                                                                           sigma=sigma,interval=100,test_loader=test_loader,data_len=float(len(test_class)))
    
    tensor_state_list_psgld=torch.stack(tuple(state_list_PSGLD),dim=0)
    torch.save(tensor_state_list_psgld,'./ReLU_Generalization_Long_Run/Test_long_run_psgld_%s_step_1.4'%(ind+1))

    draw_list_PSGLD=draw_samples(state_list_PSGLD,burn_in=0,interval=1,draw_method='Cross')
    Acc,NLL=Test_accuracy(test_loader,MLP_mnist,draw_list_PSGLD[-1],data_number=10000.)
    Result=np.concatenate(([Acc],[NLL.cpu().data.numpy()]))
    np.savetxt('./ReLU_Generalization_Long_Run/Test_long_run_psgld_%s_result_step_1.4'%(ind+1),Result)

    

    

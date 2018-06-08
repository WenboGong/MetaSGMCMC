import os



import argparse
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
import torch.backends.cudnn as cudnn
''' Import Essential functions'''
from NNSGHMC_toy import *
from SGHMC_toy import *
from Stein import *
from Training_func import *
from Util import *



'''
This is to train the Meta Sampler on Gaussian with randomly generated diagonal covariance matrix. The default dimensionality is 10. It will use GPU as default. If the GPU memory is not sufficient, reduce the chain number, TBPTT or total steps.
'''








cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Training NNSGHMC using 2D strong correlated Gaussian')

###### GPU settings ########
parser.add_argument('--gpu_id', default='3', type=str)


###### Training Parameters ##################

parser.add_argument('--init_pos_range', default=6, type=float)
parser.add_argument('--init_mom_range', default=1., type=float)

parser.add_argument('--noise_injection', default=1., type=float)
parser.add_argument('--noise_estimate', default=0., type=float)
parser.add_argument('--total_step', default=100, type=int)
parser.add_argument('--num_chain', default=50, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--epoch_second', default=3, type=int)# Sub epoches 
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--step_size', default=0.01, type=float)
parser.add_argument('--TBPTT', default=20, type=int) # Stop Gradient steps
parser.add_argument('--burn_in_cross', default=0, type=int) # Cross chain training burn-in
parser.add_argument('--interval_cross', default=2, type=int) # Cross chain training thinning
parser.add_argument('--burn_in_sub', default=50, type=int) # In chain burn-in
parser.add_argument('--interval_sub', default=3, type=int) # In chain thinning
parser.add_argument('--momentum_resample_step', default=100000000, type=int) # Steps before momentum resample
parser.add_argument('--momentum_resample_scale', default=1., type=float) # Std of sampled momentum
parser.add_argument('--clamp',default=5.,type=float) # Clamp of Q output
parser.add_argument('--dim',default=10,type=int) # training density dimensions
parser.add_argument('--cov_range',default=1.,type=float) # Controlled the random generated cov for training density




###### Stein Estimator Parameters ###########
#parser.add_argument('--coef_med', default=0.5, type=float)




opt = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id

###### Set to automatically use cuda ######
torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''

Training Script

'''

######## Parse the argument #######
timestr = time.strftime("%Y%m%d-%H%M")
################## Cross Chain Training (should result in fast burn-in but may be high bias )#######################
dim=opt.dim
#### Energy def #####
mu=Variable(3.*torch.ones(1,dim))

# cov=np.loadtxt('./saveModel/Cov_20180413-1226')
# cov=torch.from_numpy(cov).float()

var=generate_Cov_diag(dim,scale=opt.cov_range)
var=Variable(var.cuda())

np.savetxt('./saveModel/Cov_%s'%(timestr),var.data.cpu().numpy())
U=U_Gaussian(mu,var)

#### NNSGHMC def ######
num_chain=opt.num_chain
#### Pre conditioning ####
Q_MLP=MLP(input_dim=2,hidden=40)
D_MLP=Positive_MLP(input_dim=3,hidden=40)
Q=parallel_Q_eff(dim=dim,Q_MLP=Q_MLP,U=U,num_chain=num_chain,clamp=opt.clamp)
D=parallel_D_eff(dim=dim,D_MLP=D_MLP,U=U,num_chain=num_chain,clamp=opt.clamp)
Gamma=parallel_Gamma_eff(dim,Q,D_NN=D)

B=Variable(torch.diag(opt.noise_estimate*torch.ones(dim)))# Noise estimation matrix
NN_SGHMC_obj=NN_SGHMC(dim,U,D,Q,Gamma)
##### Hyper parameter settings #####
epoch=opt.epoch
second_epoch=opt.epoch_second
Adam=torch.optim.Adam(list(Q_MLP.parameters())+list(D_MLP.parameters()),lr=opt.lr,betas=(0.5,0.99))
eps=opt.step_size
total_step=opt.total_step

######## Training ##########
for ep in range(epoch):
    ###### Model Save ###
    if (ep+1)%10==0:
        torch.save(Q_MLP.state_dict(), './saveModel/Q_state_%s_dim_%s_step_%s_clamp_%s_range_%s_time_%s'%((ep+1),dim,opt.step_size,opt.clamp,opt.init_pos_range,timestr))
        torch.save(D_MLP.state_dict(),'./saveModel/D_state_%s_dim_%s_step_%s_clamp_%s_range_%s_time_%s'%((ep+1),dim,opt.step_size,opt.clamp,opt.init_pos_range,timestr))	
    print('Ep:%s'%ep)
    Adam.zero_grad()
    state_mom_init=Variable(opt.init_mom_range*torch.randn(dim,num_chain),requires_grad=True)
    state_pos_init=Variable(opt.init_pos_range*torch.rand(dim,num_chain),requires_grad=True)
    ################# Cross Chain Training #############
    state_list_nnsg,state_mom_nnsg=NN_SGHMC_obj.parallel_sample(state_pos_init,state_mom_init,B,num_chain=num_chain,TBPTT_step=opt.TBPTT,total_step=total_step,
flag_manual_noise=True,eps=eps,mom_resample=opt.momentum_resample_step,mom_scale=opt.momentum_resample_scale,inject_scale=opt.noise_injection)
    
    grad_log_p,grad_log_p_D,grad_Qparam_log_q,grad_Dparam_log_q,mean_log_p=Chain_grad(state_list_nnsg,'Cross',Q_MLP,D_MLP,
U,end=total_step,burn_in=opt.burn_in_cross,interval=opt.interval_cross) #### Obtain gradient of the cross chain loss
    
    ##### Assign Gradient #########
    for index,p in zip(range(len(list(Q_MLP.parameters()))),Q_MLP.parameters()):
        if type(grad_Qparam_log_q[index])!=type(None) :
            #Tracer()()
            G=grad_Qparam_log_q[index]+1*grad_log_p[index]
            p.grad=G
    #### D Gradient assign ######## 
    for index,p in zip(range(len(list(D_MLP.parameters()))),D_MLP.parameters()):
        if type(grad_Dparam_log_q[index])!=type(None) :
            #Tracer()()
            G=grad_Dparam_log_q[index]+1*grad_log_p_D[index]
            p.grad=G
    ######Take Cross update step ########
    Adam.step()
    ###### Output Accumu energy #########
    print('     Cross Accumu log:%s'%(mean_log_p.cpu().data.numpy()))
    ############## In Chain Training ###########
    Adam.zero_grad()
    print('            In Chain Training')
    #state_list_nnsg,state_mom_nnsg=NN_SGHMC_obj.parallel_sample(state_pos_init,state_mom_init,B,num_chain=num_chain,TBPTT_step=10,total_step=total_step_in,flag_manual_noise=True,eps=eps,mom_resample=10000,mom_scale=1.)
    
    grad_log_p,grad_log_p_D,grad_Qparam_log_q,grad_Dparam_log_q,mean_log_p=All_Chain_grad(state_list_nnsg,'Within',Q_MLP,D_MLP,
U,end=total_step,burn_in=opt.burn_in_sub,interval=opt.interval_sub) ### Obtain In chain loss gradient
    
    ##### Assign Gradient #########
    for index,p in zip(range(len(list(Q_MLP.parameters()))),Q_MLP.parameters()):
        if type(grad_Qparam_log_q[index])!=type(None) :
            #Tracer()()
            G=grad_Qparam_log_q[index]+1*grad_log_p[index]
            p.grad=G
    #### D assign ###### 
    for index,p in zip(range(len(list(D_MLP.parameters()))),D_MLP.parameters()):
        if type(grad_Dparam_log_q[index])!=type(None) :
            #Tracer()()
            G=grad_Dparam_log_q[index]+1*grad_log_p_D[index]
            p.grad=G

    Adam.step()
    print('     Sub Accumu log:%s'%(mean_log_p.cpu().data.numpy()))
    
    
    
    
    
    
    ######### Sub epoch Training #########
    for ep_second in range(second_epoch):
        print('   EP_second:%s'%(ep_second))
        Adam.zero_grad()
        state_mom_init=Variable(state_mom_nnsg[-1].data.clone(),requires_grad=True)
        state_pos_init=Variable(state_list_nnsg[-1].data.clone(),requires_grad=True)
        state_list_nnsg,state_mom_nnsg=NN_SGHMC_obj.parallel_sample(state_pos_init,state_mom_init,B,num_chain=num_chain,TBPTT_step=opt.TBPTT,total_step=total_step,
flag_manual_noise=True,eps=eps,mom_resample=opt.momentum_resample_step,mom_scale=opt.momentum_resample_scale,inject_scale=opt.noise_injection)
        grad_log_p,grad_log_p_D,grad_Qparam_log_q,grad_Dparam_log_q,mean_log_p=Chain_grad(state_list_nnsg,'Cross',Q_MLP,D_MLP,
U,end=total_step,burn_in=opt.burn_in_cross,interval=opt.interval_cross)# Cross Chain loss gradient
        ##### Assign Gradient #########
        for index,p in zip(range(len(list(Q_MLP.parameters()))),Q_MLP.parameters()):
            if type(grad_Qparam_log_q[index])!=type(None) :
                #Tracer()()
                G=grad_Qparam_log_q[index]+1*grad_log_p[index]
                p.grad=G
        #### D assign 
        for index,p in zip(range(len(list(D_MLP.parameters()))),D_MLP.parameters()):
            if type(grad_Dparam_log_q[index])!=type(None) :
                #Tracer()()
                G=grad_Dparam_log_q[index]+1*grad_log_p_D[index]
                p.grad=G
        Adam.step()
        print('     Cross Accumu log:%s'%(mean_log_p.cpu().data.numpy()))
        ############## Subsample Chain Training ##################
        
        Adam.zero_grad()
        #print('            Subsample Chain Training')
        #state_list_nnsg,state_mom_nnsg=NN_SGHMC_obj.parallel_sample(state_pos_init,state_mom_init,B,num_chain=num_chain,TBPTT_step=10,total_step=total_step_in,flag_manual_noise=True,eps=eps,mom_resample=10000,mom_scale=1.)

        grad_log_p,grad_log_p_D,grad_Qparam_log_q,grad_Dparam_log_q,mean_log_p=All_Chain_grad(state_list_nnsg,'Within',Q_MLP,D_MLP,
U,end=total_step,burn_in=opt.burn_in_sub,interval=opt.interval_sub) # In chain loss gradient 

        ##### Assign Gradient #########
        for index,p in zip(range(len(list(Q_MLP.parameters()))),Q_MLP.parameters()):
            if type(grad_Qparam_log_q[index])!=type(None) :
                #Tracer()()
                G=grad_Qparam_log_q[index]+1*grad_log_p[index]
                p.grad=G
        #### D assign 
        for index,p in zip(range(len(list(D_MLP.parameters()))),D_MLP.parameters()):
            if type(grad_Dparam_log_q[index])!=type(None) :
                #Tracer()()
                G=grad_Dparam_log_q[index]+1*grad_log_p_D[index]
                p.grad=G
        Adam.step()
        print('     Sub Accumu log:%s'%(mean_log_p.cpu().data.numpy()))











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
import matplotlib
matplotlib.use('agg')
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

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Test the trained Sampler')

###### GPU settings ########
parser.add_argument('--gpu_id', default='5', type=str)
###### Testing Parameters ##################
parser.add_argument('--init_pos_range', default=6, type=float)
parser.add_argument('--init_mom_range', default=0, type=float)
parser.add_argument('--noise_injection', default=1., type=float)
parser.add_argument('--noise_estimate', default=0., type=float)
parser.add_argument('--total_step', default=12000, type=int)
parser.add_argument('--num_chain', default=50, type=int)
parser.add_argument('--step_size_SGHMC', default=0.025, type=float)
parser.add_argument('--step_size_NNSGHMC', default=0.025, type=float)
parser.add_argument('--burn_in', default=0, type=int)
parser.add_argument('--interval', default=1, type=int)
parser.add_argument('--momentum_resample_step', default=100000000, type=int)
parser.add_argument('--momentum_resample_scale', default=1., type=float)
parser.add_argument('--clamp',default=5.,type=float)
parser.add_argument('--dim',default=20,type=int)
parser.add_argument('--cov_range',default=0.7,type=float)
parser.add_argument('--SGHMC_friction',default=1.,type=float)
parser.add_argument('--dim_pen',default=1.,type=float)
opt = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
timestr = time.strftime("%Y%m%d-%H%M")
###### Set to automatically use cuda ######
torch.set_default_tensor_type('torch.cuda.FloatTensor')



################# Define the Gaussian #######################
dim=opt.dim
mu=Variable(3*torch.unsqueeze(torch.ones(dim),dim=0))
var=np.loadtxt('./TestResult/Cov_rand_1_scale_0.6')
var=torch.from_numpy(var).float().cuda()
#var=generate_Cov(dim,scale=opt.cov_range)
#np.savetxt('./TestResult/Cov_%s_scale_%s'%(timestr,opt.cov_range),var.cpu().data.numpy())
U=U_Gaussian(mu,var)
num_chain=opt.num_chain
################# Draw samples from SGHMC ####################
print('Sampling from SGHMC')
num_chain=opt.num_chain
C=Variable(torch.diag(opt.SGHMC_friction*torch.ones(dim)))
B=Variable(torch.diag(opt.noise_estimate*torch.ones(dim)))
SGHMC_obj=SGHMC(dim,U,C,B)
inject_scale=opt.noise_injection
state_mom_init=Variable(opt.init_mom_range*torch.randn(dim,num_chain),requires_grad=True)
state_pos_init=Variable(opt.init_pos_range*torch.rand(dim,num_chain),requires_grad=True)
state_list_sg,state_mom_sg=SGHMC_obj.parallel_sample(state_pos_init,state_mom_init,num_chain=opt.num_chain,total_step=opt.total_step,flag_manual_noise=True,eps=opt.step_size_SGHMC,inject_scale=opt.noise_injection)


################ Draw samples from NNSGHMC ###################
print('Sampling from NNSGHMC')
Q_MLP=MLP(input_dim=2,hidden=40)
D_MLP=Positive_MLP(input_dim=3,hidden=40)
Q_MLP.load_state_dict(torch.load('./saveModel/Q_state_100_dim_10_clamp_5.0_range_6.0_time_20180413-1226'))
D_MLP.load_state_dict(torch.load('./saveModel/D_state_100_dim_10_clamp_5.0_range_6.0_time_20180413-1226'))
Q=parallel_Q_eff(dim=dim,Q_MLP=Q_MLP,U=U,num_chain=opt.num_chain,clamp=opt.clamp,dim_pen=opt.dim_pen)
D=parallel_D_eff(dim=dim,D_MLP=D_MLP,U=U,num_chain=opt.num_chain,clamp=opt.clamp,dim_pen=opt.dim_pen)
Gamma=parallel_Gamma_eff(dim,Q,D_NN=D)
NN_SGHMC_obj=NN_SGHMC(dim,U,D,Q,Gamma)

inject_scale=opt.noise_injection
B=Variable(torch.diag(opt.noise_estimate*torch.ones(dim)))
state_mom_init=Variable(opt.init_mom_range*torch.randn(dim,num_chain),requires_grad=True)
state_pos_init=Variable(opt.init_pos_range*torch.rand(dim,num_chain),requires_grad=True)
state_list_nnsg,state_mom_nnsg=NN_SGHMC_obj.parallel_sample(state_pos_init,state_mom_init,B,num_chain=opt.num_chain,TBPTT_step=10,total_step=opt.total_step,flag_manual_noise=True,eps=opt.step_size_NNSGHMC,mom_resample=opt.momentum_resample_step,inject_scale=opt.noise_injection,mode_train=False)



################ Compute average ESS #######################
end=opt.total_step
ESS_list_sg=np.zeros(num_chain)
for chain_ind in range(num_chain):
    state_list_un_sample=parallel_draw_sample(state_list_sg,end,burn_in=0,interval=1,draw_method='Single',chain_ind=chain_ind)
    np_state_list_un=state_list_un_sample.cpu().data.numpy()
    ESS_list_sg[chain_ind]=effectiveSampleSize(np_state_list_un)
print('SGHMC Avg. ESS: %s'%np.mean(ESS_list_sg))
np.savetxt('./TestResult/ESS_Cov_%s_scale_%s_sg'%(timestr,opt.cov_range),ESS_list_sg)

ESS_list_nnsg=np.zeros(num_chain)
for chain_ind in range(num_chain):
    state_list_un_sample=parallel_draw_sample(state_list_nnsg,end,burn_in=0,interval=1,draw_method='Single',chain_ind=chain_ind)
    np_state_list_un=state_list_un_sample.cpu().data.numpy()
    ESS_list_nnsg[chain_ind]=effectiveSampleSize(np_state_list_un)
print('NNSGHMC Avg. ESS: %s'%np.mean(ESS_list_nnsg))
np.savetxt('./TestResult/ESS_Cov_%s_scale_%s_nnsg'%(timestr,opt.cov_range),ESS_list_nnsg)


############## Compute KL Divergence #########################
end=opt.total_step
inter=100

KLD_nnsg_list=np.zeros(int(end/inter))
KLD_sg_list=np.zeros(int(end/inter))
print('Compute KLD of SGHMC')
for end_each in range(0,int(end/inter)):
    if (end_each+1)%20==0:
        print((end_each*inter))    
    state_list_un_sample=parallel_draw_sample(state_list_sg,int((end_each+1)*inter),
                                                                         burn_in=0,interval=1,draw_method='Cross')
    np_state_list_un=convert_to_Variable(state_list_un_sample,transpose=False)
    np_state_list_un=np_state_list_un.cpu().data.numpy()
    estimated_mean_un=np.mean(np_state_list_un,axis=0)
    estimated_var_un=np.cov(np_state_list_un,rowvar=False)

    KLD_un_sg=Gaussian_KL(np.expand_dims(estimated_mean_un,axis=1),mu.t().cpu().data.numpy(),estimated_var_un,var.cpu().data.numpy())
    KLD_sg_list[end_each]=KLD_un_sg

print('Final KLD of SGHMC:%s'%(KLD_sg_list[-1]))
np.savetxt('./TestResult/KLD_Cov_%s_scale_%s_sg'%(timestr,opt.cov_range),KLD_sg_list)



print('Compute KLD of NNSGHMC')
for end_each in range(0,int(end/inter)):
    if (end_each+1)%20==0:
        print((end_each*inter))    
    state_list_un_sample=parallel_draw_sample(state_list_nnsg,int((end_each+1)*inter),
                                              burn_in=0,interval=1,draw_method='Cross')
    np_state_list_un=convert_to_Variable(state_list_un_sample,transpose=False)
    np_state_list_un=np_state_list_un.cpu().data.numpy()
    estimated_mean_un=np.mean(np_state_list_un,axis=0)
    estimated_var_un=np.cov(np_state_list_un,rowvar=False)

    KLD_un_nnsg=Gaussian_KL(np.expand_dims(estimated_mean_un,axis=1),mu.t().cpu().data.numpy(),estimated_var_un,var.cpu().data.numpy())
    KLD_nnsg_list[end_each]=KLD_un_nnsg
print('Final KLD of NNSGHM:%s'%(KLD_nnsg_list[-1]))
np.savetxt('./TestResult/KLD_Cov_%s_scale_%s_nnsg'%(timestr,opt.cov_range),KLD_nnsg_list)
                    
###################### Generate Figure ################################
fig,ax=plt.subplots()
plt.style.use('ggplot')


ax.semilogy(inter*np.arange(1,len(KLD_nnsg_list)+1),KLD_nnsg_list,color='y',label='NNSGHMC')
ax.semilogy(inter*np.arange(1,len(KLD_nnsg_list)+1),KLD_sg_list,color='b',label='SGHMC')
ax.set_xlabel('Iterations',fontsize=14)
ax.set_ylabel('KL Divergence',fontsize=14)
ax.legend(prop={'size':15})
ax.tick_params(labelsize=12)

plt.savefig('./TestResult/Fig_Cov_%s_scale_%s.pdf'%(timestr,opt.cov_range),dpi=150,bbox_inches='tight')







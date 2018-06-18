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
import operator
from torch.utils.data import Dataset, DataLoader


def generate_accuracy(MLP_mnist,state_list_sghmc,state_list_nnsghmc,interval=100):
    len_sghmc=len(state_list_sghmc)
    len_nnsghmc=len(state_list_nnsghmc)
    Acc_sghmc_list=np.zeros(len_sghmc)
    Acc_nnsghmc_list=np.zeros(len_nnsghmc)
    NLL_sghmc_list=np.zeros(len_sghmc)
    NLL_nnsghmc_list=np.zeros(len_nnsghmc)
    
    for ind,state_sghmc in zip(range(len_sghmc),state_list_sghmc):
        #draw_list_sghmc=draw_samples(state_list_sghmc,burn_in=0,interval=1,draw_method='Cross')
        print('%s'%(ind+1))
        state_nnsghmc=state_list_nnsghmc[ind]
        Acc_sghmc,NLL_sghmc=Test_accuracy(test_loader,MLP_mnist,state_sghmc,data_number=10000.)
        Acc_nnsghmc,NLL_nnsghmc=Test_accuracy(test_loader,MLP_mnist,state_nnsghmc,data_number=10000.)
        
        Acc_sghmc_list[ind]=1.-Acc_sghmc
        Acc_nnsghmc_list[ind]=1.-Acc_nnsghmc
        NLL_sghmc_list[ind]=-NLL_sghmc.data.cpu().numpy()
        NLL_nnsghmc_list[ind]=-NLL_nnsghmc.data.cpu().numpy()
    return Acc_sghmc_list,Acc_nnsghmc_list,NLL_sghmc_list,NLL_nnsghmc_list
def DQContour(Q_MLP,D_MLP,dim=[0,1],range1=[-2.5,2.5],range2=[-2.5,2.5],range3=-0.1,num=200,flag_D=False,flag_energy=False):
    if flag_D==False:
        x1=np.linspace(range1[0],range1[1],num)
        x2=np.linspace(range2[0],range2[1],num)
        X1,X2=np.meshgrid(x1,x2)
        X_grid=np.stack((-X1,X2),axis=2)
        X_tensor=Variable(torch.from_numpy(X_grid).float().cuda())
        out=np.squeeze(np.abs(Q_MLP.forward(X_tensor).cpu().data.numpy()))
    
        return X1,X2,out
    if flag_D==True:
        if flag_energy==True:
            x1=np.linspace(range1[0],range1[1],num)
        else:
            x1=np.linspace(range1[0],range1[1],num)
            
        x2=np.linspace(range2[0],range2[1],num)
        X1,X2=np.meshgrid(x1,x2)
        X3=np.tile(range3,X1.shape)
        #Tracer()()
        if dim==[0,1]:
            X_grid=np.stack((-X1,X2,X3),axis=2)
        elif dim==[0,2]:
            X_grid=np.stack((-X1,X3,X2),axis=2)
        elif dim==[1,2]:
            X_grid=np.stack((X3,X1,X2),axis=2)
            
        X_tensor=Variable(torch.from_numpy(X_grid).float().cuda())
        
        out=np.squeeze(np.abs(50*D_MLP.forward(X_tensor).cpu().data.numpy()))
        return X1,X2,out
    
    
class MLP(nn.Module):
    def __init__(self,input_dim,hidden,out_size=1):
        super(MLP,self).__init__()
        self.input_dim=input_dim
        self.hidden=hidden
        self.out_func=nn.Linear(hidden,out_size)
        self.features=nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            
#             nn.Linear(hidden,hidden),
#             nn.ReLU(),
            nn.Linear(hidden,out_size)
        )
    def forward(self,x):
        #out=1+1./(1+torch.exp(10*self.features(x)))
        #out=1+torch.log((1+0.01*torch.abs(self.features(x))))
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
#             nn.Linear(hidden,hidden),
#             nn.ReLU(),
            nn.Linear(hidden,out_size), 
        )
    def forward(self,x):
        out=torch.abs(self.features(x))
        return out    
    
    
    
###########################################################################################
torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''
Read Stored Results
'''

# Acc_sghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_sghmc')
# Acc_nnsghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_nnsghmc_fd_0.18')
# Acc_SGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_sgld_correct_0.2')
# Acc_PSGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_psgld_step_1.4')
# Std_sghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_sghmc')
# Std_nnsghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_nnsghmc_fd_0.18')
# Std_SGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_sgld_correct_0.2')
# Std_PSGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_psgld_step_1.4')


Acc_sghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_sghmc_TEST')
Acc_nnsghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_nnsghmc_TEST')
Acc_SGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_sgld_TEST')
Acc_PSGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Avg_psgld_step_1.4_TEST')
Std_sghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_sghmc_TEST')
Std_nnsghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_nnsghmc_TEST')
Std_SGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_sgld_TEST')
Std_PSGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/Acc_Std_psgld_step_1.4_TEST')






Acc_sghmc_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Avg_sghmc')
Acc_nnsghmc_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Avg_nnsghmc')
Acc_SGLD_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Avg_sgld')
Acc_PSGLD_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Avg_psgld_step_1.3')
Std_sghmc_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Std_sghmc')
Std_nnsghmc_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Std_nnsghmc')
Std_SGLD_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Std_sgld')
Std_PSGLD_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/Acc_Std_psgld_step_1.3')




Acc_sghmc_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Avg_sghmc')
Acc_nnsghmc_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Avg_nnsghmc')
Acc_SGLD_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Avg_sgld')
Acc_PSGLD_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Avg_psgld_step_1.3')
Std_sghmc_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Std_sghmc')
Std_nnsghmc_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Std_nnsghmc')
Std_SGLD_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Std_sgld')
Std_PSGLD_d=np.loadtxt('./Dataset_Gen_Long_Run/Acc_Std_psgld_step_1.3')



##############################################################################################################

NLL_sghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Avg_sghmc')
NLL_nnsghmc_n=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Avg_nnsghmc_fd_0.18')
NLL_SGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Avg_sgld_correct_0.2')
NLL_PSGLD_n=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Avg_psgld_step_1.4')
Std_sghmc_n_nl=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Std_sghmc')
Std_nnsghmc_n_nl=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Std_nnsghmc_fd_0.18')
Std_SGLD_n_nl=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Std_sgld_correct_0.2')
Std_PSGLD_n_nl=np.loadtxt('./ReLU_Generalization_Long_Run/NLL_Std_psgld_step_1.4')




NLL_sghmc_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Avg_sghmc')
NLL_nnsghmc_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Avg_nnsghmc')
NLL_SGLD_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Avg_sgld')
NLL_PSGLD_s=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Avg_psgld_step_1.3')
Std_sghmc_s_nl=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Std_sghmc')
Std_nnsghmc_s_nl=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Std_nnsghmc')
Std_SGLD_s_nl=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Std_sgld')
Std_PSGLD_s_nl=np.loadtxt('./Sigmoid_Generalization_Long_Run/NLL_Std_psgld_step_1.3')




NLL_sghmc_d=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Avg_sghmc')
NLL_nnsghmc_d=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Avg_nnsghmc')
NLL_SGLD_d=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Avg_sgld')
NLL_PSGLD_d=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Avg_psgld_step_1.3')
Std_sghmc_d_nl=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Std_sghmc')
Std_nnsghmc_d_nl=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Std_nnsghmc')
Std_SGLD_d_nl=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Std_sgld')
Std_PSGLD_d_nl=np.loadtxt('./Dataset_Gen_Long_Run/NLL_Std_psgld_step_1.3')



########################################################################

time_list_sghmc=12000/12000*100/120*np.linspace(0,len(Std_sghmc_n),len(Std_sghmc_n))
time_list_nnsghmc=12000/12000*100/120*np.linspace(0,len(Std_sghmc_n),len(Std_sghmc_n))
time_list_sgld=12000/12000*100/120*np.linspace(0,len(Std_sghmc_n),len(Std_sghmc_n))


time_list_sghmc_d=12000/12000*40/120*np.linspace(0,len(Std_sghmc_d),len(Std_sghmc_d))
time_list_nnsghmc_d=12000/12000*40/120*np.linspace(0,len(Std_sghmc_d),len(Std_sghmc_d))
time_list_sgld_d=12000/12000*40/120*np.linspace(0,len(Std_sghmc_d),len(Std_sghmc_d))

plt.style.use('ggplot')
f,ax=plt.subplots(2,3,sharex='col',figsize=(20,7))


ax[0,0].errorbar(time_list_sghmc,Acc_sghmc_n,color='b',yerr=Std_sghmc_n/np.sqrt(10),linewidth=2,label='SGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[0,0].errorbar(time_list_nnsghmc,Acc_nnsghmc_n,color='y',yerr=Std_nnsghmc_n/np.sqrt(10),linewidth=2,label='NNSGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[0,0].errorbar(time_list_sgld,Acc_SGLD_n,color='m',yerr=Std_SGLD_n/np.sqrt(10),linewidth=2,label='SGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[0,0].errorbar(time_list_sgld,Acc_PSGLD_n,color='r',yerr=Std_PSGLD_n/np.sqrt(10),linewidth=2,label='PSGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[0,0].legend(prop={'size':17})
ax[0,0].set_ylim(0.016,0.025)
ax[0,0].tick_params(labelsize=15)
ax[0,0].set_ylabel('Error',fontsize=17)
ax[0,0].set_title('Network Generalization',fontsize=15)




ax[0,1].errorbar(time_list_sghmc,Acc_sghmc_s,color='b',yerr=Std_sghmc_s/np.sqrt(10),linewidth=2,label='SGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[0,1].errorbar(time_list_nnsghmc,Acc_nnsghmc_s,color='y',yerr=Std_nnsghmc_s/np.sqrt(10),linewidth=2,label='NNSGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[0,1].errorbar(time_list_sgld,Acc_SGLD_s,color='m',yerr=Std_SGLD_s/np.sqrt(10),linewidth=2,label='SGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[0,1].errorbar(time_list_sgld,Acc_PSGLD_s,color='r',yerr=Std_PSGLD_s/np.sqrt(10),linewidth=2,label='PSGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[0,1].set_ylim(0.022,0.029)
ax[0,1].tick_params(labelsize=15)
ax[0,1].set_title('Sigmoid Generalization',fontsize=15)





ax[0,2].errorbar(time_list_sghmc_d,Acc_sghmc_d,color='b',yerr=Std_sghmc_d/np.sqrt(10),linewidth=1.5,label='SGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[0,2].errorbar(time_list_nnsghmc_d,Acc_nnsghmc_d,color='y',yerr=Std_nnsghmc_d/np.sqrt(10),linewidth=1.5,label='NNSGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[0,2].errorbar(time_list_sgld_d,Acc_SGLD_d,color='m',yerr=Std_SGLD_d/np.sqrt(10),linewidth=1.5,label='SGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[0,2].errorbar(time_list_sgld_d,Acc_PSGLD_d,color='r',yerr=Std_PSGLD_d/np.sqrt(10),linewidth=1.5,label='PSGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[0,2].set_ylim(0.013,0.02)
ax[0,2].tick_params(labelsize=15)
ax[0,2].set_title('Dataset Generalization',fontsize=15)



######################################################################################################################
ax[1,0].errorbar(time_list_sghmc,NLL_sghmc_n,color='b',yerr=Std_sghmc_n_nl/np.sqrt(10),linewidth=2,label='SGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[1,0].errorbar(time_list_nnsghmc,NLL_nnsghmc_n,color='y',yerr=Std_nnsghmc_n_nl/np.sqrt(10),linewidth=2,label='NNSGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[1,0].errorbar(time_list_sgld,NLL_SGLD_n,color='m',yerr=Std_SGLD_n_nl/np.sqrt(10),linewidth=2,label='SGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[1,0].errorbar(time_list_sgld,NLL_PSGLD_n,color='r',yerr=Std_PSGLD_n_nl/np.sqrt(10),linewidth=2,label='PSGLD',errorevery=25,capsize=6,elinewidth=2.)
#ax[1,0].legend(prop={'size':17})
ax[1,0].set_ylim(600,900)
ax[1,0].tick_params(labelsize=15)
ax[1,0].set_xlim(0,105)
ax[1,0].set_xlabel('Epoch',fontsize=15)
ax[1,0].set_ylabel('Neg. LL',fontsize=17)


ax[1,1].errorbar(time_list_sghmc,NLL_sghmc_s,color='b',yerr=Std_sghmc_s_nl/np.sqrt(10),linewidth=2,label='SGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[1,1].errorbar(time_list_nnsghmc,NLL_nnsghmc_s,color='y',yerr=Std_nnsghmc_s_nl/np.sqrt(10),linewidth=2,label='NNSGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[1,1].errorbar(time_list_sgld,NLL_SGLD_s,color='m',yerr=Std_SGLD_s_nl/np.sqrt(10),linewidth=2,label='SGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[1,1].errorbar(time_list_sgld,NLL_PSGLD_s,color='r',yerr=Std_PSGLD_s_nl/np.sqrt(10),linewidth=2,label='PSGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[1,1].set_ylim(850,1300)
ax[1,1].set_xlim(0,105)
ax[1,1].tick_params(labelsize=15)
ax[1,1].set_xlabel('Epoch',fontsize=15)




ax[1,2].errorbar(time_list_sghmc_d,NLL_sghmc_d,color='b',yerr=Std_sghmc_d_nl/np.sqrt(10),linewidth=1.5,label='SGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[1,2].errorbar(time_list_nnsghmc_d,NLL_nnsghmc_d,color='y',yerr=Std_nnsghmc_d_nl/np.sqrt(10),linewidth=1.5,label='NNSGHMC',errorevery=25,capsize=6,elinewidth=2.)
ax[1,2].errorbar(time_list_sgld_d,NLL_SGLD_d,color='m',yerr=Std_SGLD_d_nl/np.sqrt(10),linewidth=1.5,label='SGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[1,2].errorbar(time_list_sgld_d,NLL_PSGLD_d,color='r',yerr=Std_PSGLD_d_nl/np.sqrt(10),linewidth=1.5,label='PSGLD',errorevery=25,capsize=6,elinewidth=2.)
ax[1,2].set_ylim(220,300)
ax[1,2].tick_params(labelsize=15)
ax[1,2].set_xlabel('Epoch',fontsize=15)
plt.savefig('./BNN_MNIST_Result/MNIST_Group_Result_Test.pdf',dpi=150,bbox_inches = 'tight')




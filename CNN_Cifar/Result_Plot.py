######################################### Import Necessary Packages#####################################################
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
import json
import collections
from torch.utils.data import Dataset, DataLoader
# Import Custom packages
from Cifar_Dataloader import *
from CNN_Module import *
from Test_Module import *
from Util import *
from CNN_Module import *
from CNN_Q_D import *
from Test_Module import *
from CNN_training_func import *
from CNN_Sampler import *
########################################################################################################################

# Set default tensor type in GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def DQContour(Q_MLP, D_MLP, dim=[0, 1], range1=[-2.5, 2.5], range2=[-2.5, 2.5], range3=-0.1, num=200, flag_D=False,
          flag_energy=False):
    if flag_D == False:
        x1 = np.linspace(range1[0], range1[1], num)
        x2 = np.linspace(range2[0], range2[1], num)
        X1, X2 = np.meshgrid(x1, x2)
        X_grid = np.stack((-X1, X2), axis=2)
        X_tensor = torch.tensor(torch.from_numpy(X_grid).float().data).cuda()
        out = np.squeeze(np.abs(Q_MLP.forward(X_tensor).cpu().data.numpy()))

        return X1, X2, out
    if flag_D == True:
        if flag_energy == True:
            x1 = np.linspace(range1[0], range1[1], num)
        else:
            x1 = np.linspace(range1[0], range1[1], num)

        x2 = np.linspace(range2[0], range2[1], num)
        X1, X2 = np.meshgrid(x1, x2)
        X3 = np.tile(range3, X1.shape)
        # Tracer()()
        if dim == [0, 1]:
            X_grid = np.stack((-X1, X2, X3), axis=2)
        elif dim == [0, 2]:
            X_grid = np.stack((-X1, X3, X2), axis=2)
        elif dim == [1, 2]:
            X_grid = np.stack((X3, X1, X2), axis=2)

        X_tensor = Variable(torch.from_numpy(X_grid).float().cuda())

        out = np.squeeze(np.abs(5 * D_MLP.forward(X_tensor).cpu().data.numpy()))
        return X1, X2, out

model_str='20180919-073244_30'
model_load_path='./tmp_model_save/'



################################# Generate DQ Contour ##############################


# Load Model
Q_MLP=MLP(input_dim=2,hidden=10,out_size=1)
D_MLP=Positive_MLP(input_dim=3,hidden=10,out_size=1)

Q_MLP.load_state_dict(torch.load('%sQ_MLP_%s'%(model_load_path,model_str)))
D_MLP.load_state_dict(torch.load('%sD_MLP_%s'%(model_load_path,model_str)))

# Generate Contour
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(18,5))
# Ax1 with Q Contour
X1,X2,out=DQContour(Q_MLP,D_MLP,range1=[0,15],range2=[-2,2])
CS1=ax1.contourf(X1,X2,out,50)
fig.colorbar(CS1,ax=ax1)
# Ax2 with D Contour 0,1 dim
X1_D,X2_D,out_D=DQContour(Q_MLP,D_MLP,dim=[0,1],range1=[0,15],range2=[-2,2],range3=0.5,flag_D=True)
CS2=ax2.contourf(X1_D,X2_D,out_D,50)
fig.colorbar(CS2,ax=ax2)
# Ax2 with D Contour 1,2 dim
X1_D,X2_D,out_D=DQContour(Q_MLP,D_MLP,dim=[1,2],range1=[-2,2],range2=[-2,2],range3=-5,flag_D=True)
CS3=ax3.contourf(X1_D,X2_D,out_D,50)
fig.colorbar(CS3,ax=ax3)
plt.savefig('./Results/Countour_')


# #X1,X2,out=DQContour(Q_MLP,D_MLP,range1=[0,15],range2=[-5,5])
# X1_D,X2_D,out_D=DQContour(Q_MLP,D_MLP,dim=[0,1],range1=[4,15],range2=[-0.5,0.5],range3=0.3,flag_D=True)
#
# fig1,ax1=plt.subplots()
#
# #CS=ax1.contourf(X1,X2,out,50)
# CS=ax1.contourf(X1_D,X2_D,out_D,50)
# cbar=fig1.colorbar(CS)
plt.savefig('./Results/Countour_%s.png'%(model_str))

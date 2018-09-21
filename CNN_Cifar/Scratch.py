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
import gpustat
from torch.utils.data import Dataset, DataLoader
# Import Custom packages
from Cifar_Dataloader import *
from Test_Module import *
from Util import *
# Set default tensor type in GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))
num_CNN=20
p=0.4

# u_list=np.random.uniform(0,1,20)
# p=0.4
# ind_list=(u_list<=p)
# ind_list=[i for i,x in enumerate(ind_list) if x]
# A=torch.randn(20,10)
# B=A[ind_list]
# print(ind_list)
# print(A)
# print(B)
A=torch.tensor(3.,requires_grad=True)
B=A**2
A.data=torch.tensor(4.)
print(B)
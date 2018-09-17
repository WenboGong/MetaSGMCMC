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
from torch.utils.data import Dataset, DataLoader
# Import Custom packages
from Cifar_Dataloader import *
from Test_Module import *
from Util import *
# Set default tensor type in GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#######################################################################################################################
x = torch.tensor([1, 2], requires_grad=True, dtype=torch.float)
y = torch.tensor([1, 2], requires_grad=True, dtype=torch.float)

x = x ** 2
y = y ** 2

z=torch.sum(x+y)

z.backward()

assert x.grad is None and y.grad is None
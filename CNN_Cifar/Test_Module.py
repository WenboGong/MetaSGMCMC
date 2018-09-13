############################ Import Necessary packages ##########################################
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
###########################
def Test_Accuracy(CNN,test_loader):
    correct = 0
    total = 0
    for data in enumerate(test_loader):
        x,y=data[1][0].cuda(),data[1][1].cuda()
        out=CNN.forward(x)
        _, predicted = torch.max(out, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    Acc=correct/total
    return Acc
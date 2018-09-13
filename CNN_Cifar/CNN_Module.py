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

class Example_CNN(nn.Module):
    def __init__(self,num_channel=3,height=32,width=32):
        super(Example_CNN, self).__init__()
        self.conv1=nn.Conv2d(3,16,3)
        self.conv2=nn.Conv2d(16,16,3)
        self.fc1=nn.Linear(16*6*6,100)
        self.fc2=nn.Linear(100,10)
        self.softmax=nn.Softmax()

    def forward(self, x):
        # x should be batch x height x width
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #prob=self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
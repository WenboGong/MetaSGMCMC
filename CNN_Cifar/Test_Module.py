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
def Test_Accuracy_example(CNN,test_loader):
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

def Test_Accuracy(CNN,test_loader,weight,test_number=10000.,flag_opt=False):
    correct=0
    out_overall=0 # for NLL
    for data in enumerate(test_loader):
        x, y = data[1][0].cuda(), data[1][1].cuda()
        y_ = torch.unsqueeze(y, dim=1)
        batch_y = int(y_.shape[0])
        y_hot = torch.zeros(batch_y, 10).scatter_(1, y_, 1)
        y_hot=y_hot.float()
        if flag_opt==False:
            out_log=CNN.predict(x,weight)
        else:
            out_log=CNN.predict(x)

        _, y_pred = torch.max(out_log, dim=1)
        out_overall += torch.sum(y_hot * out_log)

        correct += (y == y_pred).sum().item()
    return correct/test_number,out_overall

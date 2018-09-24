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
    out_overall=0
    for data in enumerate(test_loader):
        x,y=data[1][0].cuda(),data[1][1].cuda()
        y_ = torch.unsqueeze(y, dim=1)
        batch_y = int(y_.shape[0])
        y_hot = torch.zeros(batch_y, CNN_out_dim).scatter_(1, y_, 1)
        y_hot = y_hot.float()

        out=CNN.forward(x)
        _, predicted = torch.max(out, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        out_log=CNN.log_prob(x)
        out_overall += torch.sum(y_hot * out_log)

    Acc=correct/total
    return Acc,out_overall.data.cpu().numpy()

def Test_Accuracy(CNN,test_loader,weight,test_number=10000.,flag_opt=False,CNN_out_dim=10):
    '''
    This function only evaluate the accuracy of samples taken across chain at certain time
    :param CNN: The CNN objective
    :param test_loader: The data loader
    :param weight: The weight tensor at current time
    :param test_number: The size of the test set
    :param flag_opt: if evaluation of optimization method is used
    :return:
    '''
    correct=0
    out_overall=0 # for NLL
    for data in enumerate(test_loader):
        x, y = data[1][0].cuda(), data[1][1].cuda()
        y_ = torch.unsqueeze(y, dim=1)
        batch_y = int(y_.shape[0])
        y_hot = torch.zeros(batch_y, CNN_out_dim).scatter_(1, y_, 1)
        y_hot=y_hot.float()
        if flag_opt==False:
            out_log=CNN.predict(x,weight)
        else:
            out_log=CNN.predict(x)

        _, y_pred = torch.max(out_log, dim=1)
        out_overall += torch.sum(y_hot * out_log)

        correct += (y == y_pred).sum().item()
    return correct/test_number,out_overall

# Not usable for datagen
class Sequential_Accuracy:
    '''
    This class is to enable the evaluation for drawing samples across time and across chain

    NOTE: The test_loader should not use batch sub sampling and no random shuffle

    '''
    def __init__(self,test_loader,CNN,size_of_marginal=(10000,10)):
        self.test_loader=test_loader
        self.CNN=CNN
        self.marginal=torch.zeros(size_of_marginal).cuda()
        self.counter=1
    def accumulate(self,likelihood):
        '''
        This is to update the overall log marginal probability
        :param likelihood: The new likelihood evaluated at current time with cross chain samples
        :return: None
        '''
        self.marginal = self.marginal + 1 / self.counter * (likelihood - self.marginal)
        self.counter+=1
    def get_marginal(self):
        return self.marginal
    def Test_Accuracy(self,weight):
        '''
        This function evaluate the accuracy when samples are taken across chain and across time

        :param weight: The weight tensor at current time
        :return: The accuracy and test NLL
        '''
        correct = 0
        total = 0
        out_overall=0
        with torch.no_grad():
            for _, (x, y) in enumerate(self.test_loader):

                x, y = x.cuda(), y.cuda()
                y_ = torch.unsqueeze(y, dim=1)
                batch_y = int(y_.shape[0])
                y_hot = torch.zeros(batch_y, 10).scatter_(1, y_, 1)
                y_hot = y_hot.float()
                likelihood = F.softmax(self.CNN.predict(x,weight),dim=1)# N x 10

                self.accumulate(likelihood.data)
                marginal = self.get_marginal()
                _, yhat = torch.max(marginal, 1)
                total += y.size(0)
                correct += (yhat == y.data).sum()
                out_overall += torch.sum(y_hot * torch.log(marginal))
        return float(correct) / float(total),out_overall.data.cpu().numpy()



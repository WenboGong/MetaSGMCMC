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

from BNN_Util import *

class Classifier(nn.Module):
    def __init__(self,input_dim,hidden,out_size=1):
        super(Classifier,self).__init__()
        self.input_dim=input_dim
        self.hidden=hidden
        self.out_func=nn.Linear(hidden,out_size)
        self.features=nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_size),
            nn.LogSoftmax(dim=1)
        )
    def predict(self,X):
        X=X.view(-1,28*28)
        out=self.features(X)
        return out
    
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
class BNN:
    def __init__(self,dim,hidden,layer_num,dim_out=10,act_func='ReLU'):
        self.dim=dim
        self.hidden=hidden
        self.layer_num=layer_num
        self.dim_out=dim_out
        if act_func=='ReLU':
            self.act_func=F.relu
        elif act_func=='Sigmoid':
            self.act_func=F.sigmoid
        
    def get_total_dim(self):
        dim_counter=0
        for layer in range(self.layer_num):
            if layer==0:
                dim_in=self.dim
                dim_out=self.hidden    
            elif layer==self.layer_num-1:
                dim_in=self.hidden
                dim_out=self.dim_out   
            else:
                dim_in=self.hidden
                dim_out=self.hidden
            dim_counter+=dim_in*dim_out+dim_out
        return dim_counter
    def forward(self,X,weight):
        '''
        weight is chain x dim 
        X is N x dim
        '''
        X=X.view(-1,28*28)
        #Tracer()()
        num_chain=int(weight.data.shape[0])
        X_3d=torch.unsqueeze(X,dim=0).repeat(num_chain,1,1)
        dim_counter=0
        for layer in range(self.layer_num):
            #print(layer)
            if layer==0:
                dim_in=self.dim
                dim_out=self.hidden
                IN=X_3d
            elif layer==self.layer_num-1:
                dim_in=self.hidden
                dim_out=self.dim_out
                IN=h
            else:
                dim_in=self.hidden
                dim_out=self.hidden
                IN=h   
            W=weight[:,dim_counter:dim_counter+dim_in*dim_out].contiguous().view(num_chain,dim_in,dim_out)
            dim_counter+=dim_in*dim_out
            b=torch.unsqueeze(weight[:,dim_counter:(dim_counter+dim_out)],dim=1)
            dim_counter+=dim_out
            
            if layer==self.layer_num-1:
                h=F.log_softmax(IN.matmul(W)+b,dim=2)
            else:
                h=self.act_func(IN.matmul(W)+b)
                #h=F.relu(IN.matmul(W)+b)
                #h=F.sigmoid(IN.matmul(W)+b)
        return h
    def grad_BNN(self,X,y,weight,data_N,coef=1.,sigma=1.,flag_retain=False):
        '''
        y is N x 10 with zero and 1
        weitgh is num_chain x total_dim
        '''
        num_chain=int(weight.data.shape[0])
        batch=float(X.data.shape[0])
        y_=torch.unsqueeze(y,dim=0).repeat(num_chain,1,1) # chain x N x 10
        prob_out=data_N*torch.mean(torch.sum(y_*self.forward(X,weight),dim=2),dim=1,keepdim=True)# chain x 1
        
        
        
        
        prior_prob= torch.sum(float(-0.5*np.log(2.*3.1415926*(sigma**2)))-(weight**2)/(2*(sigma**2)),dim=1,keepdim=True)#num_chain x 1
        dtheta_data=grad(prob_out,weight,torch.ones(prob_out.data.shape),allow_unused=True,retain_graph=flag_retain)[0]
        ##### prior assume Gaussian(0, I) ####
        dtheta_prior=-weight/((sigma**2))
        return -(dtheta_data+dtheta_prior),prob_out+coef*prior_prob,-(dtheta_data+coef*dtheta_prior),prob_out+prior_prob
    def predict(self,X,weight):
        num_chain=float(weight.data.shape[0])
        out_log=logsumexp(self.forward(X,weight),dim=0)-float(np.log(num_chain))
        return out_log
    def entropy(self,X,weight):
        log_prob=self.predict(X,weight) # N x D
        prob=torch.exp(log_prob) # N x D
        H=-torch.mean(log_prob*prob,dim=1,keepdim=True) # N x 1
        return H
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h
def Test_accuracy(test_loader,BNN_obj,weight,data_number=10000.,flag_opt=False):
    correct=0
    out_overall=0
    for data in enumerate(test_loader):
        X,y=Variable(data[1][0].cuda()),data[1][1]
        y=y.cuda()
        y_=torch.unsqueeze(y,dim=1).cuda()
        batch_y=int(y_.shape[0])
        y_hot = Variable(torch.zeros(batch_y, BNN_obj.dim_out).scatter_(1, y_, 1)).float()
        if flag_opt==False:
            
            out_log=BNN_obj.predict(X,weight)
            #Tracer()()
            #out_overall+=out_log
        else:
            
            out_log=BNN_obj.predict(X)
            #Tracer()()
            #out_overall+=out_log
            #Tracer()()
        _,y_pred=torch.max(out_log,dim=1)
        out_overall+=torch.sum(y_hot*out_log)
        #Tracer()()
        
        correct+=torch.sum(y==y_pred.data).float()
    return correct/data_number,out_overall
def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
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

class Parallel_CNN:
    '''
    The Parallel CNN Object (not really parallel T_T, for loop instead...)

    Args:
        num_CNN: The number of CNNs for Cross Chain loss
        num_channel: The input channel
    Methods:
        sub_forward(weight,x): The forward pass for individual CNN
        split_dimension(weight): Split the weight vector into desired weight tensor for CNN evaluation
        forawrd(x,weight): Compute the log probability for all CNN
    '''
    def __init__(self,num_CNN=20,num_channle=3,filter_size=3,out_channel=16,flat_size=16*6*6,fc_size=100):
        self.num_CNN=num_CNN
        self.num_channel=num_channle
        self.filter_size=filter_size
        self.out_channel=out_channel
        self.flat_size=flat_size
        self.fc_size=fc_size
    def split_dimension(self,weight):
        '''
        Split the vector into the desired shape

        :param weight: The tensor with size :math:`numCNN x total dim`
        :return: tensor with correct size: conv1, conv2,fc1,fc2 each with size :math:`num_CNN x ...`
        '''
        # Define the dimension for each layer
        dim_tracker=0
        conv1_size=self.out_channel*self.num_channel*self.filter_size**2
        conv2_size=self.out_channel*self.out_channel*self.filter_size**2
        fc1_size=self.flat_size*self.fc_size
        fc1_bias_size=self.fc_size
        fc2_size=self.fc_size*10
        fc2_bias_size=10
        # Extraction the weight
        conv1_raw_tensor=weight[:,0:conv1_size]
        dim_tracker+=conv1_size

        conv1_bias_raw_tensor=weight[:,dim_tracker:dim_tracker+self.out_channel]
        dim_tracker+=self.out_channel


        conv2_raw_tensor=weight[:,dim_tracker:dim_tracker+conv2_size]
        dim_tracker+=conv2_size
        conv2_bias_raw_tensor=weight[:,dim_tracker:dim_tracker+self.out_channel]
        dim_tracker+=self.out_channel


        fc1_raw_tensor=weight[:,dim_tracker:dim_tracker+fc1_size]
        dim_tracker+=fc1_size
        fc1_bias_raw_tensor=weight[:,dim_tracker:dim_tracker+fc1_bias_size]
        dim_tracker+=fc1_bias_size
        fc2_raw_tensor=weight[:,dim_tracker:dim_tracker+fc2_size]
        dim_tracker+=fc2_size
        fc2_bias_raw_tensor=weight[:,dim_tracker:dim_tracker+fc2_bias_size]
        # reshape
        conv1_weight=conv1_raw_tensor.view(-1,self.out_channel,self.num_channel,self.filter_size,self.filter_size)
        conv1_bias=conv1_bias_raw_tensor.view(-1,self.out_channel)
        conv2_weight = conv2_raw_tensor.view(-1, self.out_channel, self.out_channel, self.filter_size,
                                             self.filter_size)
        conv2_bias=conv2_bias_raw_tensor.view(-1,self.out_channel)

        fc1_weight=fc1_raw_tensor.view(-1,self.fc_size,self.flat_size)
        fc1_bias=fc1_bias_raw_tensor.view(-1,self.fc_size)
        fc2_weight=fc2_raw_tensor.view(-1,10,self.fc_size)
        fc2_bias=fc2_bias_raw_tensor.view(-1,10)

        return [conv1_weight,conv1_bias,conv2_weight,conv2_bias,fc1_weight,fc1_bias,fc2_weight,fc2_bias]
    def sub_forward(self,weight_list,x,ind):
        '''
        The forward pass for individual sub CNN
        :param weight_list: list of Tensor containing correct weight dimension
        :param x: input image with size :math:`N x num_channel x height x width`
        :param ind: The indicator for which CNN to evaluate
        :return: Tensor with size :math:`N x 10`
        '''
        conv1_output=F.max_pool2d(F.relu(F.conv2d(x,weight_list[0][ind],weight_list[1][ind])),2)
        conv2_output=F.max_pool2d(F.relu(F.conv2d(conv1_output,weight_list[2][ind],weight_list[3][ind])),2)
        x_fc=conv2_output.view(-1,self.flat_size)
        fc1_output=F.relu(F.linear(x_fc,weight_list[4][ind],weight_list[5][ind]))
        fc2_output=F.relu(F.linear(fc1_output,weight_list[6][ind],weight_list[7][ind]))
        return fc2_output
    def forward(self,x,weight):
        '''
        The is to compute log probability for all CNN
        :param x: image with size :math: `N x num_channel x height x width`
        :param weight: weight tensor with size :math:`num_CNN x total dimension`
        :return: log_prob_all with size :math:`num_CNN x N x 10`
        '''
        # split the weight
        weight_list=self.split_dimension(weight)
        # evaluations
        total_iter=weight.shape[0]
        for ind in range(total_iter):
            #print(ind)
            out_CNN=self.sub_forward(weight_list,x,ind)
            log_prob = F.log_softmax(out_CNN, dim=1)
            if ind==0:
                log_prob_all=torch.unsqueeze(log_prob,dim=0)
            else:
                log_prob_all=torch.cat((log_prob_all,torch.unsqueeze(log_prob,dim=0)),dim=0)

        return log_prob_all
    def part_free_grad_CNN(self,x,y,weight,data_N,coef=1.,sigma=22.,flag_retain=False):
        '''
        This method implement the same function as grad_CNN, but vastly reduce the memory usage but more computational expensive.
        :param x: The image
        :param y: The label
        :param weight: Current states
        :param data_N: The size of training data
        :param coef: The regularization term to account for dimensionality mismatch, no need for training purpose
        :param sigma: The std of prior
        :param flag_retain: Whether to retain the graph after backward()
        :return: The grad_U
        '''
        # split the weight
        weight_list = self.split_dimension(weight)
        # evaluations
        total_iter = weight.shape[0]
        #y_ = torch.unsqueeze(y, dim=0).repeat(sample_size, 1, 1)
        for ind in range(total_iter):
            #print(ind)
            out_CNN = self.sub_forward(weight_list, x, ind) # N x 10
            log_prob = data_N*torch.mean(torch.sum(y*F.log_softmax(out_CNN, dim=1),dim=1,keepdim=True),dim=0) #
            grad_log_prob=grad(log_prob,weight,allow_unused=False)[0][ind:ind+1,:]

            if ind == 0:
                log_prob_all = torch.unsqueeze(log_prob, dim=0)
                grad_list=torch.tensor(grad_log_prob.data)
                del grad_log_prob
            else:
                log_prob_all = torch.cat((log_prob_all, torch.unsqueeze(log_prob, dim=0)), dim=0)
                grad_list=torch.cat((grad_list,grad_log_prob),dim=0)
                del grad_log_prob
        dtheta_prior = -weight / ((sigma ** 2))
        G=-(grad_list+dtheta_prior)
        return G

    def get_dimension(self):
        '''
        Compute the total dimension needed for weight tensor
        :return: total dimension needed
        '''
        conv1_size = self.out_channel * self.num_channel * self.filter_size ** 2
        conv2_size = self.out_channel * self.out_channel * self.filter_size ** 2
        fc1_size = self.flat_size * self.fc_size
        fc1_bias_size = self.fc_size
        fc2_size = self.fc_size * 10
        fc2_bias_size = 10
        return conv1_size+self.out_channel+conv2_size+self.out_channel+fc1_size+fc1_bias_size+fc2_size+fc2_bias_size
    def grad_CNN(self,x,y,weight,data_N,coef=1.,sigma=22,flag_retain=False):
        '''
        This compute the necessary statistics for sampling and optimizer
        :param x: Image with size seen before
        :param y: label: should be one-hot with size :math:`N x 10`
        :param weight: weight tensor with size seen before
        :param data_N: The total number of training data
        :param coef: This is the correction term for dimensionality mismatch between training and testing task
        :param sigma: Variance for prior
        :param flag_retain: Whether to retain the graph
        :return: Gradient, modified Gradient, energy, modified energy
        '''
        sample_size=weight.shape[0]
        y_= torch.unsqueeze(y, dim=0).repeat(sample_size, 1, 1)
        prob_out=data_N*torch.mean(torch.sum(y_*self.forward(x,weight),dim=2),dim=1,keepdim=True) # num_CNN x 1
        prior_prob=torch.sum(float(-0.5*np.log(2.*3.1415926*(sigma**2)))-(weight**2)/(2*(sigma**2)),dim=1,keepdim=True)#num_chain x 1
        # Compute the gradient
        dtheta_data = grad(prob_out, weight, torch.ones(prob_out.data.shape), allow_unused=False, retain_graph=flag_retain)[0]
        dtheta_prior = -weight / ((sigma ** 2))

        # loss = nn.CrossEntropyLoss()
        # loss_output = loss(torch.squeeze(self.forward(x,weight)), y_orig)
        # loss_output.backward(retain_graph=True)

        G=-(dtheta_data+dtheta_prior)
        G_M=-(dtheta_data+coef*dtheta_prior)
        E=prob_out+prior_prob
        E_M=prob_out+coef*prior_prob
        return G,G_M,E,E_M
    def predict(self,x,weight):
        '''
        The is to give the prediction based on the averaged model
        :param x: Image
        :param weight: weight tensor
        :return: the log probability of the averaged model
        '''
        sample_size=weight.shape[0]
        out_log = logsumexp(self.forward(x, weight), dim=0) - float(np.log(sample_size))
        return out_log


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


class MLP(nn.Module):
    '''
    NN for Q matrix
    '''
    def __init__(self, input_dim, hidden, out_size=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.out_func = nn.Linear(hidden, out_size)
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_size)
        )

    def forward(self, x):
        # out=1+1./(1+torch.exp(10*self.features(x)))
        # out=1+torch.log((1+0.01*torch.abs(self.features(x))))
        out = self.features(x)
        return out


class Positive_MLP(nn.Module):
    '''
    NN for D matrix
    '''
    def __init__(self, input_dim, hidden, out_size=1):
        super(Positive_MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.out_func = nn.Linear(hidden, out_size)
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_size),
        )

    def forward(self, x):
        out = torch.abs(self.features(x))
        return out



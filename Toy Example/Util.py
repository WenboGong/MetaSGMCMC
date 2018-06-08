import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad


def draw_sample(state_list,burn_in=300,interval=10):
    ### Burning for 300 points and interval points for 10 ###
    valid_list=state_list[burn_in:]
    sample_list=[]
    counter=1
    for i in valid_list:
        if counter%interval==0:
            sample_list.append(i)
        counter+=1
    return sample_list

def parallel_draw_sample(state_list,end,burn_in=0,interval=1,draw_method='Cross',chain_ind=0):
    '''
    Draw Method: 'Cross', 'Within', 'Single'
        Cross: output list obj, each element is the samples across chain with specified burn-in, thinning (interval)
        
        Within: output list obj. Each element is the samples drawn from a single chain with specified burn-in, thinning.
        
        Single: Samples drawn from single chain specifified by chain_ind
        
    '''
    assert draw_method=='Cross' or draw_method=='Within' or draw_method=='Single',' Wrong Draw method.'
    
    if draw_method=='Cross':
        state_list=state_list[burn_in:end:interval]
        sample_list=[]
        for p in state_list:
            sample_list.append(p.t())
    if draw_method=='Within':
        sample_list=[]
        state_list=state_list[burn_in:end:interval]
        
        all_sample=torch.stack(tuple(state_list),dim=0)
        all_sample_list=list(torch.split(all_sample,1,dim=2))
        for p in all_sample_list:
            sample_list.append(torch.squeeze(p))
    if draw_method=='Single':
        state_list=state_list[burn_in:end:interval]
        all_sample=torch.squeeze(torch.stack(tuple(state_list),dim=0)[:,:,chain_ind])
        sample_list=all_sample
    return sample_list

def Likelihood_func(state_list,U):
    #### Cumulated energy ###
    counter=0
    for i in state_list:
        if counter==0:
            accumu_loss=U.forward(i)
        else:
            accumu_loss+=U.forward(i)
        counter+=1
    return accumu_loss

class U_Gaussian:
    def __init__(self,mu,var):
        self.mu=mu
        self.var=var
        self.chol=Variable(torch.potrf(var.data))
        self.dim=int(mu.data.shape[0])
    def forward(self,state_pos,transp=True):
        if transp==True:
            #Tracer()()
            state_pos=state_pos.t()
            
        const=-0.5*np.log(2.*np.pi*(self.chol.data.diag().prod())**2).cpu().data.numpy()
        #Tracer()()
        e=-0.5*torch.sum((state_pos-self.mu)*torch.t(torch.matmul(torch.inverse(self.var),torch.t(state_pos-self.mu))),1)
        prob=e+const
        return -prob
def contour_plot(U,low=2,high=4,num=100):
    x = np.linspace(low, high, num)
    y = np.linspace(low, high, num)
    X, Y = np.meshgrid(x, y)
    X_torch=torch.from_numpy(X).float()
    Y_torch=torch.from_numpy(Y).float()
    X_torch=X_torch.view(-1,1)
    Y_torch=Y_torch.view(-1,1)
    
    XY=Variable(torch.cat((X_torch,Y_torch),dim=1)).cuda()
    #Tracer()()
    log_prob=torch.exp(-(U.forward(XY,transp=False)))
    #Tracer()()
    log_prob=log_prob.view(num,num).cpu().data.numpy()
    return X,Y,log_prob
def convert_samples(state_list):

    counter=0
    for i in state_list:
        if counter==0:
            state_convert=i.t().cpu().data.numpy()
        else:
            state_convert=np.concatenate((state_convert,i.t().cpu().data.numpy()),axis=0)
        counter+=1
    
    return state_convert
def convert_to_Variable(state_list,transpose=True):
    counter=0
    for i in state_list:
        if counter==0:
            #Tracer()()
            if transpose==True:
                state_convert=i.t()
            else:
                state_convert=i
        else:
            if transpose==True:
                state_convert=torch.cat((state_convert,i.t()),dim=0)
            else:
                state_convert=torch.cat((state_convert,i),dim=0)
        counter+=1
    return state_convert
def effectiveSampleSize(data, stepSize = 1) :
    """ Compute the ESS
        Args: data: np input with N x D, N is number of samples and D is dimension
        Output: ESS
    """
    samples = int(data.shape[0])

    assert len(data) > 1,"no stats for short sequences"

    maxLag = min(int(samples/3), 1000)

    gammaStat = [0,]*maxLag
      #varGammaStat = [0,]*maxLag

    varStat = 0.0;

    if type(data) != np.ndarray :
        data = np.array(data)

    normalizedData = data - np.mean(data,axis=0)

    for lag in range(maxLag) :
        v1 = normalizedData[:samples-lag]
        v2 = normalizedData[lag:]
        v = v1 * v2
        gammaStat[lag] = np.sum(v) / v.shape[0]
        #varGammaStat[lag] = sum(v*v) / len(v)
        #varGammaStat[lag] -= gammaStat[0] ** 2

        # print lag, gammaStat[lag], varGammaStat[lag]

        if lag == 0 :
            varStat = gammaStat[0]
        elif lag % 2 == 0 :
            s = gammaStat[lag-1] + gammaStat[lag]
            if s > 0 :
                varStat += 2.0*s
            else :
                break

      # standard error of mean
      # stdErrorOfMean = Math.sqrt(varStat/samples);
      # auto correlation time
    act = stepSize * varStat / gammaStat[0]

      # effective sample size
    ess = (stepSize * samples) / act

    return ess
def Gaussian_KL(mu1,mu2,var1,var2):
    #### Compute KL Divergence between two Gaussians
    det1=np.linalg.det(var1)
    det2=np.linalg.det(var2)
    inv2=np.linalg.inv(var2)
    d=int(var1.shape[0])
    KLD=0.5*(np.log(det2/det1)-d+np.trace(var1.dot(inv2))+np.dot((mu2-mu1).transpose(),np.dot(inv2,(mu2-mu1))))
    return KLD
def generate_Cov_diag(dim,scale=1.):
    ### Generated cov matrix for Gaussian distribution.
    cov=Variable(scale*torch.diag(torch.rand(dim)))
    return cov



def generate_Cov(dim,scale=1.):
    L=scale*torch.randn(dim,dim)
    #cov=Variable(scale*torch.diag(torch.rand(dim)))
    cov=Variable(L.matmul(L.t()))
    return cov
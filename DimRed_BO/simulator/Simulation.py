import os,sys
import numpy as np
import math
from pyDOE import *

""" Must have for other portions of code:
            - output_converter(y): a function that consolidates multiple y-outputs into a single y for GPs, such as MSE
            - Evaluate_y(xx): the means to evaluate a test set
            - create_dataset(num_samples,orig_range,save_filename=''): user's choice of how to create training sets
            - Setup(home_dir): function for if things must be done in order to run Evaluate_y
 """

def output_converter(y):
    out=np.mean(y,axis=-1)
    if len(out.shape)==1:
        out=out[:,None]
    return out,y

#if stochastic, ensure e
def Evaluate_y(xx):
    y=mean_y(xx)+error_y(xx)
    if len(y.shape)==1:
        y=y[:,None]
    return y


def create_dataset(num_samples,orig_range,save_filename=''):
    samples=lhs(orig_range.shape[0],samples=num_samples)
    #adjust unit hypercube to our dimension space
    dim_min=orig_range[:,0]
    dim_max=orig_range[:,1]
    x=samples*(dim_max-dim_min)+dim_min
    
    #create y
    y=Evaluate_y(x)
    
    if len(save_filename)>1:
        np.savetxt(save_filename,np.hstack((x,y)))
    return x,y

#if stochastic, ensure e
def mean_y(xx):
    if len(xx.shape)==1:
        xx=xx[None,:]
    return .2*np.sum(xx[:,:8],axis=1)

#if stochastic, ensure e
def error_y(xx):
    return 0.05 * np.random.randn(xx.shape[0])

#if stochastic, we need to generate the new estimate + the true to compare exactly
def Evaluate_yhhat(x,xhat):
    y=mean_y(x)
    yhhat=mean_y(xhat)
    e=error_y(x)
    return y+e,yhhat+e

def Setup(home_dir):
    return



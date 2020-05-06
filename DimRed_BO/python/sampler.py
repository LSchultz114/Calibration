import numpy as np
from pyDOE import *
import os

def import_samples(res_file):
    r"""A helper function to parse the results file
    
    Args:
        res_file: the file containing the results
    
    Return:
        eval_samples (ndarray): an array with each row documenting evaluated sample
        pend_samples (ndarray): an array with each row documenting unevaluated samples
    """
    if not os.path.exists(res_file) or os.path.getsize(res_file)==0:
        raise ValueError('Need at least 1 initial data point')
    else:
        #seperate into evaluated and unevaluated samples
        #each row is a sample set in the form Y X1 X2....XN
        all_samples=np.loadtxt(res_file,dtype=np.str,delimiter=" ")
        eval_samples=all_samples[all_samples[:,0]!="P",:].astype(float)
        pend_samples=all_samples[all_samples[:,0]=="P",1:].astype(float)
            
    return eval_samples,pend_samples
    
def LHS_pool(x_range,grid_num,eval_samples,pend_samples):
    r"""A helper function to produce the list of potential candidates by
        creating a latin hypercube across the entire domain and a second
        hypercube centered around the current minimum sample
    Args:
        x_range (ndarray): the ranges for each of the dimensions used in the GP [xlb,xub]
        grid num (int): the number of samples that should be produced
        eval_samples (ndarray): an array with each row documenting evaluated sample
        pend_samples (ndarray): an array with each row documenting unevaluated samples
    
    Returns: 
        sample pool (ndarray): rows of samples for potential selection by the BO
    """
    #called out for ease of use
    dim_in=x_range.shape[0]
    dim_min=x_range[:,0]
    dim_max=x_range[:,1]

    #sample using the latin hypercube maximin method= max the min dist b/w pnts,
    # but place the pnt in a randomized location within its interval
    samples=lhs(dim_in,samples=grid_num)
    #adjust unit hypercube to our dimension space
    dim_min=x_range[:,0]
    dim_max=x_range[:,1]
    #denormalize the unit hypercube
    pool=samples*(dim_max-dim_min)+dim_min
	#encourage exploitation with a scatter of points around the current best
    if len(eval_samples)>0:
        best=np.argmin(eval_samples[:,0])
        sc_sample=lhs(dim_in,samples=50)*.0001
        scatter=sc_sample+eval_samples[best,1:]
        sample_pool=np.vstack((dim_min+((dim_max-dim_min)/2),dim_min,dim_max,scatter,pool))
        #remove any already evaluated samples from our unsample pool to avoid repetition
        for i in range(0,len(eval_samples)):
            sample_pool=np.delete(sample_pool,np.where(np.all(sample_pool==eval_samples[i,1:],axis=1)),axis=0)
    else:
        sample_pool=np.vstack((dim_min+((dim_max-dim_min)/2),dim_min,dim_max,pool))
        #remove any pending samples from our unsample pool so we don't recommend same one
    if len(pend_samples)>0:
        for i in range(0,len(pend_samples)):
            sample_pool=np.delete(sample_pool,np.where(np.all(sample_pool==pend_samples[i,:],axis=1)),axis=0)
    return sample_pool

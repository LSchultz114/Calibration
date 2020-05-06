###NOTE: THIS DOES NOT CONSIDER SAMPLING AT ACKNOWLEDGED PENDING SAMPLES ALREADY IN RESULTS FILE###
#This version will use GPy with a matern ARD kernel. This acquisition function will be EI by default
#### ASSUMES THE SETUP OF THE FILE IS [Y,X]
import numpy as np
import torch, botorch, random, gpytorch
import os,sys
import Acquisitions
import Custom_GP as cgp
import sampler
import json_do
import custom_fit


def main(res_file,num_points,grid_num,acq_type='EI',M_model=None, DR_model=None):
    r"""main function to perform bayesian optimization
        Args:
            res_file (path): where the samples are being saved
            num_points (int): number of points to recommend per iteration
            grid_num (int): number of potential candidates to be generated per iteration
            M_model (class): The educated mean model if there is one
            DR_model (class): The dimension reduction model if there is on

        Return:
            recommended samples are written to the results file with a 'P' designator for pending

        Example: 
            >>> if l<num_trials:
            >>>     BO.main(num_points,grid_num,acq_type='EI')
    """
    ######################################
    #Step 1: import samples thus far     #
    ######################################

    #file results in DR form is saved
    method=DR_model.method
    
    eval_samples,pend_samples=sampler.import_samples(res_file)
    w=sys.stderr.write("#Evaluated Samples: %d #Pending: %d\n" % (np.shape(eval_samples)[0],np.shape(pend_samples)[0]))
    if len(eval_samples)>0 and acq_type=='EI':
        w=sys.stderr.write("Current Best: %f (sample %d)\n" % (np.min(eval_samples[:,0]),np.argmin(eval_samples[:,0])+1))

    ######################################
    #Step 2: create unsampled pool       #
    ######################################
    x_range=DR_model.DR_range

    unsam_pool=sampler.LHS_pool(x_range,grid_num,eval_samples,pend_samples)

    ######################################
    #Step 3: Normalize and convert Y,X   #
    ######################################
    x_mean=eval_samples[:,1:].mean(axis=0)
    x_std=eval_samples[:,1:].std(axis=0)
    x_std[x_std==0]=1
    train_x=torch.as_tensor(np.divide((eval_samples[:,1:]-x_mean),x_std))
    
    #creating torch versions of eval_samples for clarity
    y_mean=eval_samples[:,0:1].mean(axis=0)
    y_std=eval_samples[:,0:1].std(axis=0)
    y_std[y_std==0]=1
    train_y=torch.as_tensor(np.divide((eval_samples[:,0:1]-y_mean),y_std))
    unsam_pool=np.divide((unsam_pool-x_mean),x_std)
    
    cur_min=train_y.min()
    
    ######################################
    #Step 4: Loop through batch          #
    ######################################

    # no need for gradients
    state_dict=None
    for r in range(num_points):
        #create GP prior
        gp=initialize_GP(train_x,[x_mean,x_std],train_y,[y_mean,y_std],M_model,state_dict)

        if len(eval_samples)==0:
            raise ValueError('need to have a training set')
        else:
            best_x,fake_y,unsam_pool=get_recommendation(gp,acq_type,unsam_pool,cur_min,x_range)
        
        #save dict so we can load it next time for ease
        state_dict=gp.state_dict()
        
        #update evals
        train_x = torch.cat([train_x, best_x])
        train_y=torch.cat([train_y,fake_y])

        ######################################
        #Step 5: save best sample as pending #
        ######################################
        rec=(best_x[0,:].numpy()*x_std)+x_mean
        #need to de-standardize x before recording it
        output="P " + ' '.join(map(str,rec)) + "\n"
        with open(res_file,'a+') as outfile:
            outfile.write(output)


def initialize_GP(train_x, stats_X, train_y, stats_Y, M_model=None,state_dict=None):
    r"""function to create a GP and optimize hyperparameters of 
    
    Args:
        train_x (ndarray): matrix of training samples captured by row
        train_y (ndarray): a nx1 matrix of training outputs matching to corresponding train_x sample row
        M_model (class): the educated mean model
        state_dict (dictionary): a starting point to increase processing

    Return:
        a optimized gaussian process 

    Example: 
        >>> if l<num_trials:
        >>>     initialize_GP(train_x,train_y,Mean_model)
    """

    # initialize gp
    gp = cgp.SingleTaskGP(train_x, train_y,stats_X,stats_Y,mean_module=M_model).to(torch.double)
    # load state dict if exists to make faster
    if state_dict is not None:
       gp.load_state_dict(state_dict)
    # define max log likelihood type
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    # applies to all (used for cuda)
    mll.to(train_x)
    mll.train()
    options={}
    options["exclude"] = [p_name for p_name,t in mll.named_parameters() if p_name.startswith('model.mean_module.model.')]
    _=custom_fit.fit_gpytorch_torch(mll, options=options)
    mll.eval()
    return gp


def get_recommendation(gp,acq_type,unsam_pool,cur_min,x_range):
    r"""Choose next sample by determining acquisition function values, pick max
    and simulate a random 'observation' by generating a random value from the posterior of the selected point
    done via botorch

        Args:
        gp (class): the pre-built gp
        acq_type (str): details which acuqisition function to apply
        unsam_pool (ndarray): list of potential candidates for recommendation
        cur_min (array): the current best
        x_range (array): the ranges of the inputs the GP is built over

    Return:
        the recommended candidate and an updated sample pool

    Example: 
        >>> x,y,new_pool=get_recommendation(gp,'EI',unsam_pool,cur_mean,x_range)
    """
    #create acquisition function
    acqf = Acquisitions.Acq(gp,cur_min,acq_type)
    #calculate the values based on acquisition function for all unsampled points
    acq_values=acqf(torch.Tensor(unsam_pool).unsqueeze(1)) #to make b x d--> b x q=1 x d
    #pull those which have the max value (in case there are more than 1)
    indices = max_elements(acq_values)
    sys.stderr.write("Pulled best from option of %s at %s\n" % (len(indices),max(acq_values.detach().numpy())))
    #if more than one max, choose a random point from the list
    best_index=random.choice(indices)     
    #capture new x   
    best_x=torch.as_tensor(unsam_pool[best_index:best_index+1])
    sys.stderr.write("Selected sample %d from the unsampled pool.\n" % best_index)
    #sample new y based on new x
    fake_y= gp.posterior(best_x,observation_noise=True).mean.detach() + torch.randn(1)*np.sqrt(gp.posterior(best_x,observation_noise=True).variance.detach())
    #delete new x from unsamp pool since we've just recommended it
    unsam_pool=np.delete(unsam_pool,best_index,axis=0)
    #return new set and the updated pool
    return best_x,fake_y,unsam_pool


def max_elements(seq):
    r"""A helper function to return list of position(s) of largest element 
    
    Args:
        seq (ndarray): values that should be compared to find the answer
    Return:
        index of all instances of the array's minimum value
    """
    max_indices = []
    max_val = seq[0]
    for i in range(0,len(seq)):
        if seq[i]==max_val:
            max_indices.append(i)
        elif seq[i]>max_val:
            max_val = seq[i]
            max_indices = [i]
    return max_indices



#cheat function
def Predict_CI(train_x,stats_X,train_y,stats_Y,test_x,M_model=None):
    r"""A helper function to return the predicted values + confidence intervals on demand

    Args:
        train_x (ndarray or tensor): normalized training values to initialize the GP over (subspace inputs if applicable)
        stats_X (list): a list containing the mean and std of the input training values used to normalize data
        train_y (ndarray or tensor): normalized training output values to initialize the GP over
        stats_y (list): a list containing the mean and std of the output training values used to normalize data
        test_x (ndarray or tensor): normalized testing values to predict for
        M_model (class): a class container holding the educated mean function

    Return:
        predicted mean and confidence interval for testing values test_x
    """
    #ensure train_x and train_y and test_x are all standardized for best results
    train_x=torch.as_tensor(train_x)
    train_y=torch.as_tensor(train_y)
    test_x=torch.as_tensor(test_x)

    gp=initialize_GP(train_x,stats_X,train_y,stats_Y,M_model)
    with torch.no_grad():
        # compute posterior
        posterior = gp.posterior(test_x,observation_noise=True)
        #mean
        mu=posterior.mean.detach().numpy()
        # covar matrix
        lower, upper = posterior.mvn.confidence_region()
    return mu,lower.detach().numpy(),upper.detach().numpy()
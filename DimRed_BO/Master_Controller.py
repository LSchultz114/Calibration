import os, sys
import numpy as np
import threading, shutil
import time
import random


home_dir=os.getcwd()                                        #user defined
data_dir = os.path.join(home_dir,'data')                    #user defined
sim_dir = os.path.join(home_dir,'simulator')                #user defined
py_dir = os.path.join(home_dir,'python')                    #user defined
exp_dir = os.path.join(home_dir,'experiments')              #user defined

directories= [home_dir, data_dir,sim_dir,py_dir,exp_dir]

sys.path.append(py_dir)
sys.path.append(sim_dir)

import BO
import Dim_Red
import Simulator_Master
import json_do
import Simulation
import Mean_Function

#how many loops do we run in total
num_trials=30                                             
#how many points do we recommend in each loop
num_points=3                                                

#########
### Original Problem Information
#########
#number of inputs
dim_in=20
#number of outputs
dim_out=1
#file containing training data sample set of structure [X,Y]
training_file=os.path.join(data_dir,"training_data.txt")    

_,orig_range=json_do.Stats_2L(data_dir)
#train_x,train_y=Simulation.create_dataset(80,orig_range,training_file)
#test_filename=os.path.join(data_dir,"test_data.txt")
#test_x,test_y=Simulation.create_dataset(200,orig_range,test_filename)

#########
### Dimension Reduction Info
#########

## Dim Red technique to use: PCA, PLS, AS, NN, None
method='None'
## Results file: where you want it stored
res_file=os.path.join(data_dir,method+"_results.dat")                                             
##dimensions reduced to
dim_DR=1                                                    


epochs=500
#learning rate when training
lr=0.01
#the counter weight placed on the outputs--will change emphasis of network to learn y over x
lam=20
#P=the penalty weight applied for boundry penalties when training
P=100
#XDR_layer=middle layers between input and dim red layer
XDR_layer=[10]
#DRY_layer=middle layers between dim red layer and y 
DRY_layer=[10]
#DRX_layer=middle layers between dim red layer and x 
DRX_layer=[10]
#seed for consistancy of results
seed_value=13



#########
### Educated Mean Info
#########
"""User has the option of providing the GP m(x), the estimated mean, based on a trained NN
"""
#do we implement an educated mean
e_mean=False
#number of times to run through data when training mean function NN
epochs_m=1000
#learning rate when training the mean function NN
lr_m=0.01
#layers between in and out of mean function NN
layers_m=[100,100,100]
#uses the same seed as DimRed NN


#########################################
# Step 3: Setup and Training            #
#########################################
method='None'
## Results file: where you want it stored
res_file=os.path.join(data_dir,method+"_results.dat")                                             

NN_var=[]
if method=='NN':
    NN_var=[dim_in,dim_out,XDR_layer,DRY_layer,DRX_layer,lr,seed_value,epochs,lam,P] #provides NN technique required info

DR_model= Dim_Red.main(method,dim_DR,directories,training_file,res_file,NN_var)

"""
if need to just load an existing model
DR_model=Dim_Red.load_DR(method,data_dir)
"""

#################################
# Step 4: Setup and Train Mean  #
#################################
if e_mean==True:
    Mean_model=Mean_Function.Mean_NN([dim_in,dim_out,layers_m,lr_m,seed_value,epochs_m],DR_model)
    Mean_model.calculate(training_file)
    Mean_Function.save_Mean(Mean_model,data_dir)
else:
    Mean_model=None


#if need to just load an existing model
#DR_model=Mean_Function.load_model(method,data_dir)


#################################
# Step 5: Main Loop             #
#################################

loops = num_trials + 1
random.seed(0)
for l in range(0,loops):
    if l>0:
        #After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
        all_samples=np.loadtxt(res_file,dtype=np.str,delimiter=" ")
        pend_samples=all_samples[all_samples[:,0]=="P",1:].astype(float)
        for row in range(0,np.shape(pend_samples)[0]):
            Simulator_Master.Run_Task(DR_model,pend_samples[row,:],directories,res_file,row)
    if l<num_trials:
        #If less then the number of trials we run, run another Bayes set
        print("running loop number %d of %d" % (l+1,num_trials))
        BO.main(res_file,num_points,2000,acq_type='SPE',M_model=Mean_model,DR_model=DR_model)
print("Review results.dat file in data directory")        


Complete: None, PCA, PLS, AS?

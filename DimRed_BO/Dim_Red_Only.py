import os,sys
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

#########################################
#STEP 1: Set up directories & import    #
#########################################

home_dir=os.getcwd()
data_dir = os.path.join(home_dir,'data')
Sim_dir = os.path.join(home_dir,'Simulator')
Py_dir = os.path.join(home_dir,'python')
exp_dir = os.path.join(home_dir,'experiments')

directories= [home_dir, data_dir,Sim_dir,Py_dir,exp_dir]

sys.path.append(Py_dir)
sys.path.append(Sim_dir)

import Dim_Red
import Simulation
import json_do


def generate_stats(test_y,yhat,y_true,yhhat,test_x,xhat,num_round):
    MSE_yhat=torch.nn.MSELoss()(torch.from_numpy(yhat).float(),torch.from_numpy(test_y).float()).item()
    MSE_yhhat=torch.nn.MSELoss()(torch.from_numpy(yhhat).float(),torch.from_numpy(y_true).float()).item()
    MSE_xhat=torch.nn.MSELoss()(torch.from_numpy(xhat).float(),torch.from_numpy(test_x).float()).item()

    return np.round(MSE_yhat,num_round),np.round(MSE_xhat,num_round),np.round(MSE_yhhat,num_round)


def pull_data(run,test_x,test_y,num_round):
    yhats=[]
    yhhats=[]
    MSEs=[]
    xhats=[]
    for i in range(0,len(run)):
        method=run[i]
        print(run[i])
        res_file=os.path.join(data_dir,method+"_results.dat")
        DR_model=Dim_Red.load_DR(method,data_dir)
        X_0 = DR_model.Encode_X(test_x)
        #add predicted y (yhat) to list of yhats
        yhats = [*yhats,DR_model.Pred_Y(X_0)[:,0:1]]
        #convert DR to x for xhat
        xhat=DR_model.Decode_X(X_0)
        xhats = [*xhats,xhat]
        #add simulated y from xhat to list of yhhats
        yhhats = [*yhhats,Simulation.Evaluate_yhhat(test_x,xhat)]
        #add MSE stats for PLS
        MSEs=[*MSEs,generate_stats(test_y,yhats[-1],yhhats[-1][0],yhhats[-1][1],test_x,xhats[-1],num_round)]
    print(MSEs)
    return yhats,xhats,yhhats,MSEs

def pull_graph_data(run,num_round,test_x,test_y):
    yhats,xhats,yhhats,MSEs=pull_data(run,test_x,test_y,num_round)
    y_range=np.c_[np.min(test_y),np.max(test_y)]

    yhat_inputs=[]
    yhhat_inputs=[]
    y_true=[]
    yhats_s=[]
    yhhats_s=[]
    y_true_s=[]
    scaled_y=(test_y-y_range[:,0])/(y_range[:,1]-y_range[:,0])

    for i in range(0,len(run)):
        yhat_inputs=[*yhat_inputs,[yhats[i], str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][0])]]
        yhhat_inputs=[*yhhat_inputs,[yhhats[i][1], str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][2])]]
        y_true=[*y_true,yhhats[i][0]]
        yhats_s=[*yhats_s,[(yhats[i]-y_range[:,0])/(y_range[:,1]-y_range[:,0]), str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][0])]]
        yhhats_s=[*yhhats_s,[(yhhats[i][1]-np.min(yhhats[i][0])/(np.max(yhhats[i][0])-np.min(yhhats[i][0]))), str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][2])]]
        y_true_s=[*y_true_s,[(yhhats[i][0]-np.min(yhhats[i][0]))/(np.max(yhhats[i][0])-np.min(yhhats[i][0]))]]

    return scaled_y,yhat_inputs,yhhat_inputs,y_true,yhats_s,yhhats_s,y_true_s


def subplot_yhhat(y_trues,yhhats,subs,t='',fn=''):
    fig=plt.figure(figsize=(11,9))
    colors = [ cm.viridis(x) for x in np.linspace(.1,.9,np.shape(yhhats)[0])]
    for i in range(0,np.shape(yhhats)[0]):
        plt.subplot(subs[0],subs[1], i+1)
        plt.plot(y_trues[i],yhhats[i][0],'o',color=colors[i],label=yhhats[i][1])
        plt.plot(y_trues[i],y_trues[i],color="black",label='True')
        plt.xlabel('Simulated $y(\hat{x})$')
        plt.ylabel('Original Simulated $y(x)$')
        plt.legend(loc='best')
    if len(t)>0:
        fig.suptitle('Simulated $y$ given Reconstructed '+t)
        fig.subplots_adjust(left=.1, bottom=.08, right=.95, top=.9, wspace=.2, hspace=.2)
    else:
        plt.tight_layout()
    if len(fn)>0:
        plt.savefig(fn)
    else:
        plt.show()
    


def subplot_yhat(test_y,yhats,subs,t='',fn=''):
    fig=plt.figure(figsize=(11,9))
    colors = [cm.viridis(x) for x in np.linspace(.1,.9,np.shape(yhats)[0])]
    for i in range(0,np.shape(yhats)[0]):
        plt.subplot(subs[0],subs[1], i+1)
        plt.plot(test_y,yhats[i][0],'o',color=colors[i],label=yhats[i][1])
        plt.plot(test_y,test_y,color="black",label='True')
        plt.xlabel('Predicted $\hat{y}$')
        plt.ylabel('Actual $y$')
        fig.tight_layout()
        #fig.subplots_adjust(left=.1, bottom=.08, right=.95, top=.9, wspace=.2, hspace=.2)
        plt.legend(loc='best')
    if len(t)>0:
        fig.suptitle('Predicted $\hat{y}$ given Original '+t)
        fig.subplots_adjust(left=.1, bottom=.08, right=.95, top=.9, wspace=.2, hspace=.2)
    else:
        plt.tight_layout()
    if len(fn)>0:
        plt.savefig(fn)
    else:
        plt.show()

def create_graphs(run,num_round,test_x,test_y):
    yhats,xhats,yhhats,MSEs=pull_data(run,test_x,test_y,num_round)
    y_range=np.c_[np.min(test_y),np.max(test_y)]

    yhat_inputs=[]
    yhhat_inputs=[]
    y_true=[]
    yhats_s=[]
    yhhats_s=[]
    y_true_s=[]
    dim_DR=1

    for i in range(0,len(run)):
        yhat_inputs=[*yhat_inputs,[yhats[i], str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][0])]]
        yhhat_inputs=[*yhhat_inputs,[yhhats[i][1], str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][2])]]
        y_true=[*y_true,yhhats[i][0]]
        yhats_s=[*yhats_s,[(yhats[i]-y_range[:,0])/(y_range[:,1]-y_range[:,0]), str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][0])]]
        yhhats_s=[*yhhats_s,[(yhhats[i][1]-np.min(yhhats[i][0])/(np.max(yhhats[i][0])-np.min(yhhats[i][0]))), str(dim_DR)+'D '+str(run[i])+' predictions, MSE=' + str(MSEs[i][2])]]
        y_true_s=[*y_true_s,[(yhhats[i][0]-np.min(yhhats[i][0]))/(np.max(yhhats[i][0])-np.min(yhhats[i][0]))]]

    return yhat_inputs,yhhat_inputs,y_true,yhats_s,yhhats_s,y_true_s
    

############################################
##############Define Data Sets##############
############################################
training_file=os.path.join(data_dir,"training_data.txt")
test_file=os.path.join(data_dir,"test_data.txt")
#_,_=Simulation.create_dataset(200,x_range,os.path.join(data_dir,"test_data.txt"))
con_file= os.path.join(data_dir,"config.json")

################################
#Step 1: read sampleset file    #
#################################
train=np.loadtxt(training_file)
test=np.loadtxt(test_file)

#################################
#Step 2: establish parameters   #
#################################  
dim_in,orig_range=json_do.Stats_2L(data_dir)
dim_out=np.shape(train)[1]-dim_in

test_x=test[:,:dim_in]
test_y=test[:,dim_in:]



########################################################
########################################################
######### No title, unsaved +scaled graphs  ############
########################################################
########################################################
""" 
if need to just load an existing model
method='AS'
DR_model=Dim_Red.load_DR(method,data_dir)
"""
run=['PLS','PCA','AS','NN']
num_round=4
yhat_inputs,yhhat_inputs,y_true,yhats_s,yhhats_s,y_true_s=create_graphs(run,num_round,test_x,test_y)

mpl.rcParams['figure.dpi'] = 60
subplot_yhat(test_y,yhat_inputs,subs=[2,2])
subplot_yhhat(y_true,yhhat_inputs,subs=[2,2])

subplot_yhat(scaled_y,yhats_s,subs=[2,2])
subplot_yhhat(y_true_s,yhhats_s,subs=[2,2])




########################################################
########################################################
########## saved, titled + scaled graphs  ##############
########################################################
########################################################
""" 
if need to just load an existing model
method='AS'
DR_model=Dim_Red.load_DR(method,data_dir)
"""
run=['PLS','PCA','AS','NN']
num_round=4
scaled_y,yhat_inputs,yhhat_inputs,y_true,yhats_s,yhhats_s,y_true_s=create_graphs(run,num_round,test_x,test_y)

title='$\hat{x}$\n$y=.2x_1 + .2x_2 + .2x_3 + .2x_4 + \mathcal{N}(0,0.05^2)$'
subplot_yhat(test_y,yhat_inputs,subs=[2,2],fn=home_dir+'\\figs\\Yhat_nt.pdf')
subplot_yhat(test_y,yhat_inputs,subs=[2,2],t=title,fn=home_dir+'\\figs\\Yhat.pdf')
subplot_yhhat(y_true,yhhat_inputs,subs=[2,2],fn=home_dir+'\\figs\\YfromReconstructedX_nt.pdf')
subplot_yhhat(y_true,yhhat_inputs,subs=[2,2],t=title,fn=home_dir+'\\figs\\YfromReconstructedX.pdf')

subplot_yhat(scaled_y,yhats_s,subs=[2,2],t=title,fn=home_dir+'\\figs\\Yhat_nt_scaled.pdf')
subplot_yhat(scaled_y,yhats_s,subs=[2,2],fn=home_dir+'\\figs\\Yhat_scaled.pdf')
subplot_yhhat(y_true_s,yhhats_s,subs=[2,2],fn=home_dir+'\\figs\\YfromReconstructedX_nt_scaled.pdf')
subplot_yhhat(y_true_s,yhhats_s,subs=[2,2],t=title,fn=home_dir+'\\figs\\YfromReconstructedX_scaled.pdf')

plt.show()


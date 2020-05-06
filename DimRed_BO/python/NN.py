import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
######################################################

def hidden_layer(in_n,out_n,activation):
    #basic NN layer structure
    return torch.nn.Sequential(
        torch.nn.Linear(in_n, out_n),
        activation
    )

class DR_Network(torch.nn.Module):
    def __init__(self,dim_DR,NN_var):
        """Neural network for a dimension reduction procedure"""
        #NN_var=[dim_in,dim_out,XDR_layer,DRY_layer,DRX_layer,lr,seed_value,epochs,P]
        super(DR_Network,self).__init__()
        self.lr=NN_var[5]
        self.activation = torch.nn.ReLU()

        #reduction bounder + new bounds for BO
        self.activation_DR=torch.nn.Tanh()
        self.DR_range=np.c_[-np.ones(dim_DR),np.ones(dim_DR)]
        
        ###Input (X) to DR###
        self.XDR_size = [NN_var[0], *NN_var[2]] #dim_in,XDR layers
        self.encode_DR = torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.XDR_size, self.XDR_size[1:])],
            torch.nn.Linear(self.XDR_size[-1], dim_DR),
            self.activation_DR
        )

        ###DR to Estimated Output (Y_hat)###
        self.DRY_size = [dim_DR,*NN_var[3]] #dim_DR,DRY layers
        self.decode_Y=torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.DRY_size, self.DRY_size[1:])],
            torch.nn.Linear(self.DRY_size[-1], NN_var[1])
        )

        ###DR to Estimated Output (X_hat)###
        self.DRX_size = [dim_DR,*NN_var[4]] #dim_DR,DRY layers
        self.decode_X=torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.DRX_size, self.DRX_size[1:])],
            torch.nn.Linear(self.DRX_size[-1], NN_var[0])
        )
    
            #define error for the DR goal
        self.error_function = Custom_Loss()
        #Adam optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), self.lr)
        self.loss_=[]

    def forward(self, x):
        dr_output=self.encode_DR(x)
        y_hat=self.decode_Y(dr_output)
        x_hat=self.decode_X(dr_output)
        return dr_output,y_hat,x_hat

    def Pred_X(self, DR_inputs):
        x_hat=self.decode_X(DR_inputs)
        return x_hat
 
    def Pred_Y(self, DR_inputs):
        y_hat=self.decode_Y(DR_inputs)
        return y_hat
 
    def Reduce_X(self, x):
        dr_output=self.encode_DR(x)
        return dr_output

class Custom_Loss(torch.nn.Module):

    def __init__(self):
        super(Custom_Loss,self).__init__()
        
    def forward(self,x_hat,y_hat,train_x,train_y,xlb,xub,lam,P=1):
        #use the MSE loss function
        #want the prediction from the original X thorugh NN to be correct
        output1=torch.nn.MSELoss(reduction="sum")(train_x,x_hat)
        #want the prediction from the estimated X thorugh NN to be correct
        output2=torch.nn.MSELoss(reduction="sum")(train_y,y_hat)*lam
        #want to ensure the estimated X is within acceptable bounds of problem
        L1=0
        L2=0
        for d in range(0,np.shape(x_hat)[1]):
            L1+= P*(torch.clamp(x_hat[:,d]-xlb[d],max=0)**2)
            L2+= P*(torch.clamp(x_hat[:,d]-xub[d],min=0)**2)
        #print('x error %s\ny error %s\nboundry error %s'%(output1, output2,torch.sum(L1)+torch.sum(L2)))
        return output1+output2+torch.sum(L1)+torch.sum(L2)
    



class Data(Dataset):
    def __init__(self,train_x,train_y):
        self.x=train_x
        self.y=train_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]



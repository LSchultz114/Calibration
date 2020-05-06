import numpy as np
import torch, gpytorch, os, pickle, math
import Simulation, NN
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)

class Ed_Mean(gpytorch.means.Mean):
    r"""The container used by a GP to work with the educated neural network mean 
    
    Example:
        >>> mean_model=Mean_NN(mean_var,DR_model)
        >>> ExactGP(train_x,train_y,mean_module=Ed_Mean(mean_module,stats_X,stats_Y,batch_shape))
    """
    def __init__(self, mean_model, stats_X,stats_Y,batch_shape=torch.Size(), **kwargs):
        batch_shape = _deprecate_kwarg_with_transform(
            kwargs, "batch_size", "batch_shape", batch_shape, lambda n: torch.Size([n])
        )
        super(Ed_Mean, self).__init__()
        self.batch_shape = batch_shape
        self.model=mean_model #this is the Mean_NN model created from training data
        self.x_mean=torch.as_tensor(stats_X[0])
        self.x_std=torch.as_tensor(stats_X[1])
        self.y_mean=torch.as_tensor(stats_Y[0])
        self.y_std=torch.as_tensor(stats_Y[1])

    def forward(self,input_set):
        #destandardize from the GP and send that to our NN model
        output=self.model((input_set*self.x_std)+self.x_mean)
        #restandardize the output for the GP
        output=(output-self.y_mean)/self.y_std
        if output.shape[:-2] == self.batch_shape:
            return output.squeeze()
        else:
            return output.expand(_mul_broadcast_shape(input_set.shape[:-1], output.shape))


def load_Mean(data_dir):
    r"""A shortcut helper function to load a previous mean model
    
    Args:
        data_dir (path): where all data is being saved
    Returns:
        previously saved mean model
    """
    model_file=os.path.join(data_dir,'Models','Mean_model.pickle')
    if not os.path.exists(model_file):
        raise ValueError("No saved model exists for the mean model")
    
    with open(model_file, "rb" ) as f:
        Mean_model=pickle.load(f)

    return Mean_model


def save_Mean(model,data_dir):
    r"""A helper function to save the mean model
    
    Args:
        model (class): the tained mean model
        data_dir (path): where all data is being saved
    """
    if not os.path.exists(os.path.join(data_dir,'Models')):
        os.mkdir(os.path.join(data_dir,'Models'))
    model_file=os.path.join(data_dir,'Models','Mean_model.pickle')
    with open(model_file, "wb+" ) as f:
        pickle.dump(model,f)
    
    return print('Mean model saved successfully')



########################################################################################################################################
########################################################################################################################################
####################################                Mean Neural Network                    #############################################
########################################################################################################################################
########################################################################################################################################


class Mean_NN(gpytorch.means.Mean):
    r"""Deep Neural Network educated mean

    This computes a deterministic approximation of the input-output relationship in the original subspace
    to provide an initial mean value for the GP
    
    Example:
        >>> mean_model=Mean_NN(mean_var,DR_model)
        >>> CustomGP(train_x,train_y,mean_module=mean_model)
    """
    def __init__(self,mean_var,DR_model):
        r"""Args:
        mean_var: the necessary variables to construct the NN: 
            dim_in: number of input dimensions
            dim_out: number of output dimensions (can be >1)
            layers: the number of nodes per layer between the inputs and outputs, ex [10,100,10]
            lr: learning rate for training of NN on training set
            seed_value: seed to set so results of training are same every time
            epochs: number of loops to do for training
        DR_model: The dimension reduction model if there is one
        """
        super(Mean_NN,self).__init__()

        #####################################################
        #        Set seed for consistent results            #
        #####################################################
        torch.random.manual_seed(mean_var[4])

        self.mean_func=M_Network(*mean_var)
        self.DR_func=DR_model
        self.mean_var=mean_var
    
    def calculate(self,training_file):
        """
        Since GP standardizes both inputs and outputs, we do that here as well
        """
        #Assumes the training data is of [X,Y]
        train=np.loadtxt(training_file)
        train_x=train[:,:self.mean_var[0]]
        train_y=train[:,self.mean_var[0]:]    

        #####################################################
        #Step 1: standardize X AND Y                        #
        #####################################################
        X_0=torch.as_tensor(self.DR_func.Scale(train_x))
        train_y=torch.as_tensor(train_y) 
        
        #####################################################
        #Step 5: Train on training data                     #
        #####################################################
        #use batching
        if len(X_0)*.1>=2:
            num_batch=min(2**int(math.log(len(X_0)*.1,2)),256)
        else:
            num_batch=len(X_0)
        dataset=NN.Data(X_0,torch.as_tensor(train_y))
        train_loader = DataLoader(dataset = dataset, batch_size = num_batch, shuffle = True)

        self.mean_func.train()
        for epochs in range(self.mean_var[-1]):
            for batch_idx, (inputs,labels) in enumerate(train_loader):
                self.mean_func.optimiser.zero_grad()
                #get estimate of y from DR
                y_hat = self.mean_func(inputs)
                #calc loss
                loss = self.mean_func.error_function(labels,y_hat)
                loss.backward()
                self.mean_func.optimiser.step()
                if (batch_idx==0 and epochs%100==0): print("loss: %s"%(loss.item()))

        self.mean_func.eval()
        return print('NN model created')

    def forward(self,input_set):
        #it's in the reduced subspace, must decode x (Note: receiving it unstandardized from GP)
        input_set=self.DR_func.Decode_X(input_set.numpy())
        #normalize for NN
        X_0=torch.as_tensor(self.DR_func.Scale(input_set))
        #run through our mean NN
        ys=self.mean_func.forward(X_0)
        #consolidate for the GP
        MSE,Er=Simulation.output_converter(ys.detach().numpy())
        return torch.Tensor(MSE)


########################################################################################################################################
########################################################################################################################################
####################################            Mean Neural Network Architecture           #############################################
########################################################################################################################################
########################################################################################################################################


def hidden_layer(in_n,out_n,activation):
    #basic NN layer structure
    return torch.nn.Sequential(
        torch.nn.Linear(in_n, out_n),
        activation
    )

    
class M_Network(torch.nn.Module):
    def __init__(self,dim_in,dim_out,layers,lr,seed_value,epochs):
        """Neural network for a dimension reduction procedure"""
        super(M_Network,self).__init__()
        self.lr=lr
        self.activation = torch.nn.ReLU()
       
        ###Input (X) to DR###
        self.n_size = [dim_in, *layers]
        self.ff = torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.n_size, self.n_size[1:])],
            torch.nn.Linear(self.n_size[-1], dim_out)
        )

        #define error for the DR goal
        self.error_function = torch.nn.MSELoss(reduction="sum")
        #Adam optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), self.lr)
        self.loss_=[]

    def forward(self, x):
        return self.ff(x)
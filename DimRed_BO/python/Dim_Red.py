from abc import ABC,abstractmethod
import os,sys,torch, dill, math
import NN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
import active_subspaces as ac
from scipy import optimize
import Simulation,json_do
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)

def main(method,dim_DR,directories,training_file,res_file,NN_var):
    r"""Main function to create dimension reduction model and information

    Args:
        method (str): the technique to be used for reduction
        dim_DR (int): number of dimensions to reduce to
        directories (list):  [home_dir, data_dir,sim_dir,py_dir,exp_dir]
        NN_var (list): the necessary variable set to construct the NN: 
                       [dim_in,dim_out,XDR_layer,DRY_layer,DRX_layer,lr,seed_value,epochs,lam,P]
        
    Returns:
        dimension reduction model (class)
    """

    ############################################
    #Step 1: Establish needed info             #
    ############################################
    data_dir=directories[1] #for ease of use
    #file results in original subspace + all y outputs saved
    res_ofile=res_file.split(os.extsep)[0]+'_orig.'+res_file.split(os.extsep)[1]
    #where we keep the model
    if not os.path.exists(os.path.join(data_dir,'Models')):
        os.mkdir(os.path.join(data_dir,'Models'))
    model_file=os.path.join(data_dir,'Models',method+'_model.pickle')
    #pulls the req input information from the configuration file
    vnames,dim_in,orig_range=json_do.Stats_3L(data_dir) 
    

    ############################################
    #Step 1: Create instance of technique model#
    ############################################

    #initialize the appropriate class based on input method
    if method=='None':
        DR_model=No_method(orig_range)
    elif method=='PCA':
        DR_model=PCA_method(dim_DR,orig_range)
    elif method=='PLS':
        DR_model=PLS_method(dim_DR,orig_range)                
    elif method=='AS':
        DR_model=AS_method(dim_DR,orig_range)
    elif method=='NN':
        DR_model=NN_method(dim_DR,orig_range,NN_var)
    else:
        raise ValueError('Method not valid')

    ############################################
    #Step 2: read in and set up training file  #
    ############################################

    #Assumes the training data is of [X,Y]
    train=np.loadtxt(training_file)
    train_x=train[:,:dim_in]
    #if dim_y>1, apply the appropriate transformation
    train_y,train_y_orig=Simulation.output_converter(train[:,dim_in:])
    
    #save these in the original results file we keep
    np.savetxt(res_ofile,np.c_[train_y_orig,train_x])

    ############################################
    #Step 3: Run the class' funct to train     #
    ############################################

    #train it
    if method=='NN':
        DR_model.calculate(train_x,train_y_orig)
    else:
        DR_model.calculate(train_x,train_y)
    #translate the training to the new trained subspace
    train_DR= DR_model.Encode_X(train_x)


    ############################################
    #Step 4: save new subspace set in GP style #
    ############################################

    #save all to the results.dat file the GP explorer uses in the data folder
    np.savetxt(res_file,np.c_[train_y,train_DR])

    ############################################
    #Step 5: save new subspace model for later #
    ############################################

    #if the user wants it saved, they will add a folder designation
    with open(model_file, "wb+" ) as f:
            dill.dump(DR_model,f)

    return DR_model



def load_DR(method,data_dir):
    model_file=os.path.join(data_dir,'Models',method+'_model.pickle')
    if not os.path.exists(model_file):
        raise ValueError("No saved model exists for %s" % method)
    
    with open(model_file, "rb" ) as f:
        DR_model=dill.load(f)

    return DR_model


def tune_DR(DR_model,directories,training_file,res_file):
    r"""tunes the DR model for 'dynamic' updating

    Args:
        DR_model (class): the existing model that needs to be retrained
        directories (list):  [home_dir, data_dir,sim_dir,py_dir,exp_dir]
        
    Returns:
        dimension reduction model (class) updated
    """

    ############################################
    #Step 1: Establish needed info             #
    ############################################
    data_dir=directories[1] #for ease of use
    #file of the current evaluations found thus far
    res_ofile=res_file.split(os.extsep)[0]+'_orig.'+res_file.split(os.extsep)[1]
    #file of the existing model architecture
    model_file=os.path.join(data_dir,'Models',DR_model.method+'_model.pickle')
    #pulls the req input information from the configuration file
    vnames,dim_in,orig_range=json_do.Stats_3L(data_dir) 

    ############################################
    #Step 2: read in and set up training file  #
    ############################################
    #load the results file in the original dimension space
    all_samples=np.loadtxt(res_ofile,dtype=np.str,delimiter=" ")
    #pull out only the ones finished being evaluated
    train=all_samples[all_samples[:,0]!="P",:].astype(float)
    #training file structure is [x,y] but output structure is [y,x]
    train_x=train[:,-dim_in:]
    #if dim_y>1, apply the appropriate transformation
    train_y,train_y_orig=Simulation.output_converter(train[:,:-dim_in])

    ############################################
    #Step 3: Run the class' funct to train     #
    ############################################

    #train it
    if DR_model.method=='NN':
        DR_model.calculate(train_x,train_y_orig)
    else:
        DR_model.calculate(train_x,train_y)
    #translate the training to the new trained subspace
    train_DR= DR_model.Encode_X(train_x)

    ############################################
    #Step 4: save new subspace set in GP style #
    ############################################

    #save all to the results.dat file the GP explorer uses in the data folder
    np.savetxt(res_file,np.c_[train_y,train_DR])

    ############################################
    #Step 5: save new subspace model for later #
    ############################################

    #if the user wants it saved, they will add a folder designation
    with open(model_file, "wb+" ) as f:
            dill.dump(DR_model,f)

    return DR_model


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

class DR_Technique(ABC):
    r"""Abstract base class for dimension reduction techniques.
        """
    def __init__(self,method,dim_DR,orig_range):
        r"""Constructor for the base class of techniques

        Args:
            dim_DR (int): number of dimensions to reduce to
            orig_range (ndarray): the [lower,upper] bounds of the original subspace
        """
        super().__init__()
        self.method=method
        self.dim_DR=dim_DR
        self.orig_range=orig_range
        self.DR_range=None
        
    @abstractmethod
    def calculate(self):
        r"""returns a dim red model
        """
        pass

    @abstractmethod
    def Decode_X(self,DR_input):
        r"""takes a reduced input and returns estimate x in original subspace
        """
        pass

    @abstractmethod
    def Encode_X(self,x_input):
        r"""takes a original-subspace input and returns the dim reduced equivalent
        """
        pass

    @abstractmethod
    def Pred_Y(self,DR_input):
        r"""takes a reduced input and returns estimate y in original subspace
        """
        pass

    def Scale(self,x_input):
        r"""takes a original input and returns the scaled version for ease
        """
        return np.divide((x_input-self.orig_range[:,0]),(self.orig_range[:,1]-self.orig_range[:,0]))

    def Descale(self,scaled_x_input):
        r"""takes a scaled original input and returns the unscaled version for ease
        """
        return scaled_x_input*(self.orig_range[:,1]-self.orig_range[:,0])+self.orig_range[:,0]

    def Enforce_Bounds(self,x_input):
        r"""takes a original-scaled input and returns the array with bounds enforced
        """
        return np.clip(x_input, self.orig_range[:,0],self.orig_range[:,1])



##########################################################################################################
##########################################################################################################
#                                    Default No Reduction Method                                         #
##########################################################################################################
##########################################################################################################

class No_method(DR_Technique):
    r"""a placeholder for when no reduction is made for ease of use

    This simply returns information provided

    Example:
        >>> DR_model=No_method(0,[0,1])
        >>> DR_model.calculate(train_x,train_y)
    """
    def __init__(self,orig_range):
        r"""Args:
            orig_range: the bounds of the original subspace
        """
        super().__init__('None',None,orig_range)

    def calculate(self,train_x,train_y):
        ###assumes train_x or train_y is not standardized
        self.Model = None
        #automatically standardizes everything for us
        self.DR_range= self.orig_range

        return print('Placeholder created')

    def Decode_X(self,DR_input):
        return DR_input

    def Encode_X(self,x_set):
        return x_set

    def Pred_Y(self,DR_set):
        raise ValueError("No Reduction Method used; cannot predict Y")


##########################################################################################################
##########################################################################################################
#                                     Partial Least Squares Method                                      #
##########################################################################################################
##########################################################################################################

class PLS_method(DR_Technique):
    r"""Partial Least Squares dimension reduced subspace

    This computes reduced subspace by:
    (1) standardizing x and y
    (1) applying 2-blocks regression PLS2 over x and y

    Example:
        >>> DR_model=PLS_method(0,[0,1])
        >>> DR_model.calculate(train_x,train_y)
    """
    def __init__(self,dim_DR,orig_range):
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace
        """

        if dim_DR!=0 and isinstance(dim_DR, int):
            super().__init__('PLS',dim_DR,orig_range)
        else:
            raise ValueError('dim_DR cannot equal 0 for PLS or is not an integer')
        
        #need to save mean and std for later encode/decode
        self.x_mean=None
        self.x_std=None
        self.y_mean=None
        self.y_std=None

    def calculate(self,train_x,train_y):
        ###assumes train_x or train_y is not standardized
        ############################################
        #Step 1: calc params for later use         #
        ############################################
        self.x_mean=train_x.mean(axis=0)
        self.x_std=train_x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        self.y_mean=train_y.mean(axis=0)
        self.y_std=train_y.std(axis=0)
        self.y_std[self.y_std == 0.0] = 1.0
        ############################################
        #Step 2: Create instance and fit PLS       #
        ############################################

        self.Model = PLSRegression(n_components=self.dim_DR)
        #automatically standardizes everything for us
        self.Model.fit(train_x,train_y)

        ############################################
        #Step 3: Determine reduced subspace bounds #
        ############################################

        DR=self.Model.transform(train_x) #will standardize for us
        self.DR_range=np.c_[np.min(DR,axis=0)*1.1,np.max(DR,axis=0)*1.1] #pad a bit
        
        return print('PLS model created')

    def Decode_X(self,DR_input):
        #####################################################
        #        Convert DR-> orig                          #
        #####################################################
        xhat_n=np.dot(DR_input,self.Model.x_rotations_.T)
        #convert back to original domain
        xhat= (xhat_n*self.x_std)+self.x_mean
        #verify and correct domain boundaries
        return self.Enforce_Bounds(xhat)

    def Encode_X(self,x_set):
        #####################################################
        #       Convert orig->DR                            #
        #####################################################
        #standardize N(0,1)
        X_0=np.divide(x_set-self.x_mean,self.x_std)
        #convert to DR
        return np.dot(X_0,self.Model.x_weights_)

    def Pred_Y(self,DR_set):
        #####################################################
        #       predict Y from DR                          #
        #####################################################
        #Y=TQ'+F
        yhat_n=np.dot(DR_set,self.Model.y_loadings_.T)
        if len(yhat_n)==1:
            yhat_n=yhat_n[:,None]
        #destandardize
        return (yhat_n*self.y_std)+self.y_mean




##########################################################################################################
##########################################################################################################
#                                   Principal Component Analysis Method                                  #
##########################################################################################################
##########################################################################################################

class PCA_method(DR_Technique):
    r"""Principal Component Analysis dimension reduced subspace

    This computes reduced subspace by:
    (1) standardizes the inputs
    (2) applies PCA 
    (3) applies a Linear Regression over the PCA subspace and Y since this is ignored by PCA

    Example:
        >>> DR_model=PCA_method(0,[0,1])
        >>> DR_model.calculate(train_x,train_y)
    """
    def __init__(self,dim_DR,orig_range):  
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace
        """
        super().__init__('PCA',dim_DR,orig_range)

        self.x_mean=None
        self.x_std=None
        self.pca_lr=None
        self.Modely=None


    def calculate(self,train_x,train_y):
        #####################################################
        #        Learn DR Conversions                       #
        #####################################################
    
        if np.size(train_y,axis=1)!=1:
            raise ValueError('Y cannot consist of more than 1 variable or is not a 2D array')
        
        #####################################################
        #Step 1: standardize X                                #
        #####################################################
        self.x_mean=train_x.mean(axis=0)
        self.x_std=train_x.std(axis=0)
        if len(self.x_std)>0:
            self.x_std[self.x_std == 0.0] = 1.0
        X_0=np.divide(train_x-self.x_mean,self.x_std)

        #####################################################
        #Step 2: Find PCA for DR info                       #
        #####################################################  
        if self.dim_DR==0: #if the user chose to interactively choose dim_DR
            self.Model=PCA().fit(X_0)
            #Plotting the Cumulative Summation of the Explained Variance to help decide
            plt.plot(np.cumsum(self.Model.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Explained Variance (%)') #for each component
            plt.show()
            #prompt choice of dim_DR
            self.dim_DR = int(input("what number of dimensions do you want?(>0):"))
                
        #fit based on choice
        self.Model=PCA(n_components=self.dim_DR)
        self.Model.fit(X_0)
        
        #####################################################
        #Step 4: Learn new Dim range                        #
        #####################################################
        DR=self.Model.transform(X_0)
        self.DR_range=np.c_[np.min(DR,axis=0)*1.1,np.max(DR,axis=0)*1.1] #pad a bit

        #####################################################
        #Step 5: Train y on Lower Dimension Set             #
        #####################################################
        self.Modely = LinearRegression()
        self.Modely.fit(DR,train_y)

        return print('PCA model created')

    def Decode_X(self,DR_input):
        #####################################################
        #       Convert DR-> orig                          #
        #####################################################
        xhat=(self.Model.inverse_transform(DR_input)*self.x_std)+self.x_mean
        return self.Enforce_Bounds(xhat)

    def Encode_X(self,x_set):
        #####################################################
        #        estimate DR domain for GPs                 #
        #####################################################
        X_0=np.divide(x_set-self.x_mean,self.x_std)
        return self.Model.transform(X_0)

    def Pred_Y(self,DR_input):
        #####################################################
        #       predict Y from DR                           #
        #####################################################
        return self.Modely.predict(DR_input)




##########################################################################################################
##########################################################################################################
#                                           Active Subspaces Method                                      #
##########################################################################################################
##########################################################################################################


class AS_method(DR_Technique):
    r"""Active Subspace dimension reduced subspace

    This computes reduced subspace by:
    (1) normalizing between [-1,1]
    (2) estimating the derivatives locally
    (3) learns the inactive and active subspaces
    (4) applies radial basis regression from reduced dimension to y 

    Example:
        >>> DR_model=AS_method(0,[0,1])
        >>> DR_model.calculate(train_x,train_y)
    """
    def __init__(self,dim_DR,orig_range):  
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace
        """
        super().__init__('AS',dim_DR,orig_range)

        self.x_mean=None
        self.x_std=None
        self.norm_range=None
        self.W1=None
        self.W2=None


    def calculate(self,train_x,train_y):
        #####################################################
        #Step 1: normalize data                             #
        #####################################################

        XX=self.Scale(train_x)
        X_0=(2*XX)-1 #AS requires [-1,1] range
        self.norm_range=np.c_[-np.ones(self.orig_range.shape[0]),np.ones(self.orig_range.shape[0])]
        

        #####################################################
        #Step 2: Find AS for DR info                        #
        #####################################################  
        if self.dim_DR==0:
            #interactive dim finder
            ss = ac.subspaces.Subspaces()
            #calculate local linear gradients           
            df8 = ac.gradients.local_linear_gradients(X_0,train_y)
            #compute active subspace with 100 bootstraps#
            ss.compute(df=df8,nboot=1000)
            self.dim_DR=np.shape(ss.W1)[1]

            ###visualize findings + adjust W1 if needed
            #plot the top 10 eigenvalues
            ac.utils.plotters.eigenvalues(ss.eigenvals[0:10],ss.e_br[0:10])
            #plot the subspace errors
            ac.utils.plotters.subspace_errors(ss.sub_br[0:10])
            #partition and plot the 1D and 2D active subspace to see which works better/concur with previous conclusion
            if np.shape(ss.W1)[1]==1:
                ss.partition(2)
                ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), train_y[:,0])
                answer = input("Do we include second dimension in active subspace (Y or N):")
                if answer=="Y":
                    self.dim_DR=2
        
        ss = ac.subspaces.Subspaces()
        #calculate local linear gradients           
        df8 = ac.gradients.local_linear_gradients(X_0,train_y)
        #compute active subspace with 100 bootstraps#
        ss.compute(df=df8,nboot=1000)
        ss.partition(self.dim_DR)
        self.W1=ss.W1
        self.W2=ss.W2

        #####################################################
        #Step 3: Learn new Dim range                        #
        #####################################################
        DR=X_0.dot(self.W1)
        self.DR_range=np.c_[np.min(DR,axis=0)*1.1,np.max(DR,axis=0)*1.1] #pad a bit

        #####################################################
        #Step 4: Train y on Lower Dimension Set             #
        #####################################################

        avmap=ac.domains.BoundedActiveVariableDomain(ss)
        ##this output is weird? example has [-1.03,1.03] but data converted shows [-1.]
        self.Modely = ac.response_surfaces.ActiveSubspaceResponseSurface(ac.domains.BoundedActiveVariableMap(avmap))
        self.Modely._train(DR, train_y)

        return print('AS model created')
        
        
    def reconstruct_x(self,DR_input):
        ######################################################
        #Step 1: Convert DR to original x dimension          #
        ######################################################
        #x=W1y + W2*z
        w1y=np.dot(DR_input,self.W1.T)
        ######################################################
        #Step 2: determine minimum z to adjust domain of X   #
        ######################################################
        #W2z lowerbound
        W2zlb=self.norm_range[:,0]-w1y
        W2zub=self.norm_range[:,1]-w1y

        def objective(z):
            return np.sum(np.square(z))

        constraints=[]
        a=np.shape(self.W2)
        for i in range(a[0]):
            def f(z, i=i):
                return np.dot(self.W2[i,],z) - W2zlb[i]
            constraints.append(f)

        for i in range(a[0]):
            def b(z, i=i):
                return W2zub[i]-np.dot(self.W2[i,],z)
            constraints.append(b)
        constraints1= [{'type': 'ineq', 'fun': cons} for cons in constraints]
        z=optimize.minimize(objective, np.zeros(a[-1]),constraints=constraints1)

        #####################################################
        #Step 3: Return estimated input conversion          #
        #####################################################
        
        return np.dot(DR_input,self.W1.T) + np.dot(self.W2,z.x)


    def Decode_X(self,DR_input):
        #####################################################
        #   Convert DR-> orig                               #
        #####################################################
        #if a batch set, we have to loop through the 2D subsets with opt
        if len(DR_input.shape)==3:
            xhat=np.empty((DR_input.shape[0],DR_input.shape[1],len(self.orig_range)))
            for i in range(0,DR_input.shape[0]):
                xhat[i,:,:]=self.reconstruct_x(DR_input[i])
        else:
            xhat=self.reconstruct_x(DR_input)
        xhat=((xhat+1)*(self.orig_range[:,1]-self.orig_range[:,0])/2) + self.orig_range[:,0]
        return self.Enforce_Bounds(xhat)

    def Encode_X(self,x_set):
        #####################################################
        #        convert x to [-1,1] & reduce               #
        #####################################################
        XX=self.Scale(x_set)
        X_0=(2*XX)-1 #AS requires [-1,1] range
        return X_0.dot(self.W1)


    def Pred_Y(self,DR_input):
        #####################################################
        #   predict y from the DR subspace input            #
        #####################################################
        return self.Modely.predict_av(DR_input)[0]


##########################################################################################################
##########################################################################################################
#                                         Neural Networks Method                                         #
##########################################################################################################
##########################################################################################################


class NN_method(DR_Technique):
    r"""Deep Neural Network dimension reduced subspace

    This computes reduced subspace by:
    (1) normalizes the inputs
    (2) applies a constructed NN which learns how to translate to the reduce dimension subspace
    while taking into account the X-Y relationship and the necessity to translate from reduced subspace
    to the original subspace

    Example:
        >>> DR_model=NN_method(0,[0,1],NN_var)
        >>> DR_model.calculate(train_x,train_y)
    """
    def __init__(self,dim_DR,orig_range,NN_var):
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace
            NN_var: the necessary variables to construct the NN: 
                dim_in: number of input dimensions
                dim_out: number of output dimensions (can be >1)
                XDR_layer: the number of nodes per layer between the inputs and the reduction layer, ex [10,100,10]
                DRY_layer: the number of nodes per layer between the reduction and the output layer, ex [10,100,10]
                DRX_layer:the number of nodes per layer between the reduction and the reconstructed input layer, ex [10,100,10]
                lr: learning rate for training of NN on training set
                seed_value: seed to set so results of training are same every time
                epochs: number of loops to do for training
                P: penalty value for exceeding the bounds of the original subspace when reconstructing the inputs
        """
        if dim_DR!=0 and isinstance(dim_DR, (int)):
            super().__init__('NN',dim_DR,orig_range)
        else:
            raise ValueError('dim_DR cannot equal 0 for PLS or is not an integer')

        self.norm_range=None
        self.seed_value=NN_var[6]
        self.NN_var=NN_var

    def calculate(self,train_x,train_y):
        ###assumes train_x or train_y is not standardized
        """
        NN works best with normalized values
        """
        dim_in=self.NN_var[0]
        #####################################################
        #Step 1: normalize X                                #
        #####################################################

        X_0=torch.as_tensor(self.Scale(train_x))
        self.norm_range=np.c_[np.zeros(dim_in),np.ones(dim_in)]

        #####################################################
        #Step 2: Set seed for consistent results            #
        #####################################################

        torch.random.manual_seed(self.seed_value)

        #####################################################
        #Step 3: create Model                               #
        #####################################################

        self.Model=NN.DR_Network(self.dim_DR,self.NN_var)

        #####################################################
        #Step 4: Train on training data                     #
        #####################################################
        #use batching
        if len(X_0)*.1>=2:
            num_batch=min(2**int(math.log(len(X_0)*.1,2)),256)
        else:
            num_batch=len(X_0)
        dataset=NN.Data(X_0,torch.as_tensor(train_y))
        train_loader = DataLoader(dataset = dataset, batch_size = num_batch, shuffle = True)

        self.Model.train()
        for epochs in range(self.NN_var[-3]):
            for batch_idx, (inputs,labels) in enumerate(train_loader):
                self.Model.optimiser.zero_grad()
                #get estimate of y from DR
                _,y_hat,x_hat = self.Model(inputs)
                #calc loss
                loss = self.Model.error_function(x_hat,y_hat,inputs,labels,self.norm_range[:,0],self.norm_range[:,1],self.NN_var[-2],self.NN_var[-1])
                #error function(x estimate, y estimate, true normalized x, true y, xlb, xub, lam, P)
                loss.backward()
                self.Model.optimiser.step()
                if (batch_idx==0 and epochs%100==0): print("loss: %s"%(loss.item()))

        self.Model.eval()
        self.DR_range=self.Model.DR_range
        #self.DR_range=np.c_[self.Model.min_val.detach().numpy()*np.ones(self.dim_DR),self.Model.max_val.detach().numpy()*np.ones(self.dim_DR)]

        return print('NN model created')

    def Decode_X(self,DR_input):
        #####################################################
        #       Convert DR-> orig                           #
        #####################################################
        DR_input=torch.as_tensor(DR_input)    
        xhat_NN=self.Model.Pred_X(DR_input)
        #convert back to orig range
        xhat=self.Descale(xhat_NN.detach().numpy())
        return self.Enforce_Bounds(xhat)


    def Encode_X(self,x_set):
        #####################################################
        #        estimate DR domain for GPs                 #
        #####################################################   
        X_0=torch.as_tensor(self.Scale(x_set))
        DR=self.Model.Reduce_X(X_0)
        return DR.detach().numpy()
    
    def Pred_Y(self,DR_input):
        #####################################################
        #       predict Y from DR                           #
        #####################################################
        DR_input=torch.as_tensor(DR_input)    
        yhat=self.Model.Pred_Y(DR_input)
        return yhat.detach().numpy()

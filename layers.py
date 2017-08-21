import numpy as np

class layer:

    def __init__(self,activation='linear',in_dim=1,out_dim=1):
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim

    #Initialize the weight matrix and the bias vector via Xavier Initialization
    def init_variables(self):
        self.W = 0.01*np.random.randn(self.in_dim,self.out_dim)/np.sqrt(self.in_dim)
        self.b = np.zeros((1,self.out_dim))

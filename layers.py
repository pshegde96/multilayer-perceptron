import numpy as np
import activations as act
from softmax import softmax

class Layer:

    def __init__(self,activation='relu',in_dim=1,out_dim=1):
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim

    #Initialize the weight matrix and the bias vector via Xavier Initialization
    def init_variables(self):
        self.W = 0.01*np.random.randn(self.in_dim,self.out_dim)/np.sqrt(self.in_dim)
        self.b = 0.01*np.ones((1,self.out_dim)) #initialize with a small +ve value so that relu neurons don't go to 0 at birth

    '''
    The operation is A = f(Z)
    Z = XW
    '''
    def forward(self,X):
        self.Z = X.dot(self.W)+self.b
        
        if self.activation == 'linear':
            self.A = self.Z
        elif self.activation == 'sigmoid':
            self.A = act.sigmoid_fn(self.Z)
        elif self.activation =='relu':
            self.A = act.relu_fn(self.Z)
        else:
            self.A = softmax(self.Z)  

        return self.A

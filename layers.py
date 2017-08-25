import numpy as np
import activations as act

class Layer:

    def __init__(self,activation='relu',in_dim=1,out_dim=1,posn='hidden'):
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.posn=posn

    #Initialize the weight matrix and the bias vector via Xavier Initialization
    def init_variables(self):
        self.W = 0.01*np.random.randn(self.in_dim,self.out_dim)/np.sqrt(self.in_dim)
        self.b = 0.01*np.ones((1,self.out_dim)) #initialize with a small +ve value so that relu neurons don't go to 0 at birth

    '''
    The operation is A = f(Z)
    Z = XW
    '''
    def forward(self,X):
        self.X = X
        self.Z = X.dot(self.W)+self.b
        
        if self.activation == 'linear':
            self.A = self.Z
        elif self.activation == 'sigmoid':
            self.A = act.sigmoid_fn(self.Z)
        else :
            self.A = act.relu_fn(self.Z)

        return self.A

    def backward(self,delta_plus,W_plus):
        
        #process the final layer differently
        if self.posn == 'final':
            delta = np.copy(delta_plus)
        
        else:
            if self.activation == 'linear':
                f_derivative = np.ones_like(self.Z)
            elif self.activation == 'sigmoid':
                f_derivative = act.sigmoid_derivative(self.Z)
            else:
                f_derivative = act.relu_derivative(self.Z)
            delta = (delta_plus.dot(W_plus.T))*f_derivative

        self.dW = self.X.T.dot(delta)
        self.db = np.ones((1,self.X.shape[0])).dot(delta)
        #return delta to calc grad for the previous layer
        return delta

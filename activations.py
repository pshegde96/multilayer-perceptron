'''
Contains all the activation functions implemented along with their derivatives
'''
import numpy as np

def sigmoid_fn(X):
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X):
    sigm = sigmoid_fn(X)
    return sigm*(1-sigm)

def relu_fn(X):
    return np.clip(X,0,None)

def relu_derivative(X):
    der = np.zeros_like(X)
    der[X>=0] = 1
    return der
        

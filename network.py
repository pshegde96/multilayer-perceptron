import numpy as np
from layers import Layer
from softmax import softmax

class Network:

    def __init__(self,layers_size,activation='relu',task='classification'):
        self.layers_size = layers_size
        self.activation = activation
        self.task = task

    def init_network(self):
        
        self.layers = []
        #initialize all layers except the last one witht the specified activation
        for l in range(len(self.layers_size)-2):
            self.layers.append(Layer(activation='relu',
                                in_dim=self.layers_size[l],
                                out_dim=self.layers_size[l+1]))
            self.layers[l].init_variables() #initialize the weights of the layer

        #Now add the final softmax layer
        self.layers.append(Layer(activation='linear',
                            in_dim=self.layers_size[-2],
                            out_dim=self.layers_size[-1],
                            posn='final'))
        self.layers[-1].init_variables() #initialize the weights of the layer

    def forward_pass(self,X):
        X_new = np.copy(X)

        for layer in self.layers:
            X_old = np.copy(X_new)
            X_new = layer.forward(X_old)

        if self.task == 'classification':
            self.Y_hat = softmax(X_new)
        #Yet to implement for regression
        else:
            pass

        return self.Y_hat 

    def backward_pass(self,Y_vec):
        
        #encode Y_vec in one-hot form
        Y = np.zeros_like(self.Y_hat)
        Y[range(self.Y_hat.shape[0]),Y_vec] = 1
        delta_plus = (self.Y_hat - Y)/self.Y_hat.shape[0] 

        #process the final layer differently:
        delta_plus = self.layers[-1].backward(delta_plus=delta_plus,W_plus=None)

        #go backwards through the layers, omitting the last layer
        for i in range(len(self.layers)-1):
            delta_plus = self.layers[-2-i].backward(delta_plus=delta_plus,W_plus=np.copy(self.layers[-1-i].W))

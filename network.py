import numpy as np
from layers import Layer

class Network:

    def __init__(self,layers_size,activation='relu'):
        self.layers_size = layers_size
        self.activation = activation

    def init_network(self):
        
        self.layers = []
        #initialize all layers except the last one witht the specified activation
        for l in range(len(self.layers_size)-2):
            self.layers.append(Layer(activation='relu',
                                in_dim=self.layers_size[l],
                                out_dim=self.layers_size[l+1]))
            self.layers[l].init_variables() #initialize the weights of the layer

        #Now add the final softmax layer
        self.layers.append(Layer(activation='softmax',
                            in_dim=self.layers_size[-2],
                            out_dim=self.layers_size[-1]))
        self.layers[-1].init_variables() #initialize the weights of the layer

    def forward_pass(self,X):
        new_x = np.copy(X)

        for layer in self.layers:
            old_x = np.copy(new_x)
            new_x = layer.forward(old_x)
            print new_x
            print '\n\n'
        return new_x

import numpy as np
import cPickle,gzip
from network import Network
from cross_entropy import cross_entropy
import copy

'''Important Parameters'''
BATCH_SIZE = 64
LAYERS_SIZE = [784,50,20,10]
LEARNING_RATE = 0.001
LR_DECAY = 0.85 #EVERY 200 ITERATIONS
LAMBDA_REG = 0.005
NO_ITER = 8000
h = 1e-5



'''Load the Data-Set'''
f = gzip.open('mnist.pkl.gz','rb')
train_set,val_set,test_set = cPickle.load(f)
f.close()

X_train =train_set[0]
Y_train = train_set[1]
X_test = test_set[0]
Y_test = test_set[1]

'''Let the training begin '''
net = Network(LAYERS_SIZE,activation='sigmoid')
net.init_network()


X_batch = X_train[0:1000]
Y_batch = Y_train[0:1000]

Y_hat = net.forward_pass(X_batch)

#Calculate Numerical Gradient
net.backward_pass(Y_batch)
diff = 0
count = 0
for k in range(len(net.layers)):
    for i in range(net.layers[k].W.shape[0]):
        for j in range(net.layers[k].W.shape[1]):
            net2 = copy.deepcopy(net)
            net2.layers[k].W[i,j] += h
            f1 = cross_entropy(net2.forward_pass(X_batch),Y_batch,one_hot='False')
            net2.layers[k].W[i,j] -= 2*h
            f2 = cross_entropy(net2.forward_pass(X_batch),Y_batch,one_hot='False')
            diff += (net.layers[k].dW[i,j] - (f1-f2)/2/h)**2
            count +=1
    print count

print diff/count

            



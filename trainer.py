import numpy as np
import cPickle,gzip
from network import Network
from cross_entropy import cross_entropy
import matplotlib.pyplot as plt
import math
import time
import argparse

'''Parse CommandLine Arguments '''
parser = argparse.ArgumentParser()
parser.add_argument('-filename',help='Name of the image') #not yet implemented
parser.add_argument('-activation',help='Activation in the Hidden Layers') 
parser.add_argument('-layers',help='Hidden Layers, pass as string with numbers separated by commas')
parser.add_argument('-no_iter',help='Number of mini-batch iterations to train',type=int)
parser.add_argument('-batch_size',help='Batch size',type=int)
parser.add_argument('-initial_lr',help='Initial Learning Rate',type=float)
parser.add_argument('-lr_decay',help='Learning Rate Decay every 200 epochs',type=float)
parser.add_argument('-lambda_reg',help='L2 norm regularization parameter',type=float)
parser.add_argument('-savemodel',help='1 to save,default 0',type=int) #not yet implemented
parser.add_argument('-modeldir',help='Specify dir to store models with / suffixed.Default:models/') #not yet implemented
args = parser.parse_args()

'''Important Parameters'''
BATCH_SIZE = 64
if args.batch_size:
    BATCH_SIZE = int(args.batch_size)
LAYERS_SIZE = [784,1000,500,250,10]
if args.layers:
    LAYERS_SIZE = map(int,args.layers.split(','))
LEARNING_RATE = 0.3
if args.initial_lr:
    LEARNING_RATE = float(args.initial_lr)
LR_DECAY = 1.0 #EVERY 200 ITERATIONS
if args.lr_decay:
    LR_DECAY = float(args.lr_decay)
LAMBDA_REG = 0.005
if args.lambda_reg:
    LAMBDA_REG = float(args.lambda_reg)
NO_ITER = 8000
if args.no_iter:
    NO_ITER = int(args.no_iter)
ACTIVATION = 'sigmoid'
if args.activation:
    ACTIVATION = str(args.activation)

'''Print the parameters so that user can verify them '''
print 'Architecture: {}'.format(LAYERS_SIZE)
print 'Batch Size: {}'.format(BATCH_SIZE)
print 'Initial Learning Rate: {}'.format(LEARNING_RATE)
print 'Learning Rate Decay every 200 iterations: {}'.format(LR_DECAY)
print 'Lambda of L2 Weight Regularization: {}'.format(LAMBDA_REG)
print 'Total Number of Iterations: {}'.format(NO_ITER)
print 'Activation in Hidden Layers: {}'.format(ACTIVATION)
print '\n Press Enter to Continue'
raw_input()



'''Load the Data-Set'''
f = gzip.open('mnist.pkl.gz','rb')
train_set,val_set,test_set = cPickle.load(f)
f.close()

X_train =train_set[0]
Y_train = train_set[1]
X_test = test_set[0]
Y_test = test_set[1]

X_mean = np.mean(X_train,axis=0)
X_train = X_train-X_mean
X_std = np.sqrt(np.mean(X_train**2,axis=0))
X_train = X_train/(X_std+1e-10)

'''Let the training begin '''
index = 0 #start from the first element
net = Network(LAYERS_SIZE,activation=ACTIVATION)
net.init_network()

loss_train = []
steps = []

#Use try block to stop the training when Ctrl-C is pressed
try:
    for step in range(NO_ITER):
        if index+BATCH_SIZE >= X_train.shape[0]:
            index = 0
            #permute the data to instill a sense of random sampling
            permute = np.random.permutation(X_train.shape[0])
            X_train = X_train[permute]
            Y_train = Y_train[permute]

        X_batch = X_train[index:index+BATCH_SIZE]
        Y_batch = Y_train[index:index+BATCH_SIZE]

        Y_hat = net.forward_pass(X_batch)

        #Record the training loss
        loss = cross_entropy(Y_hat,Y_batch,one_hot='False')
        loss_train.append(loss)
        steps.append(step)
        #if math.isnan(loss) :
        #    wait = input('Press something to continue')

        #Update parameters
        net.backward_pass(Y_batch)
        for layer in net.layers:
            layer.W -= LEARNING_RATE*(layer.dW+LAMBDA_REG*layer.W)
            layer.b -= LEARNING_RATE*layer.db

        if step%200 == 0:
            #compute test loss
            LEARNING_RATE *= LR_DECAY
            Y_hat_test = net.forward_pass(X_test)
            loss_test = cross_entropy(Y_hat_test,Y_test,one_hot='False')
            print 'STEP: {} \t BATCH LOSS: {} \t TEST LOSS: {}'.format(step,loss,loss_test)

        index += BATCH_SIZE

#If Ctrl-C is pressed, exit the training
except KeyboardInterrupt:
    print '\n'

plt.plot(steps,loss_train)
plt.show()

X_test = (X_test-X_mean)/(X_std+1e-7)
p_test = net.forward_pass(X_test)
Y_test_hat = np.zeros_like(p_test)
Y_test_onehot = np.zeros_like(p_test)
for i in range(len(Y_test)):
    Y_test_hat[i,np.argmax(p_test[i])]=1
    Y_test_onehot[i,Y_test[i]] =1

print np.sum(Y_test_hat*Y_test_onehot)/Y_test.shape[0]

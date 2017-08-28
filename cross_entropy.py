import numpy as np

def cross_entropy(Y_hat,y,one_hot='True'):

    if one_hot == 'False':
        Y = np.zeros_like(Y_hat)
        Y[range(y.shape[0]),y] = 1
    else:
        Y=y

    inter = Y*np.log(Y_hat+1e-7)
    cross_entropy = -1/Y.shape[0]*(np.sum(inter))
    return cross_entropy

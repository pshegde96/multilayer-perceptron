import numpy as np

'''
Computes the softmax of a matrix considering the rows as input variables
'''
def softmax(x,tmp=1):
    big = np.max(x,axis=1)
    x = x-big.reshape(-1,1)
    print x
    exp = np.exp(x*tmp)
    print exp
    return exp/(np.sum(exp,axis=1)).reshape(-1,1)

if __name__ == "__main__":
    main()

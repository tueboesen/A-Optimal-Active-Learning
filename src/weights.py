import numpy as np

def find_weights_from_labels(y,idxs,include_class_balance=True):
    '''
    This function finds the training weights
    The training weights can include a lot of different components
    Entropy (how certain we are of the label being correct.)
    Class balance (classes with few examples get higher weights, but it isn't just that. The weight is based on the total amount of weight in a class.)
    Class_balance makes sure that the total weight in each class is equal.
    The weight vector is made such that it will sum to n, (hence each point will have an average weight of 1).

    :param y: label probabilities
    :param idxs: known labels
    :param include_class_balance: Makes sure each class has the same total amount of weight distributed among their datapoints
    :return: w: weights for the training
    '''
    #First calculate the entropy weight.
    n,nc = y.shape
    labels = np.argmax(y,axis=1)
    w = np.zeros(n)
    # ysort = np.sort(y,axis=1)
    # w_entropy = ysort[:,-1] - ysort[:,-2]
    w_entropy = y[np.arange(y.shape[0]),labels]
    w_entropy[idxs] = 1 #All the known labels we set to 1.

    #Now lets balance that weight among the classes
    if include_class_balance:
        #Find the total weight in a class:
        w_sum = np.zeros(nc)
        for i in range(nc):
            w_sum[i] = np.sum(w_entropy[labels == i])
            w[labels == i] = (n/nc) * w_entropy[labels == i] / w_sum[i]
    else:
        w = w_entropy / np.sum(w_entropy) * n
    return w


def create_cardinal_Weight(U):
    #This one should not just balance the weight according to the cardinal number, but to the entropy weights.
    #Hence it should come after
    nx, nc = U.shape
    dt = U.dtype
    e1 = np.ones((nx, 1), dtype=dt)
    weights = nx / nc * U @ (U.T @ e1)
    return weights

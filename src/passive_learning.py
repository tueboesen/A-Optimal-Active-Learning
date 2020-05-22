import random

import numpy as np


def run_passive_learning(idx_labels,y,nlabels_pr_class,class_balance=False):
    n, nc = y.shape
    labels = np.argmax(y,axis=1)
    if class_balance:
        labels_unique = np.asarray(range(nc))
        for i, label in enumerate(labels_unique):
            indices = np.where(labels == label)[0]
            indices = list(set(indices).difference(set(idx_labels)))
            assert len(indices) >= nlabels_pr_class, "There were not enough datapoints of class {} left. Needed {} indices, but there are only {} available. Try increasing the dataset.".format(i, nlabels_pr_class, len(indices))
            np.random.shuffle(indices)
            idx_labels += indices[0:nlabels_pr_class]
    else:
        indices = list(set(range(n)).difference(set(idx_labels)))
        random.shuffle(indices)
        idx_labels += indices[:nlabels_pr_class * nc]
    return idx_labels

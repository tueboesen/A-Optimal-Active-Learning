import random

import numpy as np


def run_passive_learning(idx_labels: list, y: np.ndarray, nlabels_pr_class: int, class_balance: bool = False):
    """
    Opposed to active learning, this routine selects new datapoints to label randomly or randomly but balanced (equal number from each class).
    :param idx_labels: idx of already known labels
    :param y: true label pseudo-probabilities (before softmax)
    :param nlabels_pr_class: number of labels to learn per class
    :param class_balance: whether to learn new labels evenly between the different classes or just draw new labels randomly
    :return: list of labels to learn
    """
    n, nc = y.shape
    labels = np.argmax(y, axis=1)
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

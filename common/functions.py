import numpy as np


def softmax(x):
    """
    Softmax function
    """
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    """
    Cross entropy loss, negative log likelihood
    """
    # we expect predicted values to be two dimensional, such as (N,V) where N is the sample size
    # and V is the vocab size, or number of classes in traditional classification problem
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        # we want labels to be one dimensional array of (N,).
        # each element specifies correct class number.
        # in the case of word2vec each element represents word ID
        # if labels and predicted values have the same size, it means labels are also represented as two dimensional data
        # and each element of labels is a one-hot vector with size V where one value is non zero and all other values are zero
        t = np.argmax(t, axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))

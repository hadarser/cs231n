import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        f = np.dot(X[i, :], W)
        f = f - np.max(f)  # To avoid numerical instability
        softmax = np.exp(f) / (np.sum(np.exp(f)))
        loss -= np.log(softmax[y[i]])
        dW[:, y[i]] -= X[i, :]
        for j in range(num_classes):
            dW[:, j] += softmax[j] * X[i, :]

    # Average and regularization
    dW = dW / num_train + 2 * reg * W
    loss = loss / num_train + reg * np.sum(W ** 2)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    f = np.dot(X, W)
    f = f - f[np.arange(num_train), np.argmax(f, axis=1)].reshape(-1, 1)  # numeric stability
    softmax_mat = np.exp(f) / (np.sum(np.exp(f), axis=1).reshape(-1, 1))  # reshape for broadcast
    softmax_yi = softmax_mat[np.arange(num_train), y]

    # Calc loss
    loss = -1 * np.sum(np.log(softmax_yi))

    # Calc gradient
    softmax_mat[np.arange(num_train), y] -= 1  # For the missing part of the gradient only for w_yi
    dW = np.dot(X.T, softmax_mat)

    # Average and regularization
    dW = dW / num_train + 2 * reg * W
    loss = loss / num_train + reg * np.sum(W ** 2)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


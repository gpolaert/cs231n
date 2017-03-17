import numpy as np


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
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    for i in range(num_train):
        f_i = X[i, :].dot(W)  # 1xC

        # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
        f_i -= np.max(f_i)

        sum_i = 0
        for j in range(num_classes):
            sum_i += np.exp(f_i[j])

        loss += - f_i[y[i]] + np.log(sum_i)

        # Compute gradient
        # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
        # Here we are computing the contribution to the inner sum for a given i.
        for j in range(num_classes):
            p = np.exp(f_i[j]) / sum_i
            dW[:, j] += (p - (j == y[i])) * X[i, :]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

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

    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################


    scores = X.dot(W)
    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    scores -= np.max(scores)

    sums = np.exp(scores).sum(axis=1)
    correct_class_scores = scores[np.arange(num_train), y]

    loss = np.sum(- correct_class_scores + np.log(sums))

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    scores_norm = np.exp(scores) / (sums[:, np.newaxis])

    # My understanding: scores_norm are [0,1]. So subtracting 1 to the correct classes makes it neg
    # A grad step is self.W -= learning_rate * grad
    # During the the training update:
    #  - Neg values will reinforce the weights (minus * minus)
    #  - Pos values will penalize the weights (minus * plus)
    scores_norm[np.arange(num_train), y] -= 1

    dW = X.T.dot(scores_norm)

    dW /= num_train
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

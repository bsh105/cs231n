import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sum_Wyi = 0.0			#sum of number that satisfies  np.transpose(W[:,j]).dot(X[i]) - np.transpose(W[:,y[i]).dot(X[i]) + 1 > 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1.0 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += np.transpose(X[i])
        sum_Wyi += 1.0
    dW[:,y[i]] += np.transpose(np.multiply(-sum_Wyi, X[i]))
		

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg *  W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  
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

  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  
  delta = 1.0
  scores = np.transpose(X.dot(W))

  correct_class_scores = np.choose(y, scores)
  
  margins = np.maximum(0, np.subtract(scores, correct_class_scores) + delta)
  
  margins = np.transpose(margins)
  row_index = np.arange(len(y))

  margins[row_index, y] = 0

  # computing the entire loss
  loss = np.sum(margins)
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass

  # computing the gradient of all the position except yi
  mask = np.transpose(margins) > 0
  mask = mask.astype(int)
  
  dW = np.transpose(mask.dot(X))
 
  # computing the gradient of yi 
  sum_Wyi = np.sum(margins > 0, axis = 1)
  X_bar = np.multiply(X, sum_Wyi[:, np.newaxis])
  
  mask_bar = np.zeros(np.transpose(margins).shape)
  mask_bar[y, row_index] = -1
  
  dW_yi = mask_bar.dot(X_bar)
  dW += np.transpose(dW_yi)

  # post processing of gradient
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

__author__ = 'harri'
__project__ = 'dds'

import theano.tensor as T


def mse(target, prediction):
    return T.mean((target-prediction)**2)

def binary_cross_entropy(target, prediction):
    return T.nnet.binary_crossentropy(prediction,target).mean()

def categorical_cross_entropy(target,prediction):
    target = T.extra_ops.to_one_hot(target[:,0],prediction.shape[1])
    return - T.sum(target*T.log(prediction), axis=1).mean()


def binary_accuracy(target, prediction, threshold=0.5):
    return T.eq(target, T.ge(prediction, threshold)).mean()

def categorical_accuracy(target,prediction):
    predictions = T.argmax(prediction, axis=1)
    return T.eq(target, predictions.dimshuffle(0,"x")).mean()



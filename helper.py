__author__ = 'harri'
__project__ = 'dds'
import os
import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import theano

def load_data(path=None, return_shared=False):
    ''' Loads the dataset
    '''
    folder="MNIST_data"

    #############
    # LOAD DATA #
    #############
    if path is None:


        # Download the MNIST dataset if it is not present
        dataset=os.path.join(folder, "mnist.pkl.gz")

        if os.path.isfile(dataset):
            pass

        else:
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)
    else:
        dataset=path


    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_set = list(train_set)
    valid_set = list(valid_set)
    test_set = list(test_set)

    def foo_dtype(x):
        return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
    def foo_cast_int(x):
        return theano.tensor.cast(x, "int32")
    def foo_reshape(x):
        return np.reshape(x, (-1,1))
    if return_shared:
        train_set[1] = foo_reshape(train_set[1])
        train_set = [foo_dtype(x) for x in train_set]
        train_set[1] = foo_cast_int(train_set[1])
        valid_set[1] = foo_reshape(valid_set[1])
        valid_set = [foo_dtype(x) for x in valid_set]
        valid_set[1] = foo_cast_int(valid_set[1])
        test_set[1] = foo_reshape(test_set[1])
        test_set = [foo_dtype(x) for x in test_set]
        test_set[1] = foo_cast_int(test_set[1])


    return {"train":train_set, "validation":valid_set, "test":test_set}

def plot_image(x, save_path = None, width=28, height=28):
    #Plots a single greyscale image vector.
    plt.imshow(x.reshape(width, height), cmap = plt.cm.Greys_r)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def test():
    #This is a test, and this comment is also a test of the slack integration.
    data = load_data(return_shared=True)
    train = data["train"]

    train_X = train[0].get_value(borrow=True)
    plot_image(train_X[1,:])
    print train[1].eval()[:5]


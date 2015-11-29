__author__ = 'harri'

import costs
import helper
import layers
import theano

#Load data
data = helper.load_data(path=None, return_shared=True)
train_X, train_y = data["train"]
val_X, val_y = data["validation"]


#Some useful variables.
batch_size, input_dim = train_X.get_value(borrow=True).shape
hidden_dim = 100
output_dim = 10
learning_rate = 0.1
mini_batch_size = 100
n_epochs = 100

#Build network

hidden_layer = layers.DenseLayer(nonlinearity=theano.tensor.nnet.relu, input_dim=input_dim,
                                 output_dim=hidden_dim, name="hidden0")

output_layer = layers.DenseLayer(nonlinearity=theano.tensor.nnet.softmax, input_dim=hidden_dim,
                                 output_dim = output_dim, name="softmax_layer")

net = layers.NeuralNetwork(layers=[hidden_layer,output_layer])

#Create symbolic variables.
index1, index2 = theano.tensor.lscalar("index1"), theano.tensor.lscalar("index2")
X_var = theano.tensor.matrix("X_var")
y_var = theano.tensor.imatrix("y_var")
y_pred = net.get_output(X_var)


#Get cost and updates.
cost =  costs.categorical_cross_entropy(target=y_var, prediction=y_pred)
params = net.get_params()

grads = [theano.grad(cost, param) for param in params]
updates = [(param, param - learning_rate*grad) for param,grad in zip(params,grads)]


#Define monitoring channels.
accuracy = costs.categorical_accuracy(target=y_var, prediction=y_pred)

#Compile functions.

train_foo = theano.function([index1,index2], updates=updates,
                            givens = {X_var:train_X[index1:index2],
                                      y_var:train_y[index1:index2]})

get_accuracy = theano.function(inputs=[index1, index2], outputs=accuracy,
                               givens = {X_var:val_X[index1:index2],
                                      y_var:val_y[index1:index2]})


for epoch in range(n_epochs):
    print get_accuracy(0,1000)
    for lower_index in range(0,batch_size,mini_batch_size):
        train_foo(lower_index, lower_index+mini_batch_size)
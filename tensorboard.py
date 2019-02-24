################################################################################################
#                                                                                              #
# Title: Tensorboard                                                                           #
# Author: Madura Senadeera                                                                     #
# Date Created: 24/02/2019                                                                     #
# Description: utilising the same code from the neural_network_training.py file this example   #
#              is to demonstrate methods that can be used to ensure the correct type of network#
#              is being created with the code.                                                 #
################################################################################################

import tensorflow as tf

# Creating the different nodes and weights to mimic network structure

Input = tf.placeholder('float', shape=[None, 2], name="Input")
Target = tf.placeholder('float', shape=[None, 1], name="Target") # target for the system training

inputBias = tf.Variable(initial_value = tf.random_normal(shape = [3], stddev = 0.4),dtype = 'float', name= "input_bias")

weights = tf.Variable(initial_value = tf.random_normal(shape = [2, 3], stddev = 0.4),dtype = 'float', name= "hidden_weights")
hiddenBias = tf.Variable(initial_value = tf.random_normal(shape = [1], stddev = 0.4),dtype = 'float', name= "hidden_bias")

outputWeights = tf.Variable(initial_value = tf.random_normal(shape = [3, 1], stddev = 0.4),dtype = 'float', name= "output_weights")

# referring to the figure: operations_network.png, the values in the hidden nodes are calculated by multiplying the
# node value by the weight of the link and placing the value in the hidden node. These values are then summated.

# calculating the values in the hidden layer nodes
hiddenLayer = tf.matmul(Input, weights) # applying TensorFlow matrix multiplication
hiddenLayer = hiddenLayer + inputBias

# need to pass layer through a non-linear filter
hiddenLayer = tf.sigmoid(hiddenLayer, name="hidden_layer_activation")


# calculating the value in the output node
output = tf.matmul(hiddenLayer, outputWeights)
output = output + hiddenBias

# passing output through non-linear filter
output = tf.sigmoid(output, name="output_layer_activation")


# applying a training method - target introduced so we train system to reach that target
# applying the squared difference error method
cost = tf.squared_difference(Target, output) # cost meaning the difference between the outcome and the target value (deviation)

# need to reduce this cost (difference in values)
cost = tf.reduce_mean(cost)

# using optimiser to reduce the cost. Target cannot be changed, so output has to be changed and thus changes the weights and bias (backpropogating)
optimizer = tf.train.AdamOptimizer().minimize(cost) # AdamOptimizer is considered one of the best ones, all variables left default,
                                                    # howeever the learning rate can be changed for the optimizer ot work faster or slower

# explicitly stating inputs and targets
inp = [[1,1],[1,0],[0,1],[0,0]]
out = [[0],[1],[1],[0]]

# can initialise a loop to make the learning process become more accurate
epochs = 4000

# therefore now we can run session to begin training
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        err, _ = sess.run([cost, optimizer], feed_dict={Input:inp, Target:out}) # err, _ is used to say that we want to know the error value, but do not want to know about the optimiser
        print(i, err)

    # after the testing is done values can be inputted to identify the values. e.g. input first and second input equal 1, output should be 0<0.5
    while True:
        inp = [[0, 0]]
        inp[0][0] = input("type first input: ")
        inp[0][1] = input("type second input: ")
        print(sess.run([output], feed_dict={Input: inp})[0][0])
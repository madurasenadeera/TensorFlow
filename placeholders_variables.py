################################################################################################
#                                                                                              #
# Title: Placeholders & Variables                                                              #
# Author: Madura Senadeera                                                                     #
# Date Created: 24/02/2019                                                                     #
# Description: demonstrating the initialisation of placeholders as well as variables           #
#                                                                                              #
################################################################################################

import tensorflow as tf


Input1 = tf.placeholder('float', shape=[None, 3], name="Input_1") #like an input of a node. Shape indicates size of inputted data.
Input2 = tf.placeholder('float', shape=[None, 3], name="Input_2") # None represents any size batch can be taken in, while the 3 represents exactly a matrix with 3 columns
x = tf.Variable(0, dtype='float') # when explicitly declaring a variable, must state it's initial value
output = tf.Variable(0, dtype='float')

# creating a flow
x = Input1 * Input2
output = x * x

# just declared the nodes of the network

sess = tf.Session()
print(sess.run(output, feed_dict={Input1: [[1,2,3]], Input2: [[4,3,2]]})) # want to run the output, but to do this require the x value,
                                                        # which means needing the Input_1 and Input_2. As they are simply
                                                        # placeholders, we must assign values to them

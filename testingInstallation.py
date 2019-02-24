################################################################################################
#                                                                                              #
# Title: Testing TensorFlow Installation                                                       #
# Author: Madura Senadeera                                                                     #
# Date Created: 20/02/2019                                                                     #
# Description: simple code to ensure all tensor flow libraries have been installed correctly   #
#                                                                                              #
################################################################################################


import tensorflow as tf


# have to always run the the classic phrase
hello = tf.constant('hello world')

# initialising a TensorFlow session to run text
# (if session not there prints out tensor values)
sess = tf.Session()

# printing
print(sess.run(hello))
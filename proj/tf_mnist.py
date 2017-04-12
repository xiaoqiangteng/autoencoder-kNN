__author__ = 'Jinyi Zhang'

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def conv2d_transpose(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, 
        output_shape=output_shape,
        strides=[1, 2, 2, 1], 
        padding='SAME')


class Autoencoder(object):

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.kernel_size = 3

        # None mans a dimension can be of any length
        self.x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(self.x, [-1, self.img_rows, self.img_cols, 1])

        # Build the encoder
        # Conv layer 1, 32 filters
        W_conv1 = weight_variable([self.kernel_size, self.kernel_size, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        print(h_conv1.get_shape().as_list())

        # Conv layer 2, 64 filters
        W_conv2 = weight_variable([self.kernel_size, self.kernel_size, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        # Store the encoded tensor
        self.z = h_conv2

        print(h_conv2.get_shape().as_list())

        # Build the decoder using the same weights
        W_conv3 = W_conv2
        b_conv3 = bias_variable([32])

        output_shape = tf.stack([tf.shape(self.x)[0], 
            tf.shape(h_conv1)[1], tf.shape(h_conv1)[2], tf.shape(h_conv1)[3]])

        h_conv3 = tf.nn.relu(conv2d_transpose(h_conv2, W_conv3, output_shape) + b_conv3)

        print(h_conv3.get_shape().as_list())

        # Layer 3
        W_conv4 = W_conv1
        b_conv4 = bias_variable([1])

        output_shape = tf.stack([tf.shape(self.x)[0], 
            tf.shape(x_image)[1], tf.shape(x_image)[2], tf.shape(x_image)[3]])

        h_conv4 = tf.nn.relu(conv2d_transpose(h_conv3, W_conv4, output_shape) + b_conv4)

        self.y = h_conv4

        print(self.y.get_shape().as_list())

        # MSE loss function
        self.loss = tf.reduce_sum(tf.square(self.y - x_image))


def cnn_nca_mnist_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)

    auto = Autoencoder()

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(auto.loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    epochs = 3
    for epoch_i in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_x, _ = mnist.train.next_batch(batch_size)

            # Normalization
            train = np.array([img - mean_img for img in batch_x])

            sess.run(optimizer, feed_dict={auto.x: train})

        print(epoch_i, sess.run(auto.loss, feed_dict={auto.x: train}))


def main():
    train_percentage = 0.01
    test_percentage = 0.01
    trial = 1

    cnn_nca_mnist_experiment(trial, train_percentage, test_percentage)
    
if __name__ == '__main__':
    main()

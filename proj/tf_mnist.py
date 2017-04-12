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

        # Conv layer 2, 64 filters
        W_conv2 = weight_variable([self.kernel_size, self.kernel_size, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        # Store the encoded tensor
        self.z = h_conv2

        # Build the decoder using the same weights
        W_conv3 = W_conv2
        b_conv3 = bias_variable([32])

        output_shape = tf.stack([tf.shape(self.x)[0], 
            tf.shape(h_conv1)[1], tf.shape(h_conv1)[2], tf.shape(h_conv1)[3]])

        h_conv3 = tf.nn.relu(conv2d_transpose(h_conv2, W_conv3, output_shape) + b_conv3)

        # Layer 3
        W_conv4 = W_conv1
        b_conv4 = bias_variable([1])

        output_shape = tf.stack([tf.shape(self.x)[0], 
            tf.shape(x_image)[1], tf.shape(x_image)[2], tf.shape(x_image)[3]])

        h_conv4 = tf.nn.relu(conv2d_transpose(h_conv3, W_conv4, output_shape) + b_conv4)

        self.y = h_conv4

        # MSE loss function
        self.loss = tf.reduce_sum(tf.square(self.y - x_image))


def cnn_nca_mnist_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    auto = Autoencoder()

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(auto.loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    epochs = 10
    for epoch_i in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_x, _ = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={auto.x: batch_x})

        print(epoch_i, sess.run(auto.loss, feed_dict={auto.x: batch_x}))

    n = 10
    test_x, _ = mnist.test.next_batch(n)

    recon = sess.run(auto.y, feed_dict={auto.x: test_x})
  
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)

        plt.imshow(recon[i].reshape(28, 28))

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('./tmp/tf_mnist.png')


def main():
    train_percentage = 0.01
    test_percentage = 0.01
    trial = 1

    cnn_nca_mnist_experiment(trial, train_percentage, test_percentage)
    
    import gc
    gc.collect()

if __name__ == '__main__':
    main()

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

        # Define the labels
        self.y = tf.placeholder(tf.int8, [None])

        # Cluster the sample
        m = tf.shape(self.x)[0]

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
        # self.encoded_x = h_conv2
        self.encoded_x = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
        print(self.encoded_x.get_shape().as_list())

        # Build the decoder using the same weights
        W_conv3 = W_conv2
        b_conv3 = bias_variable([32])

        output_shape = tf.stack([m, 
            tf.shape(h_conv1)[1], tf.shape(h_conv1)[2], tf.shape(h_conv1)[3]])

        h_conv3 = tf.nn.relu(conv2d_transpose(h_conv2, W_conv3, output_shape) + b_conv3)

        # Layer 3
        W_conv4 = W_conv1
        b_conv4 = bias_variable([1])

        output_shape = tf.stack([m, 
            tf.shape(x_image)[1], tf.shape(x_image)[2], tf.shape(x_image)[3]])

        h_conv4 = tf.nn.relu(conv2d_transpose(h_conv3, W_conv4, output_shape) + b_conv4)

        self.reconstructed_x = h_conv4

        # MSE loss function
        reconstruction_error = tf.reduce_sum(tf.square(self.reconstructed_x - x_image))

        # NCA objection function
        # dX = h_conv2[:,None] - h_conv2[None]
        # tmp = np.einsum('...i,...j->...ij', dX, dX)


        self.loss = reconstruction_error


def cnn_nca_mnist_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    train_m = int(mnist.train.num_examples * train_percentage)
    test_m = int(mnist.test.num_examples * test_percentage)

    auto = Autoencoder()

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(auto.loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 32
    epochs = 5
    for epoch_i in range(epochs):
        for batch_i in range(train_m // batch_size):
        # for batch_i in range(1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={auto.x: batch_x, auto.y: batch_y})

        print(epoch_i, sess.run(auto.loss, feed_dict={auto.x: batch_x, auto.y: batch_y}))

    n = 10
    x_test, _ = mnist.test.next_batch(n)

    reconstructed_imgs = sess.run(auto.reconstructed_x, feed_dict={auto.x: x_test})
  
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)

        plt.imshow(reconstructed_imgs[i].reshape(28, 28))

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('./tmp/tf_mnist.png')


def main():
    train_percentage = 0.1
    test_percentage = 0.1
    trial = 1

    cnn_nca_mnist_experiment(trial, train_percentage, test_percentage)
    
    import gc
    gc.collect()

if __name__ == '__main__':
    main()

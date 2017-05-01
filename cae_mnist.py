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


class ContractiveAutoencoder(object):

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

        # Conv layer 3, 

        # Store the encoded tensor
        # self.encoded_x = h_conv2
        self.encoded_x = tf.reshape(h_conv2, [-1, 7 * 7 * 64])

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

        # Regularization error
        alpha = tf.constant(0.001)

        gradient = tf.reduce_sum(tf.square(tf.gradients(self.encoded_x, self.x)[0]))

        self.loss = reconstruction_error + tf.multiply(alpha, gradient)
        


def cae_mnist_encoding(train_percentage=0.1, test_percentage=0.1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    train_m = int(mnist.train.num_examples * train_percentage)
    test_m = int(mnist.test.num_examples * test_percentage)
    validation_m = mnist.validation.num_examples

    auto = ContractiveAutoencoder()

    learning_rate = 0.001
    optimizer_loss = tf.train.AdamOptimizer(learning_rate).minimize(auto.loss)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    epochs = 5
    minimum_loss = np.inf
    for epoch_i in range(epochs):
        val_loss_list = []

        for batch_i in range(train_m // batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(optimizer_loss, feed_dict={auto.x: batch_x, auto.y: batch_y})

        for batch_i in range(validation_m // batch_size):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)            
            val_loss = sess.run([auto.loss], feed_dict={auto.x: batch_x, auto.y: batch_y})
            
            val_loss_list.append(val_loss)
            
        validation_loss = np.mean(val_loss_list)

        print(epoch_i, validation_loss)

        if validation_loss < minimum_loss:
            minimum_loss = validation_loss
            save_path = saver.save(sess, "./models/cae_mnist/model.ckpt")

    # Encode training and testing samples
    # Save the encoded tensors

    saver.restore(sess, "./models/cae_mnist/model.ckpt")

    encoding_train_imgs_path = './data/MNIST_encoding/cae_train.encoding'
    encoding_test_imgs_path = './data/MNIST_encoding/cae_test.encoding'
    train_labels_path = './data/MNIST_encoding/cae_train.labels'
    test_labels_path = './data/MNIST_encoding/cae_test.labels'

    encoded_imgs_list = []
    labels_list = []
    for batch_i in range(train_m // batch_size):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        encoded_batches = sess.run(auto.encoded_x, feed_dict={auto.x: batch_x, auto.y: batch_y})
        
        encoded_imgs_list.append(encoded_batches)
        labels_list.append(batch_y)

    for batch_i in range(validation_m // batch_size):
        batch_x, batch_y = mnist.validation.next_batch(batch_size)            
        encoded_batches = sess.run(auto.encoded_x, feed_dict={auto.x: batch_x, auto.y: batch_y})

        encoded_imgs_list.append(encoded_batches)
        labels_list.append(batch_y)

    encoded_train_imgs = np.array(encoded_imgs_list)
    m, n, d = encoded_train_imgs.shape
    encoded_train_imgs = encoded_train_imgs.reshape(m * n, d)
    print(encoded_train_imgs.shape)

    train_labels = np.array(labels_list).flatten()
    print(train_labels.shape)

    # Save the encoded imgs
    pickle.dump(encoded_train_imgs, open(encoding_train_imgs_path, 'wb'))
    pickle.dump(train_labels, open(train_labels_path, 'wb'))

    encoded_imgs_list = []
    labels_list = []
    for batch_i in range(test_m // batch_size):
        batch_x, batch_y = mnist.test.next_batch(batch_size)            
        encoded_batches = sess.run(auto.encoded_x, feed_dict={auto.x: batch_x, auto.y: batch_y})

        encoded_imgs_list.append(encoded_batches)
        labels_list.append(batch_y)

    encoded_test_imgs = np.array(encoded_imgs_list)
    m, n, d = encoded_test_imgs.shape
    encoded_test_imgs = encoded_test_imgs.reshape(m * n, d)
    print(encoded_test_imgs.shape)

    test_labels = np.array(labels_list).flatten()
    print(test_labels.shape)

    # Save the encoded imgs
    pickle.dump(encoded_test_imgs, open(encoding_test_imgs_path, 'wb'))
    pickle.dump(test_labels, open(test_labels_path, 'wb'))

    # Show 10 reconstructed images
    n = 32
    x_test, _ = mnist.test.next_batch(n)

    reconstructed_imgs = sess.run(auto.reconstructed_x, feed_dict={auto.x: x_test})

    n = 10
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

    plt.savefig('./tmp/cae_mnist.png')


def main():
    train_percentage = 1
    test_percentage = 1

    cae_mnist_encoding(train_percentage, test_percentage)
    
    import gc
    gc.collect()

if __name__ == '__main__':
    main()

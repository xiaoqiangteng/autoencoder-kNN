__author__ = 'Jinyi Zhang'

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_mnist import Autoencoder
from tf_mnist import cal_loss

def cnn_nca_mnist_pretrain(trial, train_percentage=0.1, test_percentage=0.1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    train_m = int(mnist.train.num_examples * train_percentage)
    test_m = int(mnist.test.num_examples * test_percentage)
    validation_m = mnist.validation.num_examples

    auto = Autoencoder()

    learning_rate = 0.001

    optimizer_rec_error = tf.train.AdamOptimizer(learning_rate).minimize(auto.reconstruction_error)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Pre-train step
    batch_size = 100
    epochs = 50
    rec_error = np.inf
    for epoch_i in range(epochs):
        for batch_i in range(train_m // batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer_rec_error, feed_dict={auto.x: batch_x, auto.y: batch_y})

        validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, 
                                                            mnist.validation, validation_m, batch_size)
        print(epoch_i, validation_loss, reconstruction_error, nca_obj)

        if reconstruction_error < rec_error:
            rec_error = reconstruction_error
            save_path = saver.save(sess, "./models/tf_mnist/model.ckpt")

    # Report the loss
    validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, mnist.test, test_m, batch_size)
    print("Report the test loss of the best model: ")
    print(validation_loss, reconstruction_error, nca_obj)

    # Show 10 reconstructed images
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
    train_percentage = 1
    test_percentage = 1
    trial = 1

    cnn_nca_mnist_pretrain(trial, train_percentage, test_percentage)
    
    import gc
    gc.collect()

if __name__ == '__main__':
    main()

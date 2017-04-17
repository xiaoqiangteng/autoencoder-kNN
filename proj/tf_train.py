__author__ = 'Jinyi Zhang'

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_mnist import Autoencoder
from tf_mnist import cal_loss

def cnn_nca_mnist_train(trial, train_percentage=0.1, test_percentage=0.1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    train_m = int(mnist.train.num_examples * train_percentage)
    test_m = int(mnist.test.num_examples * test_percentage)
    validation_m = mnist.validation.num_examples

    auto = Autoencoder()

    learning_rate = 0.001
    optimizer_nca_obj = tf.train.AdamOptimizer(learning_rate).minimize(auto.nca_obj)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # Restored the pre-trained model
    saver.restore(sess, "./models/tf_mnist/model.ckpt")

    # Report the loss
    validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, mnist.test, test_m, batch_size)
    print("Report the initial test loss: ")
    print(validation_loss, reconstruction_error, nca_obj)


    # Train step
    batch_size = 2000
    epochs = 50
    minimum_loss = np.inf
    for epoch_i in range(epochs):
        for batch_i in range(train_m // batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(optimizer_nca_obj, feed_dict={auto.x: batch_x, auto.y: batch_y})

        validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, 
                                                            mnist.validation, validation_m, batch_size)
        print(epoch_i, validation_loss, reconstruction_error, nca_obj)

    # Report the loss
    validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, mnist.test, test_m, batch_size)
    print("Report the test loss of the final model: ")
    print(validation_loss, reconstruction_error, nca_obj)


    # Encode training and testing samples
    # Encode the images
    encoding_train_imgs_path = './data/MNIST_encoding/tf_train.encoding'
    encoding_test_imgs_path = './data/MNIST_encoding/tf_test.encoding'
    train_labels_path = './data/MNIST_encoding/tf_train.labels'
    test_labels_path = './data/MNIST_encoding/tf_test.labels'

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

    print("Done encoding. Show some reconstructed images.")

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

    plt.savefig('./tmp/tf_mnist.png')


def main():
    train_percentage = 1
    test_percentage = 1
    trial = 1

    cnn_nca_mnist_train(trial, train_percentage, test_percentage)
    
    import gc
    gc.collect()

if __name__ == '__main__':
    main()

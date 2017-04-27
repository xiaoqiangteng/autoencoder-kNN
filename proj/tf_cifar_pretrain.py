__author__ = 'Jinyi Zhang'

import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_cifar import Autoencoder
from tf_cifar import cal_loss

from cifar import CifarPreprocess
from cifar import reshape_cifar


def cnn_nca_cifar_pretrain():
    cp = CifarPreprocess()

    batchs = [1, 2, 3, 4, 5]
    cp.load_cifar_data(batchs)

    x_train_float = cp.X_train.astype(np.float) / 255.
    x_test_float = cp.X_test.astype(np.float) / 255.

    x_train_list = [reshape_cifar(x) for x in x_train_float]
    x_test_list = [reshape_cifar(x) for x in x_test_float]
    
    m = len(x_train_list)
    spl = int(m * 0.9)

    x_train = np.array(x_train_list)[:spl]
    x_validation = np.array(x_train_list)[spl:]
    x_test = np.array(x_test_list)

    y_train = cp.y_train[:spl]
    y_validation = cp.y_train[spl:]
    y_test = cp.y_test

    auto = Autoencoder()

    learning_rate = 0.0001

    optimizer_rec_error = tf.train.AdamOptimizer(learning_rate).minimize(auto.reconstruction_error)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Pre-train step
    batch_size = 32
    epochs = 25
    rec_error = np.inf

    m = len(y_train)
    index_list = list(range(m))

    k = m // batch_size

    for epoch_i in range(epochs):
        random.shuffle(index_list)
        for i in range(k):
            sel = index_list[i * batch_size:(i + 1) * batch_size]
            batch_x = x_train[sel]
            batch_y = y_train[sel]

            sess.run(optimizer_rec_error, feed_dict={auto.x: batch_x, auto.y: batch_y})

        validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, 
                                                            x_validation, y_validation, batch_size)

        print(epoch_i, validation_loss, reconstruction_error, nca_obj)

        if reconstruction_error < rec_error:
            rec_error = reconstruction_error
            save_path = saver.save(sess, "./models/tf_cifar/model.ckpt")

    # Report the loss
    validation_loss, reconstruction_error, nca_obj = cal_loss(auto, sess, x_test, y_test, batch_size)
    print("Report the test loss of the best model: ")
    print(validation_loss, reconstruction_error, nca_obj)

    # Show 10 reconstructed images
    n = 10

    reconstructed_imgs = sess.run(auto.reconstructed_x, feed_dict={auto.x: x_test[:10]})

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)

        plt.imshow(reconstructed_imgs[i].reshape(32, 32, 3))

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('./tmp/tf_cifar.png')


if __name__ == '__main__':
    cnn_nca_cifar_pretrain()
    
    import gc
    gc.collect()


__author__ = 'Jinyi Zhang'

import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1, 2, 2, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1]):
    return tf.nn.conv2d_transpose(x, W, 
        output_shape=output_shape,
        strides=strides, 
        padding='SAME')


class Autoencoder(object):

    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.kernel_size = 5

        # None mans a dimension can be of any length
        self.x = tf.placeholder(tf.float32, [None, self.img_rows, self.img_cols, 3])

        # Define the labels
        self.y = tf.placeholder(tf.int8, [None])

        # batch size
        m = tf.shape(self.x)[0]

        # Build the encoder
        # Conv layer 1, 32 filters
        W_conv1 = weight_variable([self.kernel_size, self.kernel_size, 3, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)

        # Conv layer 2, 64 filters
        W_conv2 = weight_variable([self.kernel_size, self.kernel_size, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        # Conv layer 3, 128 filters
        W_conv3 = weight_variable([self.kernel_size, self.kernel_size, 128, 256])
        b_conv3 = bias_variable([256])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

        # Dense layer 1, 30 hidden nodes
        h_conv3_flatten = tf.reshape(h_conv3, [-1, 4 * 4 * 256])

        W_dense_1 = weight_variable([4 * 4 * 256, 256])
        b_dense_1 = bias_variable([256])
        h_dense_1 = tf.matmul(h_conv3_flatten, W_dense_1) + b_dense_1

        # Store the encoded tensor
        self.encoded_x = h_dense_1

        print(self.encoded_x.get_shape().as_list())
        print(h_dense_1.get_shape().as_list())

        # Decode dense 
        W_dense_2 = weight_variable([256, 4 * 4 * 256])
        b_dense_2 = bias_variable([4 * 4 * 256])
        h_dense_2 = tf.matmul(h_dense_1, W_dense_2) + b_dense_2

        h_dense_tensor = tf.reshape(h_dense_2, [-1, 4, 4, 256])


        # Build the decoder using the same weights
        W_conv4 = weight_variable([self.kernel_size, self.kernel_size, 128, 256])
        b_conv4 = bias_variable([128])

        output_shape = tf.stack([m, 
            tf.shape(h_conv2)[1], tf.shape(h_conv2)[2], tf.shape(h_conv2)[3]])

        h_conv4 = tf.nn.relu(conv2d_transpose(h_dense_tensor, W_conv4, output_shape) + b_conv4)

        W_conv5 = weight_variable([self.kernel_size, self.kernel_size, 64, 128])
        b_conv5 = bias_variable([64])

        output_shape = tf.stack([m, 
            tf.shape(h_conv1)[1], tf.shape(h_conv1)[2], tf.shape(h_conv1)[3]])

        h_conv5 = tf.nn.relu(conv2d_transpose(h_conv4, W_conv5, output_shape) + b_conv5)

        # Layer 3
        W_conv6 = weight_variable([self.kernel_size, self.kernel_size, 3, 64])
        b_conv6 = bias_variable([3])

        output_shape = tf.stack([m, 
            tf.shape(self.x)[1], tf.shape(self.x)[2], tf.shape(self.x)[3]])

        h_conv6 = tf.nn.sigmoid(conv2d_transpose(h_conv5, W_conv6, output_shape) + b_conv6)

        self.reconstructed_x = h_conv6
        reconstruction_error = tf.reduce_sum(tf.square(self.x - self.reconstructed_x))

        # NCA objection function
        dx = tf.subtract(self.encoded_x[:, None], self.encoded_x[None])
        masks = tf.equal(self.y[:, None], self.y[None])

        dx_square = tf.square(dx)
        softmax_tensor = tf.exp(-dx_square)
        softmax_matrix = tf.reduce_sum(softmax_tensor, 2)

        zero_diagonal = tf.zeros([m])
        softmax_zero_diagonal = tf.matrix_set_diag(softmax_matrix, zero_diagonal)
        softmax = softmax_zero_diagonal / tf.reduce_sum(softmax_zero_diagonal, 1)

        zero_matrix = tf.zeros([m, m])
        neighbor_psum = tf.where(masks, softmax, zero_matrix)

        nca_obj = tf.reduce_sum(neighbor_psum)

        fm = tf.cast(m, tf.float32)

        self.nca_obj = tf.negative(tf.div(nca_obj, fm))
        self.reconstruction_error = tf.div(reconstruction_error, fm)

        # Define the total loss
        alpha1 = tf.constant(0.99)
        alpha2 = tf.constant(0.01)
        self.loss = tf.multiply(alpha1, self.nca_obj) + tf.multiply(alpha2, self.reconstruction_error)


def cal_loss(auto, sess, X, y, batch_size):
    """
    X: validation data.
    """
    # Report the loss
    val_loss_list = []
    rec_error_list = []
    nca_obj_list = []
    
    m = len(y)
    index_list = list(range(m))
    random.shuffle(index_list)

    k = m // batch_size

    for i in range(k):
        sel = index_list[i * batch_size:(i + 1) * batch_size]
        batch_x = X[sel]
        batch_y = y[sel]

        val_loss, rec_error, nca_obj = sess.run([auto.loss, auto.reconstruction_error, auto.nca_obj], 
                                                    feed_dict={auto.x: batch_x, auto.y: batch_y})
        
        val_loss_list.append(val_loss)
        rec_error_list.append(rec_error)
        nca_obj_list.append(nca_obj)

    validation_loss = np.mean(val_loss_list)
    reconstruction_error = np.mean(rec_error_list)
    nca_objective = np.mean(nca_obj_list)

    return validation_loss, reconstruction_error, nca_obj 
    
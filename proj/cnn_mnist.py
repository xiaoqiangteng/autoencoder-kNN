
__author__ = 'Jinyi Zhang'

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint

class CNN(object):
    
    def __init__(self):
        self.batch_size = 128
        self.epochs = 5

        self.log_dir_path = './logs/cnn_mnist/'
        self.best_model_path = './models/cnn_mnist/weights.best.hdf5'
        self.tmp_image_output_path = './tmp/cnn_mnist.png'

        self.img_rows = 28
        self.img_cols = 28
        input_img = Input(shape=(self.img_rows, self.img_cols, 1))

        kernel_size = (3, 3)
        pooling_size = (2, 2)

        x = Conv2D(32, kernel_size, activation='relu', padding='same')(input_img)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(64, kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(128, kernel_size, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pooling_size, padding='same')(x)

        x = Conv2D(128, kernel_size, activation='relu', padding='same')(encoded)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(64, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(32, kernel_size, activation='relu')(x)
        x = UpSampling2D(pooling_size)(x)
        decoded = Conv2D(1, kernel_size, activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

        self.encoder = Model(input_img, encoded)
        self.autoencoder = autoencoder

        print(self.autoencoder.summary())

    def load_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = x_train.astype(np.float) / 255.
        x_test = x_test.astype(np.float) / 255.
        self.x_train = np.reshape(x_train, (len(x_train), self.img_rows, self.img_cols, 1))  # adapt this if using `channels_first` image data format
        self.x_test = np.reshape(x_test, (len(x_test), self.img_rows, self.img_cols, 1))

    def train(self):
        
        if os.path.isfile(self.best_model_path):
            self.load_weights()
            return

        tensorboard = TensorBoard(log_dir=self.log_dir_path)
        mc = ModelCheckpoint(self.best_model_path,
                            save_best_only=True)

        self.autoencoder.fit(self.x_train, self.x_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_data=(self.x_test, self.x_test),
                    callbacks=[tensorboard, mc])

    def load_weights(self):
        self.autoencoder.load_weights(self.best_model_path)

        # Re-compile
        self.autoencoder.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

    def encode(self, X):
        encoded_imgs = self.encoder.predict(X)
        
        return encoded_imgs

    def evaluate(self):
        # Load the best model
        self.load_weights()

        # Show the loss and validation acc
        loss, val_acc = self.autoencoder.evaluate(self.x_test, self.x_test, verbose=0)
        print("Loss: {}, Val_acc: {}".format(loss, val_acc))

        # Show some reconstruction results
        reconstructed_test = self.predict(self.x_test)
        self.show_samples(self.x_test, reconstructed_test)

    def predict(self, X):
        reconstructed_imgs = self.autoencoder.predict(X)

        return reconstructed_imgs

    def show_samples(self, x_test, reconstructed_imgs):
        n = 10  # how many digits we will display

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

        plt.savefig(self.tmp_image_output_path)


def main():
    cnn = CNN()

    cnn.load_mnist_data()
    cnn.train()
    cnn.evaluate()

    encoding_train = cnn.encode(cnn.x_train)
    encoding_test = cnn.encode(cnn.x_test)

    # Save the encoded tensors
    encoding_train_imgs_path = './data/MNIST_encoding/train.encoding'
    encoding_test_imgs_path = './data/MNIST_encoding/test.encoding'

    pickle.dump(encoding_train, open(encoding_train_imgs_path, 'wb'))
    pickle.dump(encoding_test, open(encoding_test_imgs_path, 'wb'))

    import gc
    gc.collect()


if __name__ == '__main__':
    main()








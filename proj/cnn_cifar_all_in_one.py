
__author__ = 'Jinyi Zhang'

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import *

from cifar import CifarPreprocess
from cifar import reshape_cifar

class CNN_CIFAR(object):
    
    def __init__(self):
        self.batch_size = 32
        self.epochs = 2

        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.num_classes = 2

        kernel_size = (5, 5)
        pooling_size = (2, 2)

        input_img = Input(shape=(self.img_rows, self.img_cols, self.channels))

        x = Conv2D(32, kernel_size, activation='relu', padding='same')(input_img)
        x = Conv2D(64, kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(128, kernel_size, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pooling_size, padding='same')(x)

        x = Conv2D(128, kernel_size, activation='relu', padding='same')(encoded)
        x = Conv2D(64, kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        decoded = Conv2D(self.channels, kernel_size, activation='sigmoid', padding='same')(x)

        # Define the optimizer
        self.optimizer = Adam(lr=1e-4)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer=self.optimizer, 
                loss='mse', 
                metrics=['accuracy'])

        self.encoder = Model(input_img, encoded)
        
    def load_cifar_data(self):
        cp = CifarPreprocess()

        batchs = [1, 2, 3, 4, 5]
        cp.load_cifar_data(batchs)

        x_train_float = cp.X_train.astype(np.float) / 255.
        x_test_float = cp.X_test.astype(np.float) / 255.

        x_train_list = [reshape_cifar(x) for x in x_train_float]
        x_test_list = [reshape_cifar(x) for x in x_test_float]
        
        self.x_train = np.array(x_train_list)
        self.x_test = np.array(x_test_list)

        print(self.x_train.shape)
        print(self.x_test.shape)

        
    def train(self):
        log_dir_path = './logs/cnn_cifar/all_in_one/'
        best_model_path = './models/cnn_cifar/weights.best.all_in_one.hdf5'
    
        if os.path.isfile(best_model_path):
            self.load_weights(best_model_path)
            return

        tensorboard = TensorBoard(log_dir=log_dir_path)
        mc = ModelCheckpoint(best_model_path,
                            save_best_only=True)

        self.autoencoder.fit(self.x_train, self.x_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_data=(self.x_test, self.x_test),
                    callbacks=[tensorboard, mc])

    def load_weights(self, best_model_path):
        self.autoencoder.load_weights(best_model_path)

        # Re-compile
        self.autoencoder.compile(optimizer=self.optimizer, 
                loss='mse', 
                metrics=['accuracy'])

    def encode(self, X):
        encoded_imgs = self.encoder.predict(X)
        
        return encoded_imgs

    def evaluate(self):
        # Show the loss and validation acc
        best_model_path = './models/cnn_cifar/weights.best.all_in_one.hdf5'
        # Load the best model
        self.load_weights(best_model_path)

        loss, val_acc = self.autoencoder.evaluate(self.x_test, self.x_test, verbose=0)
        print("Loss: {}, Val_acc: {}\n".format(loss, val_acc))

        # Show some reconstruction results
        reconstructed_test = self.predict(self.x_test)
        self.show_samples(self.x_test, reconstructed_test)

    def predict(self, X):
        reconstructed_imgs = self.autoencoder.predict(X)

        return reconstructed_imgs

    def show_samples(self, x_test, reconstructed_imgs):
        tmp_image_output_path = './tmp/cnn_cifar.all_in_one.png'

        n = 10  # how many digits we will display

        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(self.img_rows, self.img_cols, self.channels))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed_imgs[i].reshape(self.img_rows, self.img_cols, self.channels))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(tmp_image_output_path)


def main():
    cnn = CNN_CIFAR()

    cnn.load_cifar_data()

    cnn.train()
    cnn.evaluate()

    encoding_train = cnn.encode(cnn.x_train)
    encoding_test = cnn.encode(cnn.x_test)

    # Save the encoded tensors
    encoding_train_imgs_path = './data/CIFAR_encoding/train.all_in_one.encoding'
    encoding_test_imgs_path = './data/CIFAR_encoding/test.all_in_one.encoding'

    pickle.dump(encoding_train, open(encoding_train_imgs_path, 'wb'), protocol=4)
    pickle.dump(encoding_test, open(encoding_test_imgs_path, 'wb'), protocol=4)

    import gc
    gc.collect()


if __name__ == '__main__':
    main()








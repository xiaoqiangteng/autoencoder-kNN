
__author__ = 'Jinyi Zhang'

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint

class CNN(object):
    
    def __init__(self):
        self.batch_size = 128
        self.epochs = 1

        self.log_dir_path = './logs/cnn/'
        self.best_model_path = './models/cnn/weights.best.hdf5'

        self.img_rows = 28
        self.img_cols = 28
        input_img = Input(shape=(self.img_rows, self.img_cols, 1))

        kernel_size = (3, 3)
        pooling_size = (2, 2)

        x = Conv2D(16, kernel_size, activation='relu', padding='same')(input_img)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(8, kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(8, kernel_size, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pooling_size, padding='same')(x)

        x = Conv2D(8, kernel_size, activation='relu', padding='same')(encoded)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(8, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(16, kernel_size, activation='relu')(x)
        x = UpSampling2D(pooling_size)(x)
        decoded = Conv2D(1, kernel_size, activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

        self.encoder = Model(input_img, encoded)
        self.autoencoder = autoencoder

    def train(self):
        (x_train, _), (x_test, _) = mnist.load_data()
        
        x_train = x_train.astype(np.float) / 255.
        x_test = x_test.astype(np.float) / 255.
        self.x_train = np.reshape(x_train, (len(x_train), self.img_rows, self.img_cols, 1))  # adapt this if using `channels_first` image data format
        self.x_test = np.reshape(x_test, (len(x_test), self.img_rows, self.img_cols, 1))

        # tensorboard = TensorBoard(log_dir=self.log_dir_path)

        # mc = ModelCheckpoint(self.best_model_path,
        #                     save_best_only=True)

        # self.autoencoder.fit(self.x_train, self.x_train,
        #             epochs=self.epochs,
        #             batch_size=self.batch_size,
        #             shuffle=True,
        #             validation_data=(self.x_test, self.x_test),
        #             callbacks=[tensorboard, mc])

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

        plt.savefig('tmp.png')


def main():
    cnn = CNN()

    cnn.train()

    cnn.evaluate()

    import gc

    gc.collect()

if __name__ == '__main__':
    main()









__author__ = 'Jinyi Zhang'

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint

from cifar import CifarPreprocess
from cifar import reshape_cifar

class CNN_CIFAR(object):
    
    def __init__(self):
        self.batch_size = 128
        self.epochs = 2

        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.num_classes = 10

        kernel_size = (5, 5)
        pooling_size = (2, 2)

        input_img = Input(shape=(self.img_rows, self.img_cols, self.channels))

        x = Conv2D(36, kernel_size, activation='relu', padding='same')(input_img)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(64, kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(128, kernel_size, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pooling_size, padding='same')(x)

        x = Conv2D(128, kernel_size, activation='relu', padding='same')(encoded)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(64, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        decoded = Conv2D(self.channels, kernel_size, activation='sigmoid', padding='same')(x)

        self.autoencoders = []
        self.encoders = []
        # Create 10 autoencoders. 1 for a class
        for _ in range(self.num_classes):
            autoencoder = Model(input_img, decoded)
            autoencoder.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])

            encoder = Model(input_img, encoded)
            
            self.autoencoders.append(autoencoder)
            self.encoders.append(encoder)

    def load_cifar_data(self):
        cp = CifarPreprocess()

        batchs = [1, 2, 3, 4, 5]
        cp.load_cifar_data(batchs)

        x_train_float = cp.X_train.astype(np.float) / 255.
        x_test_float = cp.X_test.astype(np.float) / 255.

        x_train_list = [reshape_cifar(x) for x in x_train_float]
        x_test_list = [reshape_cifar(x) for x in x_test_float]
        
        x_train = np.array(x_train_list)
        x_test = np.array(x_test_list)

        print(x_train.shape)
        print(x_test.shape)

        # Load the labels for clustering
        y_train = cp.y_train
        y_test = cp.y_test
        
        # Need to cluster the data by classes
        self.x_train = {}
        self.x_test = {}

        for i, x in enumerate(x_train):
            label = y_train[i]

            if label not in self.x_train:
                self.x_train[label] = [x]
            else:
                self.x_train[label].append(x)

        for label in range(self.num_classes):
            self.x_train[label] = np.array(self.x_train[label])

        for i, x in enumerate(x_test):
            label = y_test[i]

            if label not in self.x_test:
                self.x_test[label] = [x]
            else:
                self.x_test[label].append(x)

        for label in range(self.num_classes):
            self.x_test[label] = np.array(self.x_test[label])

    def train(self):
        for label in range(self.num_classes):
            log_dir_path = './logs/cnn_cifar/{}/'.format(label)
            best_model_path = './models/cnn_cifar/weights.best.{}.hdf5'.format(label)
        
            if os.path.isfile(best_model_path):
                self.load_weights(label, best_model_path)
                return

            tensorboard = TensorBoard(log_dir=log_dir_path)
            mc = ModelCheckpoint(best_model_path,
                                save_best_only=True)

            self.autoencoders[label].fit(self.x_train[label], self.x_train[label],
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(self.x_test[label], self.x_test[label]),
                        callbacks=[tensorboard, mc])

    def load_weights(self, label, best_model_path):
        self.autoencoders[label].load_weights(best_model_path)

        # Re-compile
        self.autoencoders[label].compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

    def encode(self, label, X):
        encoded_imgs = self.encoders[label].predict(X)
        
        return encoded_imgs

    def evaluate(self):
        # Load the best model
        self.load_weights()

        # Show the loss and validation acc
        for label in range(self.num_classes):
            loss, val_acc = self.autoencoders[label].evaluate(self.x_test[label], self.x_test[label], verbose=0)
            print("Label: {}, Loss: {}, Val_acc: {}\n".format(label, loss, val_acc))

            # Show some reconstruction results
            reconstructed_test = self.predict(self.x_test[label])
            self.show_samples(self.x_test[label], reconstructed_test, label)

    def predict(self, label, X):
        reconstructed_imgs = self.autoencoders[label].predict(X)

        return reconstructed_imgs

    def show_samples(self, x_test, reconstructed_imgs, label):
        tmp_image_output_path = './tmp/cnn_cifar.{}.png'.format(label)

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

    for label in range(cnn.num_classes):
        encoding_train = cnn.encode(cnn.x_train[label])
        encoding_test = cnn.encode(cnn.x_test[label])

        # Save the encoded tensors
        encoding_train_imgs_path = './data/CIFAR_encoding/train.{}.encoding'.format(label)
        encoding_test_imgs_path = './data/CIFAR_encoding/test.{}.encoding'.format(label)

        pickle.dump(encoding_train, open(encoding_train_imgs_path, 'wb'))
        pickle.dump(encoding_test, open(encoding_test_imgs_path, 'wb'))

    import gc
    gc.collect()


if __name__ == '__main__':
    main()








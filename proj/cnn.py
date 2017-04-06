
__author__ = 'Jinyi Zhang'

import numpy as np

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

from keras.callbacks import TensorBoard

def main():
    (x_train, _), (x_test, _) = mnist.load_data()

    input_img = Input(shape=(28, 28, 1))

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
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    tensorboard = TensorBoard(log_dir='./logs/cnn/')

    autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard])


if __name__ == '__main__':
    main()








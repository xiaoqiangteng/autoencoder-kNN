
__author__ = 'Jinyi Zhang'

from keras.datasets import mnist
import numpy as np


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)

if __name__ == '__main__':
    main()
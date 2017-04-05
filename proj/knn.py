
__author__ = 'Jinyi Zhang'

import random

from keras.datasets import mnist
import numpy as np

class kNN(object):

    def __init__(self, k):
        self._k = k

    def load_data(self, percentage=0.1):
        (x_train, y_train), _ = mnist.load_data()
        m, w, h = x_train.shape

        self.m = int(m * percentage)

        sel = random.sample(range(m), self.m)

        self.x = x_train.reshape((m, w * h)).astype(np.float)[sel]
        self.y = y_train[sel]

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k): 
        self._k = k

    def evaluate(self, percentage=0.01):
        _, (x_test, y_test) = mnist.load_data()

        total_m, w, h = x_test.shape

        m = int(total_m * percentage)

        x_test = x_test[:m]
        y_test = y_test[:m]

        x_test = x_test.reshape((m, w * h)).astype(np.float)

        x_square = np.diag(np.dot(self.x, self.x.T))

        ed_matrix = (np.ones((len(x_test), 1)) * x_square.T) - 2 * (np.dot(x_test, self.x.T))

        label_index_array = np.argmin(ed_matrix, axis=1)

        preds = self.y[label_index_array]

        correct_number = np.count_nonzero(preds == y_test)

        accuracy = correct_number / m

        print(accuracy)

def main():
    k = 3
    knn = kNN(k)

    knn.load_data(0.5)

    knn.evaluate(1)

if __name__ == '__main__':
    main()





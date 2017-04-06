
__author__ = 'Jinyi Zhang'

import random
import pickle

from keras.datasets import mnist
import numpy as np
from scipy.stats import mode

class kNN(object):

    def __init__(self, k=1):
        self._k = k

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k): 
        self._k = k

    def load_mnist_data(self, train_percentage=0.1, test_percentage=0.1):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        m, w, h = x_train.shape
        self.train_m = int(m * train_percentage)
        sel = random.sample(range(m), self.train_m)

        self.x = x_train.reshape((m, w * h)).astype(np.float)[sel]
        self.y = y_train[sel]

        m = len(x_test)
        self.test_m = int(m * test_percentage)
        sel = random.sample(range(m), self.test_m)

        self.x_test = x_test.reshape((m, w * h)).astype(np.float)[sel]
        self.y_test = y_test[sel]


    
    def evaluate(self, X, y, X_test, y_test):
        x_square = np.diag(np.dot(X, X.T))
        ed_matrix = (np.ones((len(X_test), 1)) * x_square.T) - 2 * (np.dot(X_test, X.T))

        if self._k == 1:
            label_index_array = np.argmin(ed_matrix, axis=1)
        else:
            label_index_array = np.argpartition(ed_matrix, self._k, axis=1)[:, :self._k]

        preds = y[label_index_array]
        if self._k != 1:
            for i, p in enumerate(preds):
                if len(np.unique(p)) == len(p):
                    preds[i][-1] = preds[i][0]
            preds = mode(preds.T).mode[0]

        correct_number = np.count_nonzero(preds == y_test)

        accuracy = correct_number / len(y_test)

        return accuracy

def mnist_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    knn = kNN()
    knn.load_mnist_data(train_percentage, test_percentage)

    k_valus = [1, 3, 5, 7]
    for k in k_valus:
        knn.k = k

        acc_list = []
        for _ in range(trial):
            acc = knn.evaluate(knn.x, knn.y, knn.x_test, knn.y_test)
            acc_list.append(acc)

        print(np.mean(np.array(acc_list)))

def encoding_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    encoding_train_imgs_path = './data/MNIST/train.encoding'
    encoding_test_imgs_path = './data/MNIST/test.encoding'

    encoding_train = pickle.load(open(encoding_train_imgs_path, 'rb'))
    encoding_test = pickle.load(open(encoding_test_imgs_path, 'rb'))

    print(encoding_train.shape)

    # Load labels
    (_, y_train), (_, y_test) = mnist.load_data()

    m, w, h, l = encoding_train.shape
    train_m = int(m * train_percentage)
    sel = random.sample(range(m), train_m)
    X = encoding_train.reshape((m, w * h * l)).astype(np.float)[sel]
    y = y_train[sel]

    m = len(encoding_test)
    test_m = int(m * test_percentage)
    sel = random.sample(range(m), test_m)

    X_test = encoding_test.reshape((m, w * h * l)).astype(np.float)[sel]
    y_test = y_test[sel]

    knn = kNN()
    k_valus = [1, 3, 5, 7]
    for k in k_valus:
        knn.k = k

        acc_list = []
        for _ in range(trial):
            acc = knn.evaluate(X, y, X_test, y_test)
            acc_list.append(acc)

        print(np.mean(np.array(acc_list)))

def main():
    train_percentage = 0.1
    test_percentage = 0.1
    trial = 5

    mnist_experiment(trial, train_percentage, test_percentage)
    encoding_experiment(trial, train_percentage, test_percentage)
    
if __name__ == '__main__':
    main()





__author__ = 'Jinyi Zhang'

import pickle
import random

import numpy as np
from metric_learn import NCA

from knn import kNN

def nca_mnist_experiment(trial, train_percentage=0.1, test_percentage=0.1):

    encoding_train_imgs_path = './data/MNIST_encoding/tf_train.encoding'
    encoding_test_imgs_path = './data/MNIST_encoding/tf_test.encoding'

    train_labels_path = './data/MNIST_encoding/tf_train.labels'
    test_labels_path = './data/MNIST_encoding/tf_test.labels'

    encoding_train = pickle.load(open(encoding_train_imgs_path, 'rb'))
    encoding_test = pickle.load(open(encoding_test_imgs_path, 'rb'))

    print(encoding_train.shape)

    train_labels = pickle.load(open(train_labels_path, 'rb'))
    test_labels = pickle.load(open(test_labels_path, 'rb'))

    print(train_labels.shape)

    m = len(encoding_train)
    train_m = int(m * train_percentage)
    sel = random.sample(range(m), train_m)
    X = encoding_train.astype(np.float)[sel]
    y = train_labels[sel]

    print(X.shape)
    print(y.shape)

    m = len(encoding_test)
    test_m = int(m * test_percentage)
    sel = random.sample(range(m), test_m)

    X_test = encoding_test.astype(np.float)[sel]
    y_test = test_labels[sel]

    print(X_test.shape)
    print(y_test.shape)

    knn = kNN()
    k_valus = [1, 3, 5, 7]
    for k in k_valus:
        knn.k = k

        acc_list = []
        for _ in range(trial):
            acc = knn.evaluate(X, y, X_test, y_test)
            acc_list.append(acc)

        print(np.mean(np.array(acc_list)))


    nca = NCA(max_iter=100, learning_rate=0.01)
    nca.fit(X, y)
    x_train = nca.transform()
    x_test = nca.transform(X_test)

    for k in k_valus:
        knn.k = k

        acc_list = []
        for _ in range(trial):
            acc = knn.evaluate(x_train, y, x_test, y_test)
            acc_list.append(acc)

        print(np.mean(np.array(acc_list)))


def main():
    train_percentage = 0.01
    test_percentage = 0.01
    trial = 1

    nca_mnist_experiment(trial, train_percentage, test_percentage)
    
if __name__ == '__main__':
    main()

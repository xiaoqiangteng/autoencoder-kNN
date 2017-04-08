__author__ = 'Jinyi Zhang'

import pickle
import random
from functools import reduce

import numpy as np

from knn import kNN

class CifarPreprocess(object):

    def __init__(self):

        self.batch_dir_path = './data/CIFAR_10/cifar-10-batches-py/'
        self.train_batchs = [self.batch_dir_path + 'data_batch_{}'.format(i) for i in range(1, 6)]
        self.test_batch = self.batch_dir_path + 'test_batch'
    
        self.img_rows = 32
        self.img_cols = 32
        self.num_channels = 3
        self.num_classes = 10

    def load_cifar_data(self, batchs=[1]):

        # Load the train data
        data_list = []
        y_train = []
        for i in batchs:
            # Load the CIFAR data dict
            batch_dict = pickle.load(open(self.train_batchs[i - 1], 'rb'), encoding='bytes')
            
            data_list.append(batch_dict[b'data'])
            y_train += batch_dict[b'labels']

        self.X_train = reduce(lambda a, b: np.concatenate((a, b), axis=0), data_list)
        self.y_train = np.array(y_train)

        # Load the test data
        test_dict = pickle.load(open(self.test_batch, 'rb'), encoding='bytes')
        self.X_test = test_dict[b'data']
        self.y_test = np.array(test_dict[b'labels'])


def cifar_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    cp = CifarPreprocess()

    batchs = [1]
    cp.load_cifar_data(batchs)

    # Random pick samples
    m, _ = cp.X_train.shape
    train_m = int(m * train_percentage)
    sel = random.sample(range(m), train_m)

    X_train = cp.X_train.astype(np.float)[sel]
    y_train = cp.y_train[sel]

    m, _ = cp.X_test.shape
    test_m = int(m * test_percentage)
    sel = random.sample(range(m), test_m)

    X_test = cp.X_test.astype(np.float)[sel]
    y_test = cp.y_test[sel]

    # Init knn model
    knn = kNN()

    # Do the experiment
    k_valus = [1, 3, 5, 7]
    for k in k_valus:
        knn.k = k

        acc_list = []
        for _ in range(trial):
            acc = knn.evaluate(X_train, y_train, X_test, y_test)
            acc_list.append(acc)

        print(np.mean(np.array(acc_list)))


def main():
    # cp = CifarPreprocess()

    # batchs = [1]
    # cp.load_cifar_data(batchs)

    # print(cp.X_train.dtype)

    trial = 1
    train_percentage = 1
    test_percentage = 1

    cifar_experiment(trial, train_percentage, test_percentage)

if __name__ == '__main__':
    main()




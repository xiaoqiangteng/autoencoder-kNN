#! /usr/bin/env python

__author__ = 'Jinyi Zhang'

import random

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    ocr = loadmat('ocr.mat')

    test_data = ocr['testdata']
    test_labels = ocr['testlabels']

    trials = 1
    mean_error_rates = []
    std_error_rates = []
    N = [60000]
    for n in N:
        error_rates = []
        for count in range(trials):
            sel = random.sample(range(60000), n)
            data = ocr['data'][sel]
            labels = ocr['labels'][sel]

            preds = nn_classifier(data, labels, test_data)

            correct_number = 0
            for i, v in enumerate(preds):
                if v[0] == test_labels[i][0]:
                    correct_number += 1

            error_rates.append(1 - float(correct_number) / len(preds))

        u = np.mean(error_rates)
        std = np.std(error_rates, dtype=np.float64)
        mean_error_rates.append(u)
        std_error_rates.append(std)
        print("N = {0}, Mean error rate = {1}, standard deviation = {2}".format(n, u, std))

    # learning_curve_plot = plt.plot(N, mean_error_rates, '-o')
    # plt.errorbar(N, mean_error_rates, yerr=std_error_rates)

    # plt.axis([0, N[-1] + 1000, 0, max(error_rates) + 0.1])
    # plt.xlabel('Number of Training Samples')
    # plt.ylabel('Error Rate')
    # plt.title('Learning Curve Plot')
    # plt.savefig('p1.png')

def nn_classifier(X, Y, test):

    X_float = X.astype(np.float64)
    test_float = test.astype(np.float64)

    # Calculate x^2 
    x_square_list = np.diag(X_float.dot(X_float.T))

    # Calculate the distance for each t for each x
    euclidean_distance_matrix = (np.ones((len(test_float), 1)) * x_square_list.T) - 2 * (test_float.dot(X_float.T))

    # Find the index of minimum distance for each column
    label_index_array = np.argmin(euclidean_distance_matrix, axis=1)

    return Y[label_index_array]


if __name__ == '__main__':
    main()



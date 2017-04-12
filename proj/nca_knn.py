__author__ = 'Jinyi Zhang'

import numpy as np
from metric_learn import NCA, LMNN

from knn import kNN

def nca_mnist_experiment(trial, train_percentage=0.1, test_percentage=0.1):
    knn = kNN()
    knn.load_mnist_data(train_percentage, test_percentage)

    print(knn.x.shape)

    # nca = NCA(max_iter=100, learning_rate=0.01)
    # nca.fit(knn.x[:, :30], knn.y)

    # x_train = nca.transform()
    # x_test = nca.transform(knn.x_test[:, :30])

    # print(x_train.shape)
    # print(x_test.shape)

    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(knn.x[:, :30], knn.y)

    x_train = lmnn.transform()
    x_test = lmnn.transform(knn.x_test[:, :30])

    print(x_train.shape)
    print(x_test.shape)


    # k_valus = [1, 3, 5, 7]
    # for k in k_valus:
    #     knn.k = k

    #     acc_list = []
    #     for _ in range(trial):
    #         acc = knn.evaluate(knn.x, knn.y, knn.x_test, knn.y_test)
    #         acc_list.append(acc)

    #     print(np.mean(np.array(acc_list)))



def main():
    train_percentage = 0.01
    test_percentage = 0.01
    trial = 1

    nca_mnist_experiment(trial, train_percentage, test_percentage)
    
if __name__ == '__main__':
    main()

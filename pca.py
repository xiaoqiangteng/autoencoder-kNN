
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data

def show_pca(X, y, title):

    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'navy', 'turquoise', 'darkorange']
    lw = 0.001

    for color, i in zip(colors, list(range(10))): 
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw, label=str(i))

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig('./tmp/' + title)

def pixel_space():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    X = mnist.test.images
    y = mnist.test.labels

    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    show_pca(X, y, 'pca_pixel.png')

def feature_space():
    encoding_test_imgs_path = './data/MNIST_encoding/tf_test.encoding'
    test_labels_path = './data/MNIST_encoding/tf_test.labels'

    encoding_test = pickle.load(open(encoding_test_imgs_path, 'rb'))
    test_labels = pickle.load(open(test_labels_path, 'rb'))

    X = encoding_test.astype(np.float)
    y = test_labels

    show_pca(X, y, 'pca_feature.png')


def main():
    pixel_space()

    feature_space()

if __name__ == '__main__':
    main()




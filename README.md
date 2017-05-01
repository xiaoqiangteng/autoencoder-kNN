# Learning a Neighborhood Preserving Embedding by Convolutional Autoencoder

In this project, I aim to learn a non-linear encoding function that projects image data samples from the high-dimensional input space to a low-dimensional feature space by a convolutional autoencoder. During the compression, the encoding function preserves the latent neighborhood structure in the input data. Therefore, the K-nearest neighbor classification algorithm performs well in the feature space. I test the method on the MNIST handwritten digits. The network is implemented in pure Tensorflow and I denote the system as CNN-NCA.

## Requirements
0. Python 3
1. Tensorflow
2. Scipy
3. Numpy
4. h5py
5. matplotlib
6. scikit-learn (optional)
7. metric_learn (optional)
8. Keras (optional)

## Scripts 
1. cae_mnist.py
  * A tensorflow implementation of Contractive autoencoder for comparison.
2. cnn_mnist.py
  * A keras implementation of Convolutional autoencoder for comparison.
3. knn.py
  * A vector implementation of KNN algorithm. A large memory is needed if traning on the whole dataset.
4. nca_knn.py
  * Learn a linear transformation in order to improve KNN accuracy. Require too much memory in practical.
5. pca.py
  * Show PCA and CNN-NCA compressed data points in 2D respectively.
6. setup.py
  * Setup the current directory for running these scripts.
7. tf_mnist.py
  * A tensorflow implementation of CNN-NCA.
8. tf_pretrain.py
  * Pretrain the model to learn a good representation.
9. tf_train.py
  * Fine-tune the model to favor KNN algorithm. 

## How to run
```
python setup.py
python tf_pretrain.py
python tf_train.py
python knn.py
```




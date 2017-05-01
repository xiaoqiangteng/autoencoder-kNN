# Learning a Neighborhood Preserving Embedding by Convolutional Autoencoder

In this project, I aim to learn a non-linear encoding function that projects image data samples from the high-dimensional input space to a low-dimensional feature space by a convolutional autoencoder. During the compression, the encoding function preserves the latent neighborhood structure in the input data. Therefore, the K-nearest neighbor classification algorithm performs well in the feature space. I test the method on the MNIST handwritten digits. The network is implemented in pure Tensorflow.

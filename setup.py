
import os

def mkdir_p(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

# logs dir
logs_dir = './logs/'
mkdir_p(logs_dir)

logs = ['cnn_mnist']
for log in logs:
    mkdir_p(logs_dir + log)

# models dir
models_dir = './models/'
mkdir_p(models_dir)

models = ['tf_mnist', 'tf_train', 'cae_mnist', 'cnn_mnist']
for model in models:
    mkdir_p(models_dir + model)

# data dir
data_dir = './data/'
mkdir_p(data_dir)

encoding_dir = ['MNIST_encoding', 'MNIST_2_encoding']
for encoding in encoding_dir:
    mkdir_p(data_dir + encoding)

# tmp dir
tmp_dir = './tmp/'
mkdir_p(tmp_dir)
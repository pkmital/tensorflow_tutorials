# UPDATE (July 12, 2016)

New **free MOOC course** covering all of this material in much more depth, as well as much more including combined variational autoencoders + generative adversarial networks, visualizing gradients, deep dream, style net, and recurrent networks: **https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i/info**

# TensorFlow Tutorials

You can find python source code under the `python` directory, and associated notebooks under `notebooks`.

| | Source code | Description |
| --- | --- | --- |
|1| **[basics.py](python/01_basics.py)** | Setup with tensorflow and graph computation.|
|2| **[linear_regression.py](python/02_linear_regression.py)** | Performing regression with a single factor and bias. |
|3| **[polynomial_regression.py](python/03_polynomial_regression.py)** | Performing regression using polynomial factors.|
|4| **[logistic_regression.py](python/04_logistic_regression.py)** | Performing logistic regression using a single layer neural network.|
|5| **[basic_convnet.py](python/05_basic_convnet.py)** | Building a deep convolutional neural network.|
|6| **[modern_convnet.py](python/06_modern_convnet.py)** | Building a deep convolutional neural network with batch normalization and leaky rectifiers.|
|7| **[autoencoder.py](python/07_autoencoder.py)** | Building a deep autoencoder with tied weights.|
|8| **[denoising_autoencoder.py](python/08_denoising_autoencoder.py)** | Building a deep denoising autoencoder which corrupts the input.|
|9| **[convolutional_autoencoder.py](python/09_convolutional_autoencoder.py)** | Building a deep convolutional autoencoder.|
|10| **[residual_network.py](python/10_residual_network.py)** | Building a deep residual network.|
|11| **[variational_autoencoder.py](python/11_variational_autoencoder.py)** | Building an autoencoder with a variational encoding.|

# Installation Guides

* [TensorFlow Installation](https://github.com/tensorflow/tensorflow)
* [OS specific setup](https://github.com/tensorflow/tensorFlow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
* [Installation on EC2 GPU Instances](http://eatcodeplay.com/installing-gpu-enabled-tensorflow-with-python-3-4-in-ec2/)

For Ubuntu users using python3.4+ w/ CUDA 7.5 and cuDNN 7.0, you can find compiled wheels under the `wheels` directory.  Use `pip3 install tensorflow-0.8.0rc0-py3-none-any.whl` to install, e.g. and be sure to add: `export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
` to your `.bashrc`.  Note, this still requires you to install CUDA 7.5 and cuDNN 7.0 under `/usr/local/cuda`.

# Resources

* [Official Tensorflow Tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/index.html)
* [Tensorflow API](https://www.tensorflow.org/versions/r0.7/api_docs/python/index.html)
* [Tensorflow Google Groups](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
* [More Tutorials](https://github.com/nlintz/TensorFlow-Tutorials)

# Author

Parag K. Mital, Jan. 2016.

http://pkmital.com

# License

See LICENSE.md

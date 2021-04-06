# 4. Testing and train of the networks for image reconstruction of ellipses images.

### Relevant files

------------------------------

* `Demo_train_nn.py` - Train a neural network.
* `config.yml` - Configuration file with parameters to `Demo_train_nn.py`.
* `Demo_test_accuracy.py` - Testing accuracy of the trained network.
* `Demo_test_stability.py` - Testing stability of the trained network.
* `COUNT.txt` - Counter for the trained model. Each run of `Demo_train_nn.py` will increase the counter here.
* `data_management.py` - File for generating and loading the data. This script is based on code from M. Genzel, J. Macdonald, and M. März [GitHub](https://github.com/jmaces/robust-nets).
* `networks.py` - Network architectures.
* `operators.py` - Tensorflow implementation of the subsampled discrete Fourier transform + other stuff.
* `find_adversarial_pert.py` - Code for running computing the worst-case perturbations.
* `utils.py` - Utility functions.

------------------------------

### Dependencies 

The code has been developed for Tensorflow v 2.3.0. To generate the data files, it is also required to install the 
[Operator Discretization Library (odl)](https://odlgroup.github.io/odl/). Other dependencies are Pillow, NumPy, and PyYAML. 

#



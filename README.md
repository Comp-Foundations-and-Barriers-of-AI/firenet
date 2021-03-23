# Can stable and accurate neural networks be computed? - On the barriers of deep learning and Smale's 18th problem

Code related to the paper *"Can stable and accurate neural networks be computed? - On the barriers of deep learning and Smale's 18th problem"* by V. Antun, M. J. Colbrook and A. C. Hansen.

# Overview of the code

The content of this repository can roughly be divided into five parts. 

1. Code in Tensorflow (v1.14) to compute perturbations meant to simulate worst-case effect for FIRENETs.
2. Code in Tensorflow (v1.14) to compute perturbations meant to simulate worst-case effect for AUTOMAP.
3. Code in Matlab used to test the accuracy of FIRENETs.
4. Code in Tensorflow (v2.3) to train and test neural networks on ellipses images.
5. Code in Tensorflow (v2.3) training a LISTA network on inexact inputs. After training one can test both the LISTA networks and FIRENETs.

To make all the code run seamlessly, you need to download the data and modify the corresponding paths in each of the scripts. The data can be downloaded from here [storage_firenet](https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage_firenet.zip) and AUTOMAP network weights from [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/cs_poisson_for_vegard.h5) (3.4 Gb).

## 1. Worst-case perturbations for FIRENET

### Relevant files

------------------------------

* `instability/Demo_test_firenet_stability_auto.py` - Compute perturbations with sizes similar to what we found for the AUTOMAP network
* `instability/Demo_test_firenet_stability_ell.py` - Computes a perturbation for an ellipse image. 
* `instability/Demo_test_firenet_stability_general.py` - Compute perturbations for many different images. All perturbations have roughly the same size.
* `instability/Demo_test_automap_as_input_to_firenet.py` - Uses the output from AUTOMAP as an initial guess for solution in the FIRENET.
* `instability/config_auto.yml` - Configuration file for the script `Demo_test_firenet_stability_auto.py`
* `instability/config_ell.yml` - Configuration file for the script `Demo_test_firenet_stability_ell.py`
* `instability/config_general.yml` - Configuration file for the script `Demo_test_firenet_stability_general.py`
* `instability/COUNT_auto.yml` - Experiment count for the script `Demo_test_firenet_stability_auto.py`
* `instability/COUNT_ell.yml` - Experiment count for the script `Demo_test_firenet_stability_ell.py`
* `instability/COUNT_general.yml` - Experiment count for the script `Demo_test_firenet_stability_general.py`
* `instability/adv_tools_PNAS/automap_config.py` - This file sets the path to much of the data. Remember to modify the paths.

------------------------------

### Dependencies 

The code depends on the [UiO-CS/optimization](https://github.com/UiO-CS/optimization/tree/exp_decay) and [UiO-CS/tf-wavelets](https://github.com/UiO-CS/tf-wavelets/tree/haarFix) packages. Note that we are using the branches "exp_dacay" and "haarFix", respectively. These packages must be installed, or you must point your PYTHONPATH to them, for all of these scripts to work. The code also has other dependencies, such as TensorFlow (v1.14), NumPy, PIL, SciPy, Matplotlib etc. but these should be straight forward to install.  

## 2. Worst-case perturbations for AUTOMAP

### Relevant files

------------------------------

* `instability/adv_tools_PNAS/*` - This directory contains all the code from the [PNAS paper](https://doi.org/10.1073/pnas.1907377117) necessary to compute the worst-case perturbations for the AUTOMAP network.
* `instability/Demo_test_automap_stability.py` - Script for computing worst-case perturbations for AUTOMAP.
* `instability/_2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg.py` - The AUTOMAP architecture (obtained after communicating with the authors of the network).
* `instability/adv_tools_PNAS/automap_config.py` - This file contains many of the paths to the data. Remember to modify these paths before trying the other scripts.

------------------------------

### Dependencies 

This code has been developed in TensorFlow v1.14.

## 3. Matlab version of the network - Testing of accuracy

### Relevant files

------------------------------

* `matlab/cil*` - Files taken from the [CIlib](https://github.com/vegarant/cilib) and edited. 
* `matlab/cp*` - A MatLab implementation of FIRENETs. 
* `matlab/generate_sparsity_weights.m` - Script used to create the weights in the weighted l^1 norm (with the right ordering).
* `Demo_test_algorithm_general.m` - Script checking that the Tensorflow and MatLab implementation computes the same. 

------------------------------

### Dependencies 

To run the experiments involving the Walsh-Hadamard transform, it is recommended to install the package [Fastwht](https://bitbucket.org/vegarant/fastwht/src/master/). It is substantially faster than MatLab's own implementation `fwht`. To read YAML configuration files install [yamlmatlab](https://github.com/ewiger/yamlmatlab).

## 4. Testing and train of the networks for image reconstruction of ellipses images.

### Relevant files

------------------------------

* `ellipses/Demo_train_nn.py` - Train a neural network.
* `ellipses/config.yml` - Configuration file with parameters to `Demo_train_nn.py`.
* `ellipses/Demo_test_accuracy.py` - Testing accuracy of the trained network.
* `ellipses/Demo_test_stability.py` - Testing stability of the trained network.
* `ellipses/COUNT.txt` - Counter for the trained model. Each run of `Demo_train_nn.py` will increase the counter here.
* `ellipses/data_management.py` - File for generating and loading the data. This script is based on code from M. Genzel, J. Macdonald, and M. März [GitHub](https://github.com/jmaces/robust-nets).
* `ellipses/networks.py` - Network architectures.
* `ellipses/operators.py` - Tensorflow implementation of the subsampled discrete Fourier transform + other stuff.
* `ellipses/find_adversarial_pert.py` - Code for running computing the worst-case perturbations.
* `ellipses/utils.py` - Utility functions.

------------------------------

### Dependencies 

The code has been developed for Tensorflow v 2.3.0. To generate the data files, it is also required to install the 
[Operator Discretization Library (odl)](https://odlgroup.github.io/odl/). Other dependencies are Pillow, NumPy, and PyYAML. 

## 5. Testing LISTA and FIRENETs on inexact inputs

This code is meant to illustrate that neither a LISTA network or FIRENETs, in general can solve the problem (P_3) given inexact training data.

### Relevant files

------------------------------

* `inexact_input/Demo_train_nn.py` - Train a single LISTA network.
* `inexact_input/Demo_train_nn_auto.py` - Train a single LISTA network. This script is intended to be called from the script `Schedule_training.py`.
* `inexact_input/config.yml` - Configuration file with parameters to `Demo_train_nn.py` and `Schedule_training.py`.
* `inexact_input/Demo_test_lista.py` - Tests all the LISTA networks trained by `Schedule_training.py` on a unseen approximation to the data. 
* `inexact_input/Demo_test_firenet.py` - Testing the firenets on a unseen approximation to the data. 
* `inexact_input/COUNT.txt` - Counter for the trained model. Each run of `Demo_train_nn.py` will increase the counter here.
* `inexact_input/data_management.py` - File for generating and loading the data. 
* `inexact_input/networks.py` - Network architecture of the LISTA network.
* `inexact_input/firenet.py` - Python implementation of the FIRENETs.
* `inexact_input/utils.py` - Utility functions.

------------------------------

### Dependencies 

The code has been developed for Tensorflow v 2.3.0.

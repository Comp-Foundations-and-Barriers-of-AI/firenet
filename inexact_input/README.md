##Testing LISTA and FIRENETs on inexact inputs

This code is meant to illustrate that neither a LISTA network or FIRENETs, in general can solve the problem (P_3) given inexact training data.

### Relevant files

------------------------------

* `Schedule_training.py` - Scrit used to train multiple LISTA networks for different values of 'n' and 'K'.
* `Demo_train_nn.py` - Train a single LISTA network.
* `Demo_train_nn_auto.py` - Train a single LISTA network. This script is intended to be called from the script `Schedule_training.py`.
* `config.yml` - Configuration file with parameters to `Demo_train_nn.py` and `Schedule_training.py`.
* `Demo_test_lista.py` - Tests all the LISTA networks trained by `Schedule_training.py` on a unseen approximation to the data. 
* `Demo_test_firenet.py` - Testing the firenets on a unseen approximation to the data. 
* `COUNT.txt` - Counter for the trained model. Each run of `Demo_train_nn.py` will increase the counter here.
* `data_management.py` - File for generating and loading the data. 
* `networks.py` - Network architecture of the LISTA network.
* `firenet.py` - Python implementation of the FIRENETs.
* `utils.py` - Utility functions.

------------------------------

### Dependencies 

The code has been developed for Tensorflow v 2.3.0.


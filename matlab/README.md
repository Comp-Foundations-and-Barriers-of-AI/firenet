## Matlab version of FIRENET 

### Relevant files

------------------------------

* `matlab/cil*` - Files taken from the [CIlib](https://github.com/vegarant/cilib) and edited. 
* `matlab/cp*` - A MatLab implementation of FIRENETs. 
* `matlab/generate_sparsity_weights.m` - Script used to create the weights in the weighted l^1 norm (with the right ordering).
* `Demo_test_algorithm_general.m` - Script checking that the Tensorflow and MatLab implementation computes the same. 
* `Demo_test_firenet_detail.m` - Script used to produce the FIRENET reconstruction in Figure 4 (of the text detail). 
* `Demo_convergence_test.m` - Script used to examing the convergence rate of FIRENET. 

------------------------------

### Dependencies 

To run the experiments involving the Walsh-Hadamard transform, it is recommended to install the package [Fastwht](https://bitbucket.org/vegarant/fastwht/src/master/). It is substantially faster than MatLab's own implementation `fwht`. To read YAML configuration files install [yamlmatlab](https://github.com/ewiger/yamlmatlab).

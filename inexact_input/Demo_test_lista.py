"""
This script tests all the LISTA models trained by `Schedule_training.py` on an
example with inexact input.
"""

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from data_management import generate_test_sample
from network import LISTA
from os.path import join, isdir
from os import mkdir
import yaml

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


nbr_of_models = 9
dest_model = 'models_auto'

for model_nbr in range(1,nbr_of_models+1):
    path_model = join(dest_model, f'model_{model_nbr:03}')
    configfile = join(path_model, 'config.yml')
    with open(configfile) as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);
    
    # Data parameters  
    K = cgf['DATA']['K'];
    delta = eval(cgf['DATA']['delta']);
    lam = cgf['DATA']['lam'];
    N1 = cgf['DATA']['N1'];
    N2 = cgf['DATA']['N2'];
    sparsity = cgf['DATA']['sparsity'];
    j = cgf['DATA']['j'];
    n = cgf['DATA']['n'];
    use_dct = cgf['DATA']['use_dct'];

    # Network parameter
    nbr_layers = cgf['NETWORK']['nbr_layers']
    
    # Load data
    yn, An, xn = generate_test_sample(delta, K, lam, j, n, N1, N2, use_dct=use_dct)
    y, A, x_star = generate_test_sample(delta, K, lam, j-1, n, N1, N2, use_dct=use_dct)



    yn = np.swapaxes(yn, 0,1);
    y  = np.swapaxes(y, 0,1);
    xn = np.swapaxes(xn, 0,1);
    x_star = np.swapaxes(x_star, 0,1);

    N = N1+N2;

    model = LISTA(N2+1, N, A, nbr_layers, dtype=tf.float64)

    #pred = model(ys);
    name_model = f'model_best_val';
    model.load_weights(join(path_model, name_model))
    x_pred = model(yn);

    diff_x = np.linalg.norm(x_pred- x_star, 2);
    print(fr" {diff_x:.7f} &  & $n = {n}$ & $  $10^{{ -{K}  }}$ & $K = {K}$ \\")



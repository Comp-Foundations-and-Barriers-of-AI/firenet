import numpy as np
import tensorflow as tf
from PIL import Image
import os
from os.path import join
import shutil



l2_norm_of_tensor = lambda x: np.sqrt((abs(x)**2).sum());

def print_array_elements(x_pred, x_true):
    x_pred = np.squeeze(x_pred)
    x_true = np.squeeze(x_true)
    for k in range(len(x_pred)):
        print(f"x_pred[{k+1:3}]: {x_pred[k]:10.8f}, x_true[{k+1:3}]: {x_true[k]:10.8f}, error: {abs(x_pred[k] - x_true[k])}")
    
    n_diff = np.linalg.norm(x_pred-x_true)
    n_x = np.linalg.norm(x_true)
    print('l2_error: ', n_diff, ', rel_err: ', n_diff/n_x)


def print_A_y_dist(y, yn, A, An, n):
    diff_y = np.linalg.norm(y-yn,2)
    diff_A = np.linalg.norm(A-An,2)
    print(f"|y - y_n|: {diff_y} < {2**(-n)} = 2**(-{n})")
    print(f"|A - A_n|: {diff_A} < {2**(-n)} = 2**(-{n})")

def print_dist_between_minimizers(x_pred, x_true, K):
    x_pred = np.squeeze(x_pred)
    x_true = np.squeeze(x_true)
    diff_xs = np.linalg.norm(x_pred-x_true, 2)
    print(f"|x_pred - x_true|: {diff_xs} > {10**(-K)} = 10**(-K) for K = {K}")

def read_count(count_path, count_name="COUNT.txt"):
    """ Read and updates the runner count. 
    
    To keep track of all the different runs of the algorithm, one store the 
    run number in the file 'COUNT.txt' at ``count_path``. It is assumed that 
    the file 'COUNT.txt' is a text file containing one line with a single 
    integer, representing number of runs so far. 

    This function reads the current number in this file and increases the 
    number by 1. 
    
    :return: Current run number (int).
    """
    fname = join(count_path, count_name);
    infile = open(fname);
    data = infile.read();
    count = int(eval(data));
    infile.close();

    outfile = open(fname, 'w');
    outfile.write('%d ' % (count+1));
    outfile.close();
    return count;


def cut_to_01(x):
    """ Creates a copy of x, where all elements in x which are greater than 1
    are set to 1 and all elements which are smaller than 0 are set to 0. 
    """
    z = x.copy();
    z[z < 0] = 0
    z[z > 1] = 1
    return z


def scale_to_01(x):
    """ Maps all elements in x to lie in the interval [0,1] using an
        elementwise affine map.
    """
    ma = np.amax(x);
    mi = np.amin(x);
    z = (x.copy()-mi)/(ma-mi);
    return z

def clear_model_dir(path):
    """ Removes the directory listed by path, and recreates it.

    Arguments
    ---------
    path (str): Path to directory

    Returns
    -------
    Nothing

    """
    if os.path.isdir(path):
        shutil.rmtree(path);
    os.mkdir(path)

def set_np_dtype(dtype):
    """ Converts a tensorflow float32 or float64 to a numpy dtype

    Arugments
    ---------
    dtype: Tensorflow dtype (one of tf.float32 and tf.float64)

    Returns
    -------
    np_dtype: Numpy version of the tensorflow dtype
    """
    
    if dtype == tf.float32:
        np_dtype = np.float32
    elif dtype == tf.float64:
        np_dtype = np.float64
    else:
        print('dtype: ', dtype, ' is unknown')
    return np_dtype

# Set up computational resource 
def set_computational_resource(use_gpu, compute_node, verbose=True):
    """ Set the enviormental variable CUDA_VISIBLE_DEVICES to the desired number

    Arguments
    ---------
    use_gpu (bool): Whether or not to use the gpu. If False, 
                    CUDA_VISIBLE_DEVICES will be set to -1.
    compute_node (int): The GPU number you would like to use
    verbose (bool): Whether or not to print information about the changes

    Returns
    -------
    Nothing
    """
    if verbose:
        print(f"""\nCOMPUTER SETUP
gpu: {use_gpu}""")
        print('PID: ', os.getpid())
    
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
        if verbose:
            print(f'Compute node: {compute_node}')
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

def set_soft_memory_allocation_on_gpu(soft=True):
    """ Turn off TensorFlows's default behaviour of allocating all GPU memory

    Arguments
    ---------
    soft (bool): True turns off default behaviour

    Returns
    -------
    Nothing
    """
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, soft)
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    


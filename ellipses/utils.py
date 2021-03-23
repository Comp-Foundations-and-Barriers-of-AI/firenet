import numpy as np
import tensorflow as tf
from PIL import Image
import os
from os.path import join
import shutil


def compute_radius(opA, images):
    return tf.math.sqrt(tf.reduce_sum(tf.math.pow(tf.math.abs(opA(images)), 2), axis=(1,2,3), keepdims=True));


def save_data_as_im_single(opA, model, data, model_nbr, im_nbr, set_name, dest):
    adj = opA.adjoint(opA(data))
    me = tf.concat((tf.math.real(adj), tf.math.imag(adj)), axis=-1)
    np_im_rec = model(me).numpy()
    np_adj = adj.numpy()

    im = np.squeeze(data[0])
    im_adj = abs(np.squeeze(np_adj[0]))
    im_rec = np.squeeze(np_im_rec[0])
    
    N = im_adj.shape[0];
    bd = 5;
    im_frame = np.ones([N, 3*N + 2*bd]);
    im_frame[:,:N] = cut_to_01(abs(im));
    im_frame[:,N+bd:2*N+bd] = im_rec;
    im_frame[:,2*N+2*bd:] = im_adj;

    im1 = Image.fromarray(np.uint8(255*cut_to_01(im_frame)))
    fname = f'im_mod_{model_nbr}_{set_name}_nbr_{im_nbr:03d}.png';
    im1.save(join(dest,fname));

def save_data_as_im(opA, model, data, model_nbr, im_nbr, set_name, dest):
    adj = opA.adjoint(opA(data))
    me = tf.concat((tf.math.real(adj), tf.math.imag(adj)), axis=-1)
    np_im_rec = model(me).numpy()
    np_adj = adj.numpy()

    im = np.squeeze(data[im_nbr])
    im_adj = abs(np.squeeze(np_adj[im_nbr]))
    im_rec = np.squeeze(np_im_rec[im_nbr])
    
    N = im_adj.shape[0];
    bd = 5;
    im_frame = np.ones([N, 3*N + 2*bd]);
    im_frame[:,:N] = cut_to_01(abs(im));
    im_frame[:,N+bd:2*N+bd] = im_rec;
    im_frame[:,2*N+2*bd:] = im_adj;

    im1 = Image.fromarray(np.uint8(255*cut_to_01(im_frame)))
    fname = f'im_mod_{model_nbr}_{set_name}_nbr_{im_nbr}.png';
    im1.save(join(dest,fname));

l2_norm_of_tensor = lambda x: np.sqrt((abs(x)**2).sum());

def load_model(model, N, path_model, model_name):
    x = tf.zeros([1,N,N,2]);
    model.predict(x);
    model.load_weights(join(path_model, model_name))
    return model

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
    """ Maps all elements in x to lie in the interval [0,1]
    """
    z = x.copy();
    z[z < 0] = 0
    z[z > 1] = 1
    return z


def scale_to_01(x):
    """ Maps all elements in x to lie in the interval [0,1]
    """
    ma = np.amax(x);
    mi = np.amin(x);
    z = (x.copy()-mi)/(ma-mi);
    return z

def clear_model_dir(path):
    """ Remove directory listed by path, and recreate it.

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
    


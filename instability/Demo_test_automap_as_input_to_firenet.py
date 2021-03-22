"""
The FIRENET takes two inputs, the measurements y (in measurement domain) and an initial guess x0 in the image domain. This script takes perturbed measurements, mean to simulate worst-case effect on AUTOMAP, feed these measurements to AUTOMAP, which then again produce a poor image. This poor image is used as initial guess x0, to FIRENET, alongside with the perturbed measurements. Given this as initial data, the FIRENET reconstructs the perturbed image.   
"""

import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import matplotlib.image as mpimg;
import numpy as np;
from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.automap_tools import load_runner, hand_f, hand_dQ, read_automap_weights, read_automap_k_space_mask, compile_network;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor
from adv_tools_PNAS.automap_tools import load_runner
from PIL import Image
from scipy.io import loadmat
import time

from optimization.gpu.operators import MRIOperator
from optimization.gpu.algorithms import SR_LASSO_Ergodic, SR_LASSO_exponential
from optimization.utils import generate_weight_matrix
from tfwavelets.dwtcoeffs import get_wavelet
from tfwavelets.nodes import idwt2d
from PIL import Image

compute_node = 2
use_gpu = True
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

# Turn on soft memory allocation
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
sess = tf.compat.v1.Session(config=tf_config)


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

N = 128
runner_id = 19
im_nbr = 3
plot_dest = 'plots_automap_to_firenet'
data_dest = 'data_automap_to_firenet'
dtype = tf.float32
npdtype = np.float32
cdtype = tf.complex64
lam   = 0.00025
tau   = 1
sigma = 1
alg_name = 'firenet'
delta = 1e-9
n_iter  = 5
p_iter  = 5
wavelet_name = 'db1';
nres = 5
sparsity_levels = [4, 12, 48, 50, 100, 100]

use_weights = True
if use_weights:
    # Warning: It is very important that mri is an NxN image
    print('Sparsity levels (coarse to fine): ', sparsity_levels)
    weights = generate_weight_matrix(N, sparsity_levels, npdtype)
    weights = np.expand_dims(weights, -1)
else:
    weights = np.ones([N, N, 1]).astype(npdtype);


db_wavelet = get_wavelet(wavelet_name, dtype)
samp = np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool))

# Load data
runner = load_runner(runner_id);
mri_data = runner.x0[0];
rr = runner.r[-1];
num_pert = len(runner.r);
batch_size = mri_data.shape[0];

epsilon_0 = l2_norm_of_tensor(samp*np.fft.fft2(mri_data[im_nbr])/N);
samp = np.expand_dims(samp, -1)

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest)
if not (os.path.isdir(data_dest)):
    os.mkdir(data_dest)

############################################################################
###               Build the Tensorflow graph for LASSO                   ### 
############################################################################

# Parameters for CS algorithm
tf_L_A = tf.constant( 1.0, dtype, shape=(), name='L_A');
pl_sigma = tf.compat.v1.placeholder(dtype, shape=(), name='sigma')
pl_tau   = tf.compat.v1.placeholder(dtype, shape=(), name='tau')
pl_lam   = tf.compat.v1.placeholder(dtype, shape=(), name='lambda')
pl_n_iter = tf.compat.v1.placeholder(tf.int32, shape=(), name='n_iter')
pl_p_iter = tf.compat.v1.placeholder(tf.int32, shape=(), name='p_iter')
pl_eps_0 = tf.compat.v1.placeholder(dtype, shape=(), name='eps_0');
pl_delta = tf.compat.v1.placeholder(dtype, shape=(), name='delta');

# For the weighted l^1-norm
pl_weights = tf.compat.v1.placeholder(dtype, shape=[N,N,1], name='weights')

# Build Primal-dual graph
tf_input = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

op = MRIOperator(tf_samp_patt, db_wavelet, nres, dtype=dtype)
tf_measurements = op.sample(tf_input)

tf_adjoint_coeffs = op(tf_measurements, adjoint=True)
adj_real_idwt = idwt2d(tf.math.real(tf_adjoint_coeffs), db_wavelet, nres)
adj_imag_idwt = idwt2d(tf.math.imag(tf_adjoint_coeffs), db_wavelet, nres)
tf_adjoint = tf.complex(adj_real_idwt, adj_imag_idwt)

tf_initial_x = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='x0')

if alg_name.lower() ==  'firenet':
    alg = SR_LASSO_exponential(tf_measurements, tf_initial_x, op, p_iter=pl_p_iter, tau=pl_tau, sigma=pl_sigma,  lam=pl_lam, weights_mat=pl_weights, L_A=tf_L_A, eps_0=pl_eps_0, delta=pl_delta, dtype=dtype)
    result_coeffs = alg.run(n_iter=pl_n_iter)
elif alg_name.lower() == 'sr_lasso_ergodic': 
    alg = SR_LASSO_Ergodic(op, p_iter=pl_n_iter, tau=pl_tau, sigma=pl_sigma, lam=pl_lam, weights_mat=pl_weights, dtype=dtype) # Obs. notice mixup between n_iter and p_iter.
    result_coeffs = alg.run(tf_measurements, tf_initial_x)
else:
    print(f"alg_name: {alg_name} not recognized");

real_idwt = idwt2d(tf.math.real(result_coeffs), db_wavelet, nres)
imag_idwt = idwt2d(tf.math.imag(result_coeffs), db_wavelet, nres)
tf_recovery = tf.complex(real_idwt, imag_idwt)

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)

# AUTOMAP reconstruction of noisy image
im_rec_automap = f(mri_data + rr);

mri = mri_data[im_nbr] + rr[im_nbr];
im_rec_automap = im_rec_automap[im_nbr];
mri = np.expand_dims(mri, -1)
im_rec_automap = np.expand_dims(im_rec_automap, -1)

print('Global initilization done')
print('mri.shape: ', mri.shape)
noiseless = sess.run(tf_recovery, feed_dict={ 'tau:0': tau,
                                              'lambda:0': lam,
                                              'sigma:0': sigma,
                                              'weights:0': weights,
                                              'n_iter:0': n_iter,
                                              'p_iter:0': p_iter,
                                              'image:0': mri,
                                              'x0:0': im_rec_automap,
                                              'eps_0:0': epsilon_0,
                                              'delta:0': delta,
                                              'sampling_pattern:0': samp,
    })
print('Computed noiseless reconstruction')

fname_out_rec  = join(plot_dest, f'im_auto_as_input_to_firenet_rID_{runner_id}_im_nbr_{im_nbr}_r_{num_pert-1}_rec.png'); 
fname_out_orig = join(plot_dest, f'im_auto_as_input_to_firenet_rID_{runner_id}_im_nbr_{im_nbr}_r_{num_pert-1}_orig.png'); 

Image_im_rec_noiseless = Image.fromarray(np.uint8(255*np.abs(np.squeeze(noiseless))));
Image_im_orig = Image.fromarray(np.uint8(255*np.abs(np.squeeze(mri))));

Image_im_rec_noiseless.save(fname_out_rec);
Image_im_orig.save(fname_out_orig);

sess.close();



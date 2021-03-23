import tensorflow as tf;

from operators import Fourier_operator
from data_management import load_sampling_pattern, Ellipses_dataset  
from os.path import join
import os
import time

import numpy as np
from PIL import Image
from utils import (set_soft_memory_allocation_on_gpu, 
                   set_computational_resource,
                   set_np_dtype,
                   l2_norm_of_tensor,
                   save_data_as_im,
                   save_data_as_im_single,
                   scale_to_01,
                   cut_to_01,
                   clear_model_dir)

from networks import UNet
from tqdm import tqdm
import yaml

model_nbrs = [120];
epoch_idx = 99
test_size = 6;
use_local_intensity_diff = False
intensity_diff = 30;
add_noise = True
noise_levels = [2,5,7,10]
set_name = 'test'
batch_size = 2
use_gpu = True
compute_node = 2

for model_nbr in model_nbrs:
    print('Model number: ', model_nbr)
    
    dest = 'plots'
    if not os.path.isdir(dest):
        os.mkdir(dest)
    
    if epoch_idx < 0:
        model_name = 'model_final'
        print('Model from final epoch')
    else:
        print(f'Model from epoch {epoch_idx}')
        model_name = f"model_epoch_{epoch_idx}" 


    path_model = f'models/model_{model_nbr:03}'
    configfile = 'config.yml'
    with open(join(path_model, configfile)) as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);
    
    set_computational_resource(use_gpu, compute_node)
    
    set_soft_memory_allocation_on_gpu()
    
    dtype = eval(cgf['COMPUTER_SETUP']['dtype'])
    cdtype = eval(cgf['COMPUTER_SETUP']['cdtype'])
    np_dtype = set_np_dtype(dtype);
    
    # DATA
    N = cgf['DATA']['N'];
    srate = cgf['DATA']['srate'];
    path_data = cgf['DATA']['path_data'];
    print('\nDATA')
    print(f'N: {N}')
    print(f'srate: {round(100*srate)}')
    
    # Train
    #batch_size = cgf['TRAIN']['batch_size']
    nbr_epochs = cgf['TRAIN']['epochs']
    shuffle = cgf['TRAIN']['shuffle']
    train_size = cgf['TRAIN']['train_size']
    #test_size = cgf['TRAIN']['test_size']
    intensity_diff_local = cgf['TRAIN']['intensity_diff']
    if use_local_intensity_diff:
        intensity_diff = intensity_diff_local
    
    print('\nTRAIN:')
    print(f'train_size: {train_size}')
    print(f'test_size: {test_size}')
    print(f'batch_size: {batch_size}')
    print(f'epochs: {nbr_epochs}')
    print(f'shuffle: {shuffle}')
    print(f'intensity diff: {intensity_diff_local}')
    
    kernel_size = eval(cgf['NETWORK']['kernel_size'])
    use_bias = cgf['NETWORK']['use_bias']
    init_features = cgf['NETWORK']['init_features']
    
    print('\nNETWORK:')
    print(f'kernel_size: {kernel_size}')
    print(f'use_bias: {use_bias}')
    print(f'init_features: {init_features}')
    print('')
    
    
    vm = 1
    fname_pattern = f"spf2_DAS_N_{N}_srate_{round(100*srate):02d}_db{vm}.png"
    mask = load_sampling_pattern(join(path_model, fname_pattern));
    
    opA = Fourier_operator(mask, dtype=dtype);
    
    path_data = join(path_data, f'raw_data_{N}_TF_tumor');
    
    ds_test = Ellipses_dataset(join(path_data, set_name), test_size, N,
                                intensity_diff=intensity_diff,
                                dtype=dtype)
    ds_test1 = ds_test.batch(batch_size)
    
    model1 = UNet(use_bias=use_bias, init_features=init_features, kernel_size=kernel_size);
    
    print('First iteration through the test data')
    k = 0
    for x in tqdm(ds_test1, desc="Meaningless iteration", total=2*test_size//batch_size):
    
        #print(x.shape, 'k: ', k);
        k += 1
        z = opA.adjoint(opA(x));
        measurements = tf.concat((tf.math.real(z), tf.math.imag(z)), axis=-1);
        model1.predict(measurements);
        break;
    
    model1.load_weights(join(path_model, model_name))
    
    ds_test2 = ds_test.batch(1)
    
    i = 1;
    path_data_full = join(path_data, set_name);
    
    def concat_images(im, im_rec, im_adj, bd = 5):
        N = im.shape[0];
        im_frame = np.ones([N, 3*N + 2*bd])
        im_frame[:,:N] = cut_to_01(abs(im))
        im_frame[:,N+bd:2*N+bd] = cut_to_01(im_rec)
        im_frame[:,2*N+2*bd:] = cut_to_01(abs(im_adj))
        return im_frame
    
    add_noise_list = [False]
    if add_noise:
        add_noise_list.append(True);
    for i in tqdm(range(test_size), desc="Storing data iteration"):
        
        for im_type in ['clean', f'tumor_{intensity_diff}']:
            fname = join(path_data_full, f'sample_{i}_{im_type}.png');
            
            data = np.expand_dims(np.expand_dims(np.asarray(Image.open(fname), dtype=np.float32)/255, axis=2), axis=0)
            
            adj = opA.adjoint(opA(data, add_noise=False, output_real=False))
            me = tf.concat((tf.math.real(adj), tf.math.imag(adj)), axis=-1)
            np_im_rec = model1(me).numpy()
            np_adj = adj.numpy()
    
            im = np.squeeze(data[0])
            im_adj = abs(np.squeeze(np_adj[0]))
            im_rec = np.squeeze(np_im_rec[0])
    
            im_frame = concat_images(im, im_rec, im_adj) 
            im1 = Image.fromarray(np.uint8(255*im_frame))
            fname_out = f'im_mod_{model_nbr}_{set_name}_nbr_{i:03d}_{im_type}.png';
            im1.save(join(dest,fname_out));
            if add_noise:
                for noise_level in noise_levels:
                    im_type_new = f'{im_type}_noise_{noise_level:02d}'
                    adj = opA.adjoint(opA(data, add_noise=True, noise_magnitude_fixed=noise_level, output_real=False))
                    me = tf.concat((tf.math.real(adj), tf.math.imag(adj)), axis=-1)
                    np_im_rec = model1(me).numpy()
                    np_adj = adj.numpy()
    
                    im = np.squeeze(data[0])
                    im_adj = abs(np.squeeze(np_adj[0]))
                    im_rec = np.squeeze(np_im_rec[0])
    
                    im_frame = concat_images(im, im_rec, im_adj) 
                    im1 = Image.fromarray(np.uint8(255*im_frame))
                    fname_out = f'im_mod_{model_nbr}_{set_name}_nbr_{i:03d}_{im_type_new}.png';
                    im1.save(join(dest,fname_out));



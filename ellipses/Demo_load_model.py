import tensorflow as tf;

from operators import Fourier_operator
from data_management import load_dataset, load_sampling_pattern 
from os.path import join
import time

import numpy as np
from PIL import Image
from utils import (set_soft_memory_allocation_on_gpu, 
                   set_computational_resource,
                   set_np_dtype,
                   l2_norm_of_tensor,
                   save_data_as_im,
                   scale_to_01,
                   cut_to_01,
                   clear_model_dir)

from networks import UNet
from tqdm import tqdm
import yaml

model_nbr = 31;
epoch_idx = 27
path_model = f'models/model_{model_nbr:03}'
use_gpu = True
compute_node = -1

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
path_data = cgf['DATA']['dest_data'];
print('\nDATA')
print(f'N: {N}')
print(f'srate: {round(100*srate)}')

# Train
batch_size = cgf['TRAIN']['batch_size']
nbr_epochs = cgf['TRAIN']['epochs']
shuffle = cgf['TRAIN']['shuffle']
train_size = cgf['TRAIN']['train_size']

print('\nTRAIN:')
print(f'train_size: {train_size}')
print(f'batch_size: {batch_size}')
print(f'epochs: {nbr_epochs}')
print(f'shuffle: {shuffle}')

kernel_size = eval(cgf['NETWORK']['kernel_size'])
use_bias = cgf['NETWORK']['use_bias']
init_features = cgf['NETWORK']['init_features']
print('\nNETWORK:')
print(f'kernel_size: {kernel_size}')
print(f'use_bias: {use_bias}')
print(f'init_features: {init_features}')
print('')



vm = 1
fname_pattern = f"spf2_DAS_N_{N}_srate_{round(100*srate)}_db{vm}.png"
mask = load_sampling_pattern(join(path_model, fname_pattern));

opA = Fourier_operator(mask, dtype=dtype);

path_data = join(path_data, f'raw_data_{N}_TF');

data_train = load_dataset(join(path_data, 'train'), size=25);
data_val = load_dataset(join(path_data, 'val'), size=10);

#train_ds = tf.data.Dataset.from_tensor_slices(data_train).shuffle(100).batch(25);
#val_ds = tf.data.Dataset.from_tensor_slices(data_val).shuffle(100).batch(10);

model1 = UNet(use_bias=use_bias, init_features=init_features, kernel_size=kernel_size);

x = data_val;
z = opA.adjoint(opA(x));
measurements = tf.concat((tf.math.real(z), tf.math.imag(z)), axis=-1);
print(measurements.shape)
model1.predict(measurements);
model1.load_weights(join(path_model, model_name))

save_data_as_im(opA, model1, data_train, model_nbr, 0, 'train', dest);
save_data_as_im(opA, model1, data_train, model_nbr, 1, 'train', dest);
save_data_as_im(opA, model1, data_train, model_nbr, 2, 'train', dest);
save_data_as_im(opA, model1, data_val, model_nbr, 0, 'val', dest);
save_data_as_im(opA, model1, data_val, model_nbr, 1, 'val', dest);
save_data_as_im(opA, model1, data_val, model_nbr, 2, 'val', dest);







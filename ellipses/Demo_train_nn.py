import tensorflow as tf;
import sys

from operators import Fourier_operator
from data_management import load_sampling_pattern, Ellipses_dataset 
from os.path import join
import os

import numpy as np
from PIL import Image

from utils import set_soft_memory_allocation_on_gpu, set_computational_resource, set_np_dtype, scale_to_01, clear_model_dir, read_count

from networks import UNet
from tqdm import tqdm
import yaml
import shutil

configfile = 'config.yml'
with open(configfile) as ymlfile:
    cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

model_nbr = read_count('./')
path_model_base = cgf['MODEL']['dest_model']
if not os.path.isdir(path_model_base):
    os.mkdir(path_model_base)
path_model = join(path_model_base, f'model_{model_nbr:03}');
clear_model_dir(path_model);



print('Model number: ', model_nbr)

shutil.copyfile(configfile, join(path_model, configfile))

with open(join(path_model, configfile)) as ymlfile:
    cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

# Set up computational resource 
set_computational_resource(cgf['COMPUTER_SETUP']['use_gpu'], 
                           cgf['COMPUTER_SETUP']['compute_node'])

set_soft_memory_allocation_on_gpu()

dtype = eval(cgf['COMPUTER_SETUP']['dtype'])
cdtype = eval(cgf['COMPUTER_SETUP']['cdtype'])
np_dtype = set_np_dtype(dtype);


# DATA
N = cgf['DATA']['N'];
srate = cgf['DATA']['srate'];
path_data = join(cgf['DATA']['path_data'], f"raw_data_{N}_TF_tumor")
path_pattern = cgf['DATA']['path_pattern']

vm = 1
fname_pattern = f"spf2_DAS_N_{N}_srate_{round(100*srate):02d}_db{vm}.png"
mask = load_sampling_pattern(join(path_pattern, fname_pattern));
print('\nDATA')
print(f'N: {N}')
print(f'srate: {round(100*srate)}')
print(f'path_data: {path_data}')

shutil.copyfile(join(path_pattern, fname_pattern), join(path_model, fname_pattern))

# Train
batch_size = cgf['TRAIN']['batch_size']
nbr_epochs = cgf['TRAIN']['epochs']
shuffle = cgf['TRAIN']['shuffle']
train_size = cgf['TRAIN']['train_size']
val_size = cgf['TRAIN']['val_size']
test_size = cgf['TRAIN']['test_size']
add_noise = cgf['TRAIN']['add_noise']
noise_level = cgf['TRAIN']['noise_level']
intensity_diff = cgf['TRAIN']['intensity_diff']

print('\nTRAIN:')
print(f'train_size: {train_size}')
print(f'test_size: {test_size}')
print(f'val_size: {val_size}')
print(f'batch_size: {batch_size}')
print(f'epochs: {nbr_epochs}')
print(f'shuffle: {shuffle}')
print(f'add_noise: {add_noise}')
print(f'noise_level: {noise_level}')
print(f'intensity_diff: {intensity_diff}')

kernel_size = eval(cgf['NETWORK']['kernel_size'])
use_bias = cgf['NETWORK']['use_bias']
init_features = cgf['NETWORK']['init_features']
print('\nNETWORK:')
print(f'kernel_size: {kernel_size}')
print(f'use_bias: {use_bias}')
print(f'init_features: {init_features}')
print('')

tau = 1e-3
lr = 1e-2
momentun = 0.9
lam = 0.01;

ds_train = Ellipses_dataset(join(path_data, 'train'), train_size, N,
                            intensity_diff=intensity_diff,
                            dtype=dtype)

ds_val = Ellipses_dataset(join(path_data, 'val'), val_size, N,
                            intensity_diff=intensity_diff,
                            dtype=dtype)

ds_train = ds_train.batch(batch_size)
ds_val = ds_val.batch(batch_size)

if shuffle:
    ds_train = ds_train.shuffle(10);

# Initialize sampling operator
opA = Fourier_operator(mask, dtype=dtype)

# Set up training loss and optimizer
loss_object = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(2e-4)

train_loss = tf.keras.metrics.Mean(name='train_loss')

val_loss = tf.keras.metrics.Mean(name='val_loss')

model = UNet(use_bias=use_bias, init_features=init_features, kernel_size=kernel_size);




@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        z = opA.adjoint(opA(images, add_noise=add_noise, max_noise_magnitude=noise_level));
        measurements = tf.concat((tf.math.real(z), tf.math.imag(z)), axis=-1);
        predictions = model(measurements, training=True)
        loss = loss_object(images, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
    train_loss(loss)

@tf.function
def val_step(images):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    z = opA.adjoint(opA(images));
    measurements = tf.concat((tf.math.real(z), tf.math.imag(z)), axis=-1);
    predictions = model(measurements, training=False)
    v_loss = loss_object(images, predictions)

    val_loss(v_loss)


for epoch in range(nbr_epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    val_loss.reset_states()

    for images in tqdm(ds_train, desc='Training, baches', total=train_size//batch_size):
        train_step(images)

    for val_images in ds_val:
        val_step(val_images)

    fname_model = f'model_epoch_{epoch:02d}';
    model.save_weights(join(path_model, fname_model))

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Validation Loss: {val_loss.result()}, '
    )

fname_model = 'model_final';
model.save_weights(join(path_model, fname_model))




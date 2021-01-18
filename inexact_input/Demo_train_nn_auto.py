"""
This script it intended to be called from the script `Schedule_training.py`.
It trains a single LISTA network to solve the square-root LASSO 
problem. 

All hyper-parameters (except the once set in `Schedule_training.py`) are set 
in the file `config_auto.yml` and is read by this script.

The trained model and the used configuration file are stored under
`models_auto/model_{xxx}`
"""
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from data_management import generate_training_omega
from network import LISTA
from os.path import join, isdir
from os import mkdir, getpid
from utils import set_soft_memory_allocation_on_gpu, set_computational_resource, set_np_dtype, scale_to_01, clear_model_dir, read_count
import yaml
import shutil

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

configfile = 'config_auto.yml'
with open(configfile) as ymlfile:
    cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

if cgf['MODEL']['use_count']:
    model_nbr = read_count('./')
else: 
    model_nbr = cgf['MODEL']['model_nbr']

dest_model = cgf['MODEL']['dest_model'];
if not isdir(dest_model):
    mkdir(dest_model)

path_model = join(dest_model, f'model_{model_nbr:03}');
clear_model_dir(path_model);

print('Model number: ', model_nbr)
print('PID: ', getpid())
shutil.copyfile(configfile, join(path_model, 'config.yml'))

with open(join(path_model, 'config.yml')) as ymlfile:
    cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

# Set up computational resource 
set_computational_resource(cgf['COMPUTER_SETUP']['use_gpu'], 
                           cgf['COMPUTER_SETUP']['compute_node'])

set_soft_memory_allocation_on_gpu()

dtype = eval(cgf['COMPUTER_SETUP']['dtype'])
cdtype = eval(cgf['COMPUTER_SETUP']['cdtype'])
np_dtype = set_np_dtype(dtype);


# DATA
K = cgf['DATA']['K'];
delta = eval(cgf['DATA']['delta']);
lam = cgf['DATA']['lam'];
N1 = cgf['DATA']['N1'];
N2 = cgf['DATA']['N2'];
sparsity = cgf['DATA']['sparsity'];
j = cgf['DATA']['j'];
n = cgf['DATA']['n'];
use_dct = cgf['DATA']['use_dct'];
set_seed = cgf['DATA']['set_seed'];

print('\nDATA')
print(f'K: {K}')
print(f'delta: {delta}')
print(f'lambda: {lam}')
print(f'N1: {N1}')
print(f'N2: {N2}')
print(f'sparsity: {sparsity}')
print(f'j: {j}')
print(f'n: {n}')
print(f'use_dct: {use_dct}')
print(f'set_seed: {set_seed}')

# TRAIN
nbr_epochs = cgf['TRAIN']['epochs']
batch_size = cgf['TRAIN']['batch_size']
train_size = cgf['TRAIN']['train_size']
val_size = cgf['TRAIN']['val_size']
shuffle = cgf['TRAIN']['shuffle']
learning_rate = cgf['TRAIN']['learning_rate']

print('\nTRAIN:')
print(f'train_size: {train_size}')
print(f'val_size: {val_size}')
print(f'batch_size: {batch_size}')
print(f'epochs: {nbr_epochs}')
print(f'shuffle: {shuffle}')
print(f'learning_rate: {learning_rate}')

# NETWORK
nbr_layers = cgf['NETWORK']['nbr_layers']
theta_min_val_init = cgf['NETWORK']['theta_min_val_init']
theta_max_val_init = cgf['NETWORK']['theta_max_val_init']


print('\nNETWORK:')
print(f'nbr_layers: {nbr_layers}')
print(f'theta_min_val_init: {theta_min_val_init}')
print(f'theta_max_val_init: {theta_max_val_init}')
print('')

# Data parameters
ys, A, xs_true = generate_training_omega(train_size, delta, K, lam, j, n, N1,
                                          N2=N2, sparsity=sparsity, 
                                          use_dct=use_dct, set_seed=set_seed)

ys_val, A, xs_true_val = generate_training_omega(val_size, delta, K, lam, 
                                                  j, n, N1, N2=N2, 
                                                  sparsity=sparsity,
                                                  use_dct=use_dct, 
                                                  set_seed=False)

ys = np.swapaxes(ys, 0,1);
xs_true = np.swapaxes(xs_true, 0,1);
ys_val = np.swapaxes(ys_val, 0,1);
xs_true_val = np.swapaxes(xs_true_val, 0,1);

N = N1+N2;

train_ds = tf.data.Dataset.from_tensor_slices((ys,xs_true)).batch(batch_size);
val_ds = tf.data.Dataset.from_tensor_slices((ys_val,xs_true_val)).batch(batch_size);

if shuffle:
    train_ds.shuffle(10)

# Set up training loss and optimizer
loss_object = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')

val_loss = tf.keras.metrics.Mean(name='val_loss')

rel_err = 1
while rel_err > 1 - 1e-5:
    model = LISTA(N2+1, N, A, nbr_layers, theta_min_val_init, theta_max_val_init, dtype=dtype)
    pred = model(ys);
    rel_err = tf.nn.l2_loss(xs_true - pred)/tf.nn.l2_loss(xs_true)

print('---------  Prediction before training  ----------')
print('ys.shape: ', ys.shape)
print('pred.shape: ', pred.shape)
print('pred: ', pred[0:5,0:5])
print('-------------------------------------------------\n\n')

@tf.function
def train_step(y, x):
    with tf.GradientTape() as tape:
        predictions = model(y, training=True)
        loss = loss_object(x, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(tf.nn.l2_loss(x-predictions)/tf.nn.l2_loss(x))

@tf.function
def val_step(y,x):
    predictions = model(y, training=False)
    v_loss = loss_object(x, predictions)

    val_loss(tf.nn.l2_loss(x-predictions)/tf.nn.l2_loss(x))


best_val_loss = np.inf 
for epoch in range(nbr_epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    val_loss.reset_states()

    for y, x in train_ds: 
        train_step(y,x)

    for y, x in val_ds:
        val_step(y, x)

    
    val_res = val_loss.result();
    train_res = train_loss.result();
    
    if val_res  < best_val_loss:
        best_val_loss =  val_res;
        
        name_model = f'model_best_val';
        model.save_weights(join(path_model, name_model))


    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_res}, '
        f'Validation Loss: {train_res}, '
    )



def print_example(ys, model, idx):
    print(f'\n{idx}\n')
    ys1 = np.expand_dims(ys[idx], 0);
    x_pred = np.squeeze(model(ys1))
    x_label = xs_true[idx]
    for i in range(len(x_label)):
        print(f"x_pred[{i+1:2}]: {x_pred[i]:10.6f}, x_label[{i+1:2}]: {x_label[i]:10.6f}, error: {abs(x_pred[i] - x_label[i])}")
    
    n_diff = np.linalg.norm(x_pred-x_label)
    n_x = np.linalg.norm(x_label)
    print('l2_error: ', n_diff, ', rel_err: ', n_diff/n_x)

print_example(ys, model, 1)
print_example(ys, model, 2)
print_example(ys, model, 3)
print_example(ys, model, 4)














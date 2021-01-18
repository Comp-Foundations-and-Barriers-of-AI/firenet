"""
This script trains multiple LISTA networks for different values of `n` and `K`
on different GPUs.

To make the script run sucessfuly it is necessary to change the `command` variable
to use your own python implmentation. Use `which python3` to find this path.
"""


import yaml
import time
import os
n_values = [10, 20, 30]
K_values = [1, 3, 6]

configfile = 'config.yml'
with open(configfile) as ymlfile:
    cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

cgf['MODEL']['use_count'] = False;
compute_node = 1;
fname = 'config_auto.yml'
k = 1;
for i in range(len(K_values)):
    K = K_values[i]
    cgf['COMPUTER_SETUP']['compute_node'] = compute_node + i
    for j in range(len(n_values)):
        n = n_values[j]
        cgf['DATA']['K'] = K
        cgf['DATA']['n'] = n
        cgf['TRAIN']['learning_rate'] = 2e-4*(10**(-(K/3)))
        cgf['MODEL']['model_nbr'] = k;
        print(f"Starting model {k}, with K = {K}, n = {n} on compute node: {compute_node+i}")
        
        with open(fname, 'w') as file1:
            yaml.dump(cgf, file1)

        command = f'nohup nice -n 19 /opt/uio/modules/rhel8/easybuild/software/Python/3.7.4-GCCcore-8.3.0/bin/python3 -u Demo_train_nn_auto.py &> out/out{k}.txt &'

        os.system(command)
        k += 1
        if j != len(K_values)-1:
            time.sleep(4)
    if i != len(n_values)-1:
        time.sleep(5)






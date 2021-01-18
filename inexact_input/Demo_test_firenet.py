"""
This script tests FIRENETS on the inexact input example. 
"""


import numpy as np
from data_management import generate_test_sample
from firenet import firenet
from utils import print_A_y_dist, print_array_elements, print_dist_between_minimizers


# Data parameters
K = 6
delta = 1/6
lam = 1
N1 = 2
N2 = 20
sparsity = 5
j = 2
n = 30
use_dct = True

print('------------ Data Parameters -------------')
print(f'delta: {delta}')
print(f'lambda: {lam}')
print(f'N1: {N1}')
print(f'N2: {N2}')
print(f'Sparsity: {sparsity}')
print(f'j: {j}')
print(f'use_dct: {use_dct}')
print('------------------------------------------\n')

# FIRENET parameters
p_iter = 50;
n_iter = 30;
weights = np.ones(N1+N2);
delta1 = 1e-6;
store_hist = False;

print('------------ FIRENET Parameters ----------')
print(f'p_iter: {p_iter}')
print(f'n_iter: {n_iter}')
print(f'delta: {delta1}')
print(f'store_hist: {store_hist}')
print('------------------------------------------\n')


n_values = [10, 20, 30]
K_values = [1, 3, 6]


print(r"""\toprule 
$ d( \Psi_{A_n}(y_n), \Xi(A,y) )$ & $d( \Phi_{A_n}(y_n), \Xi(A,y) )$ & \begin{tabular}{c} $\|A_n-A\| \leq 2^{-n} $ \\ $\|y_n - y\|_{l^2} \leq 2^{-n}$ \end{tabular}  & $ $10^{-K}$ $ & $\Omega_K$ \\
\midrule""" )

for j in range(len(K_values)):
    for i in range(len(n_values)):
        n = n_values[i]
        K = K_values[j]
        
        yn, An, xn = generate_test_sample(delta, K, lam, j, n, N1, N2, use_dct=use_dct)
        y, A, x_star = generate_test_sample(delta, K, lam, j-1, n, N1, N2, use_dct=use_dct)
        
        y = np.squeeze(y)
        yn = np.squeeze(yn)
        xn = np.squeeze(xn)
        x_star = np.squeeze(x_star)
        
        eps_0 = np.linalg.norm(yn,2)
        L_A = np.linalg.norm(An, 2)
        tau = np.sqrt((1/L_A))-1e-4
        sigma = np.sqrt((1/L_A))-1e-4
        
        x_final, _ = firenet(yn, An, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta1, n_iter, store_hist)
        
        diff_x = np.linalg.norm(x_final- x_star, 2);

        print(fr" & {diff_x:.7f} & $n = {n}$ & $  $10^{{ -{K}  }}$ & $K = {K}$ \\")
print(r'\bottomrule \\')
        #objective = lambda y, A, x: lam*np.linalg.norm(x,1) + np.linalg.norm(np.dot(A,x)-y, 2) 
        
        #print('objective(yn, An, x_final): ', objective(yn, An, x_final))
        #print('objective(yn, An, xn):    ', objective(yn,An,xn))
        #print('objective(y, A, x_final): ', objective(y, A, x_final))
        #print('objective(y, A, x_star):  ', objective(y, A, x_star))
        #print('objective(y, An, x_star): ', objective(y, An, x_star))
        
        #print_array_elements(x_final, x_star)
        #print_dist_between_minimizers(x_final, x_star, K)
        


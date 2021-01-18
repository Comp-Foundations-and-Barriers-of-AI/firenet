"""
This module contains the necessary tools to create a training and test-set for 
the inexact input experiment. 
"""


import numpy as np
import scipy, scipy.fftpack

def generate_test_sample(delta, K, lam, j, n, N1, N2, use_dct=True):
    """
Generates a test triple (y,A,x), where x is a  minimizer for the problem 
        min_z  lam*|z|_1 + |Az-y|_{2}                           (**)

The minimizer x will be of dimension `N1 + N2`, and will have a 
single non-zero component at position `j`.

Arguments
---------
delta (float): In the interval (0, 1/4). Changes the entires in A
K (positive int): Ensures that if we change j, then the minimizer x, changes 
                  with an l^2-distance which is at least 10**K.
lam (float): In the interval (0,1]. The regularization parameter in (**). 
j (int): In the range [1,N1]. Index of the non-zero component in x.
n (positive int): If you change j, the matrices |A-A'| <= 2**(-n).
N1 (int): N1 >= 2. Dimension of x is N1+N2.
N2 (int): N1 >= 0. Dimension of x is N1+N2. Only N2 > 1 have been tested.
use_dct (bool): Whether or not to multiply the sparse vector with a DCT matrix.

Returns
-------
A triple (y,A,x). 
    """
    gamma1 = np.sqrt(2)*(1-delta)*(10**K)/(3*lam)
    # Initialize a row
    a = gamma1*lam*(1/(1-delta))*np.ones(N1);
    
    # Perturb entry j with 2**(-n-1), so that the minimizer is non-zero only at index j
    eps = 2**(-n-1)
    rho = 1-delta
    t = (eps/(gamma1*lam))*rho*rho/(1+(eps/(gamma1*lam))*rho)
    rho_min = 1-delta-t
    a[j-1] = gamma1*lam/rho_min

    na = np.linalg.norm(a, 2);
    A = np.zeros([N2+1, N1+N2]);
    A[0,0:N1] = a
    A[1:,N1:] = na*np.eye(N2);

    x = np.zeros([N1+N2,1]);
    x[j-1, :] = rho_min/(lam*gamma1)
    
    y = np.zeros([1+N2, 1]);
    y[0] = 1
    
    if use_dct:
        y = scipy.fftpack.dct(y, axis=0, norm='ortho')
        A = scipy.fftpack.dct(A, axis=0, norm='ortho')

    return y, A, x

def generate_training_omega(nbr_samples, delta, K, lam, j, n, N1, N2, sparsity, use_dct=True, set_seed=True):
    """
Generates `nbr_samples` test triples (y,A,x). Here the x's are a  minimizer of the problem 
        min_z  lam*|z|_1 + |Az-y|_{2}                           (**)

The x which minimizers (**) will be of dimension `N1 + N2`, and will have a 
single non-zero component at position `j` among the `N1` first entires. The
subsequent `N2` number of components will have `sparsity` number of non-zero
components. 

Arguments
---------
nbr_samples (int): Number of tripples to produce.
delta (float): In the interval (0, 1/4). Changes the entires in A
K (positive int): Ensures that if we change j, then the minimizer x, changes 
                  with an l^2-distance which is at least 10**K.
lam (float): In the interval (0,1]. The regularization parameter in (**). 
j (int): In the range [1,N1]. Index of the non-zero component in x.
n (positive int): If you change j, the matrices |A-A'| <= 2**(-n).
N1 (int): N1 >= 2. Dimension of x is N1+N2.
N2 (int): N1 >= 0. Dimension of x is N1+N2. Only N2 > 1 have been tested.
use_dct (bool): Whether or not to multiply the sparse vector with a DCT matrix.
set_seed (bool): If true, a np.random.seed will be set, so that the same dataset is produce every time.

Returns
-------
`nbr_samples` of triples (y,A,x). 
    """

    assert N2 >=sparsity, 'Sparsity can not exceed vector length'

    np.random.seed(5)
    # Compute A
    # Compute gamma1
    gamma1 = np.sqrt(2)*(1-delta)*(10**K)/(3*lam)
    # Initialize a row
    a = gamma1*lam*(1/(1-delta))*np.ones(N1);
    
    # Perturb entry j with 2**(-n-1), so that the minimizer is non-zero only at index j
    eps = 2**(-n-1)
    rho = 1-delta
    t = (eps/(gamma1*lam))*rho*rho/(1+(eps/(gamma1*lam))*rho)
    rho_min = 1-delta-t
    a[j-1] = gamma1*lam/rho_min
    

    na = np.linalg.norm(a, 2);
    A = np.zeros([N2+1, N1+N2]);
    A[0,0:N1] = a
    A[1:,N1:] = na*np.eye(N2);

    x = np.zeros([N1+N2, nbr_samples]);
    x[j-1, :] = rho_min/(lam*gamma1)
    
    content = np.random.randn(sparsity, nbr_samples)/na
    for i in range(nbr_samples):
        perm = N1 + np.random.permutation(N2);
        x[perm[:sparsity], i] = content[:, i];

    y = np.ones([1+N2, nbr_samples]);
    y[1:,:] = na*x[N1:N1+N2, :].copy()

    if use_dct:
        y = scipy.fftpack.dct(y, axis=0, norm='ortho')
        A = scipy.fftpack.dct(A, axis=0, norm='ortho')

    return y, A, x   

if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000, 
        formatter=dict(float=lambda x: "%.3g" % x))

    K = 3
    n = 10
    delta = 1/6
    lam = 1
    N1 = 2
    N2 = 5
    nbr_samples = 1
    sparsity=2
    j = 1
    
    ys1,A1,xs1 = generate_training_omega(nbr_samples, delta, K, lam, j, n, N1, N2=N2, sparsity=sparsity, use_dct=True, set_seed=True)
    ys2,A2,xs2 = generate_training_omega(nbr_samples, delta, K, lam, j+1, n, N1, N2=N2, sparsity=sparsity, use_dct=True, set_seed=True)
    
    
    print('|ys1-ys2|: ', np.linalg.norm(ys1-ys2, 2), 2**(-n))
    print('|A1-A2|: ', np.linalg.norm(A1-A2, 2), 2**(-n))
    print('|xs1-xs2|: ', np.linalg.norm(xs1-xs2, 2), 10**(-K))

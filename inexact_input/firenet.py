"""This module contains an implementaion of FIRENET"""
import numpy as np

def firenet(y, opA, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta, n_iter, store_hist=False):
    """
The FIRENET.

Agruments
---------
y (np.ndarry): Input data
opA (np.ndarray): Model matrix
p_iter (int): Number of inner iterations
tau (float): tau*sigma*|opA|^2 < 1, where |opA| is the operator norm, induced 
             by the l^2 norm.
sigma (float): tau*sigma*|opA|^2 < 1, where |opA| is the operator norm, induced 
             by the l^2 norm.
lam (float): Regularization paramter for the optimization problem
             min_z lam*|z|_{weights, 1} + |opA*z-y|_2
weights (np.ndarray): Weigths used in the weigthed l^1 norm
L_A (np.ndarray): operator norm of the matrix opA. Call np.linalg.norm(opA, 2).
delta (float): Algorithm parameter
n_iter (int): Number of outer iterations.
store_hist (bool): Store all the iterates x_k. 

Returns
-------
x_final (np.ndarray): The output from FIRENET
all_iterations (list of np.ndarray): All the inner iterates x_k.
"""
    y = np.squeeze(y)
    weights = np.squeeze(weights)

    psi = np.zeros(np.squeeze(np.dot(opA.T, y)).shape)
    eps = eps_0
    all_iterations = []

    for k in range(n_iter):
        eps = np.exp(-1)*(delta + eps)
        beta = eps/(2*L_A)
        al = 1/(beta*p_iter)

        psi_out, itr_list = inner_it(al*y, al*psi, opA, p_iter, sigma, tau, lam, weights, al, store_hist);

        psi = psi_out/al;
        all_iterations.append(itr_list); 


    x_final = psi;

    return x_final, all_iterations

def inner_it(y, x0, opA, p_iter, tau, sigma, lam, weights, al=1, store_hist=False):

    xp = x0;
    yp = np.zeros(y.shape);
    x_sum = np.zeros(xp.shape); 

    all_iterations = [];

    for k in range(1,p_iter+1):

        xpp = cp_prox_varphi(xp - tau*np.dot(np.conj(opA.T), yp), tau*lam, weights);
        ypp = cp_prox_psi1( yp + sigma*np.dot(opA, 2*xpp - xp) - sigma*y );

        x_sum += xpp;

        if store_hist:
            all_iterations.append(x_sum.copy()/(al*k));

        xp = xpp;
        yp = ypp;

    psi_out = x_sum/p_iter;

    return psi_out, all_iterations


def cp_prox_varphi(x, s, weights):

    x1 = np.abs(x.copy())-s*weights;
    x1[ x1 <= 0 ] = 0;
    x_out = x1*np.sign(x);

    return x_out


def cp_prox_psi1(y):

    n_y = np.linalg.norm(y, 2) + 1e-43;
    y *= min(1, 1/n_y);

    return y

if __name__ == "__main__":
    K = 6
    delta = 1/6
    lam = 1
    N1 = 2
    N2 = 20
    sparsity = 5
    j = 2
    n = 30
    use_dct = True
    set_seed = True
    
    # Training
    train_size = 8000
    
    yn, An, xn = generate_test_sample(delta, K, lam, j, n, N1, N2, use_dct=use_dct)
    y, A, x_star = generate_test_sample(delta, K, lam, j-1, n, N1, N2, use_dct=use_dct)
    
    y = np.squeeze(y)
    yn = np.squeeze(yn)
    xn = np.squeeze(xn)
    x_star = np.squeeze(x_star)
    
    print_A_y_dist(y,yn, A, An, n)
    
    
    # FIRENET parameters
    p_iter = 50;
    n_iter = 30;
    weights = np.ones(N1+N2);
    delta1 = 1e-6;
    store_hist = False;
    
    
    eps_0 = np.linalg.norm(yn,2)
    print('eps_0: ', eps_0)
    L_A = np.linalg.norm(An, 2)
    tau = np.sqrt((1/L_A))-1e-4
    sigma = np.sqrt((1/L_A))-1e-4
    x0 = np.zeros(xn.shape);
    print('x0.shape: ', x0.shape)
    print('y.shape: ', y.shape)
    x_final, _ = firenet(yn, An, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta1, n_iter, store_hist)
    
    objective = lambda y, A, x: lam*np.linalg.norm(x,1) + np.linalg.norm(np.dot(A,x)-y, 2) 
    
    print('objective(yn, An, x_final): ', objective(yn, An, x_final))
    print('objective(yn, An, xn):    ', objective(yn,An,xn))
    print('objective(y, A, x_final): ', objective(y, A, x_final))
    print('objective(y, A, x_star):  ', objective(y, A, x_star))
    print('objective(y, An, x_star): ', objective(y, An, x_star))
    print('x_final.shape: ', x_final.shape)
    
    print_array_elements(x_final, x_star)
    print_dist_between_minimizers(x_final, x_star, K)



% This function computes the FIRENET iterations.
% It is based on an exponentially convergent algorithm for the 
% square-root (SR) LASSO optimization problem.
%
% Arguments
% ---------
% y (vector) - Measurement vector.
% x0 (vector) - Initial guess of x. To get FIRENET iterations use `x0 = zeros([N*N,1]);`.
% opA (function handle) - The sampling operator. opA(x,1) is the forward 
%                         transform, and op(y,0) is the adjoint.
% p_iter (int) - Number of inner iterations.
% tau (double) - Algorithm parameter.
% sigma (double) - Algorithm parameter.
% lam (double) - The lambda parameter in the SR-LASSO problem i.e., lam*|z|_1 + |Az-y|_2.
% weights (vector) - Vector of length N*N with the weights for the weighted l^1-norm. 
% L_A (double) - Algorithm parameter.
% eps_0 (double) - Algorithm parameter.
% delta (double) - Algorithm parameter.
% n_iter (int) - Number of outer iterations.
% store_hist (logical) - Whether or not to store all the iterates computed along the way.
%
% Returns
% -------
% x_final (vector) - Reconstructed coefficients.  
% all_iterations (cell) - If store_hist = 1, this is a cell array with all the 
%                         iterates, otherwise it is an empty cell array
%
function [x_final, all_iterations] = cp_expo_sr_lasso(y, x0, opA, p_iter, tau, ...
                                                      sigma, lam, weights, L_A, ...
                                                      eps_0, delta, n_iter, ...
                                                      store_hist)

    if nargin <= 12
        store_hist = logical(0); 
    end


    psi = zeros(size(x0));
    eps = eps_0;
    all_iterations = cell([n_iter,1]);
    
    for k = 1:n_iter
        eps = exp(-1)*(delta + eps);
        beta = eps/(2*L_A);
        al = 1/(beta*p_iter);

        [psi_out, cell_inner_it] = cp_ergodic_sr_lasso(al*y, al*psi, opA, ...
                                                       p_iter, sigma, tau, ...
                                                       lam, weights, al, 1);

        psi = psi_out/al;
        all_iterations{k} = cell_inner_it; 

    end

    x_final = psi;

end



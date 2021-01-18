% Comptes the inner iterations in the FIRENET
% It is based on an linearly convergent algorithm for the 
% square-root (SR) LASSO optimization problem.
%
% Arguments
% ---------
% y (vector) - Measurement vector.
% x0 (vector) - Initial guess of x. 
% opA (function handle) - The sampling operator. opA(x,1) is the forward 
%                         transform, and op(y,0) is the adjoint.
% p_iter (int) - Number of inner iterations.
% tau (double) - Algorithm parameter.
% sigma (double) - Algorithm parameter.
% lam (double) - The lambda parameter in the SR-LASSO problem i.e., lam*|z|_1 + |Az-y|_2.
% weights (vector) - Vector of length N*N with the weights for the weighted l^1-norm. 
% al (double) - FIRENET parameter. Set to 1 of you are not using FIRENET.
% store_hist (logical) - Whether or not to store all the iterates computed along the way.
%
% Returns
% -------
% x_final (vector) - Reconstructed coefficients.  
% all_iterations (cell) - If store_hist = 1, this is a cell array with all the 
%                         iterates, otherwise it is an empty cell array
%
function [psi_out, all_iterations]  = cp_ergodic_sr_lasso(y, x0, opA, p_iter, tau, sigma, lam, weights, al, store_hist)

    if nargin <= 8
        al = 1; 
    end

    if nargin <= 9
        store_hist = logical(0); 
        all_iterations = -1;
    end

    xp = x0;
    yp = zeros(size(y));
    x_sum = zeros(size(xp)); % Obs! we are not summing x0

    all_iterations = cell([p_iter,1]);

    for k = 1:p_iter

        xpp = cp_prox_varphi(xp - tau*opA(yp, 0), tau*lam, weights);
        ypp = cp_prox_psi1( yp + sigma*opA(2*xpp - xp , 1) - sigma*y );

        x_sum = x_sum + xpp;

        if  store_hist
            all_iterations{k}=x_sum/(al*k);
        end

        xp = xpp;
        yp = ypp;

    end

    psi_out = x_sum/p_iter;

end

% Proximal map of the indicator function taking the value 0 on a set S and
% +infty for elements outside S. This function consider the set B_1(0), i.e., the
% unit l^2-ball centered in 0.
function y_out = cp_prox_psi1(y)
    
    n_y = norm(y,2) + 1e-43;
    y_out = min(1,1/n_y)*y;

end


% Proximal map of the weighted l^1-norm.
function x_out = cp_prox_varphi(x, s, weights)

    x_out = max(zeros(size(x)), abs(x)-s*weights).*sign(x);

end

function weights = generate_sparsity_weights(N, sparsity)
% This function computes a `weights` vector with the weights sqrt(s/s_j) om each 
% level j. Here, s = s_1 + ... + s_r, and `s_i = sparsity(i)`. The different 
% wavelet levels are ordered according to the ordering produced by the function 
% `wavedec2` using the 'per' dwtmode.
% 
% Arguments
% ---------
% N (int): length of signal, i.e., prod(size(image)).
% sparsity (vector): local sparsities. Coarsest level first.
%
% Returns
% -------
% weights (vector): Vector with weights sqrt(s/s_j) in each level
%

    K = round(sqrt(N));
    s = sum(sparsity(:));
    nres = length(sparsity);
    weights = zeros([N,1]);

    div_factor = 2;
    end_of_level = N;
    dist = 0;
    sp_flipped = flip(sparsity);

    if nres > 1
        for i = 1:nres-1;
            level_size = 3*(K/div_factor)^2;
            weights(end_of_level-level_size+1:end_of_level) = sqrt(s/sp_flipped(i)); 
            
            end_of_level = end_of_level - level_size;
            div_factor = 2*div_factor;
        end
    end

    i = nres;
    weights(1:end_of_level) = sqrt(s/sp_flipped(i)); 

end

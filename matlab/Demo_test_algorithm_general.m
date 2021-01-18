% This scipts readt the images and perturbations created by the script Demo_test_firenet_stability_general.py and computes an image reconstruction with the same algorithm. It verifies that the Matlab and Tensorflow implementations are computing the same. 
 

dwtmode('per', 'nodisp');

src_data = '/home/vegant/storage_firenet';

exp_nbr = 2;  % Experiment number

src = sprintf('../data_lasso_general/c%03d', exp_nbr);
cgf = yaml.ReadYaml(fullfile(src, 'config_general.yml'));
im_nbr = cgf.DATA.im_nbr;

fprintf('---- Reading exp: %d, im_nbr: %d ----\n', exp_nbr, im_nbr);

fname = sprintf('im_im_nbr_%d_experi_%03d.mat', im_nbr, exp_nbr);

% Load tensorflow data
load( fullfile(src, fname) ); % image, image_rec, rr, im_adjoint

tf_im_adj = im_adjoint;
tf_im_rec = image_rec;

alg_name = cgf.CS_param.alg_name;
lam = cgf.CS_param.lam;
tau = cgf.CS_param.tau;
sigma = cgf.CS_param.sigma;
n_iter = cgf.CS_param.n_iter;
p_iter = cgf.CS_param.p_iter;
wname = cgf.CS_param.wavelet_name;
nres = cgf.CS_param.levels;
delta = cgf.CS_param.delta;
initial_x_zero = cgf.CS_param.initial_x_zero;
use_weights = cgf.CS_param.use_weights;
sparsity_levels = cgf.CS_param.weights_param.sparsity_levels;

N = cgf.DATA.N;
HCP_nbr = cgf.DATA.HCP_nbr;
im_nbr = cgf.DATA.im_nbr;
max_norm = cgf.DATA.max_norm;
srate = cgf.DATA.srate;
runner_id = cgf.DATA.runner_id;
dest_data = cgf.DATA.dest_data;
dest_plots = cgf.DATA.dest_plots;

if use_weights 
    
    sparsity = zeros(size(sparsity_levels));
    for i = 1:length(sparsity)
        sparsity(i) = sparsity_levels{i};
    end   

    weights = generate_sparsity_weights(N*N, sparsity);    
else
    weights = ones(N*N, 1);
end

% Read sampling mask
fname_mask = sprintf('spf2_DAS_N_%d_srate_%d_db1.png', N, round(100*srate));
samp = imread(fullfile(src_data, 'samp_patt', fname_mask));
X = zeros(size(samp));
X(samp > 0)  = 1;
samp = X; clear X;
idx = find(samp);

% Sample perturbed image, 1 is forward map, 0 is adjoint
y = cil_op_fourier_2d(image(:)+rr(:), 1, N, idx);
y_noiseless = cil_op_fourier_2d(image(:), 1, N, idx);

im_adj1 = cil_op_fourier_2d(y, 0, N, idx);
im_adj = reshape(im_adj1, [N,N]);

if initial_x_zero
    x0 = zeros(size(im_adj1));
else
    x0 = im_adj1;
end

fprintf('|x0|: %g \n', norm(x0))

L_A = 1;
eps_0 = norm(y_noiseless, 2);

opA = @(x, mode) cil_op_fourier_wavelet_2d(x, mode, N, idx, nres, wname);
rec_wcoeff = cp_expo_sr_lasso(y, x0, opA, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta, n_iter);

% invese wavelet transform
im_rec = cil_op_wavelet_2d(rec_wcoeff, 0, N, nres, wname);
im_rec = reshape(im_rec, N,N);

fprintf('|im_rec - tf_im_rec|: %g\n', norm(im_rec-tf_im_rec, 'fro'))
fprintf('|im_rec|: %g, max(im_rec): %g\n', norm(im_rec, 'fro'), max(im_rec(:)))
fprintf('|tf_im_rec|: %g, max(tf_im_rec): %g\n', norm(tf_im_rec, 'fro'), max(tf_im_rec(:)))
fprintf('Rel err: %g\n', norm(im_rec-tf_im_rec, 'fro')/norm(im_rec, 'fro'))

%figure();
%subplot(231); imagesc(abs(image+rr)); colormap('gray'); title('x+r'); colorbar(); 
%subplot(232); imagesc(abs(tf_im_adj)); colormap('gray'); title('A^*(Ax+r))'); colorbar(); 
%subplot(233); imagesc(abs(tf_im_rec)); colormap('gray'); title('TF rec'); colorbar(); 
%subplot(234); imagesc(image); colormap('gray'); title('x'); colorbar(); 
%subplot(235); imagesc(abs(im_adj)); colormap('gray'); title('A^*(A(x+r))'); colorbar(); 
%subplot(236); imagesc(abs(im_rec)); colormap('gray'); title('im rec matlab'); colorbar(); 




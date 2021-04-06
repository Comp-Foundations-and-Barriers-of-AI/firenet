% This script reconstructs an image with the text "Can u see it?" using FIRENET.
clear
close all;
dwtmode('per', 'nodisp'); % Use periodic wavelets

%% Read the image
src_patt = '/mn/sarpanitu/ansatte-u4/vegarant/storage_firenet';
src_data = '/mn/sarpanitu/ansatte-u4/vegarant/storage_firenet';
dest = 'plots';
N =  1024;
id = 96;
int = 150;
fname_core = sprintf('sample_N_%d_id_%d_int_%d', N, id, int);
fname = sprintf('%s.png', fname_core);
im = double(imread(fullfile(src_data, 'images', fname)))/255; % scale to the interval [0,1]

if (exist(dest) ~= 7)
    mkdir(dest);
end


%% Fix the wavelet and sampling modality
vm = 1; % Number of vanishing moments
wname = sprintf('db%d', vm); % Wavelet name
nres = wmaxlev([N,N], wname)-1; % Number of wavelet decompositions
modality = 'fourier'; % 'fourier' or 'walsh'


use_weights = logical(1);

epsilon_threshold = 0.5;
% Estimate local sparsities in the image
local_sparsities = cil_compute_sparsity_of_image(phantom(N), vm, nres, epsilon_threshold);

srate = 0.15
fname_mask = sprintf('spf2_DAS_N_%d_srate_%d_db1.png', N, round(100*srate));
samp = imread(fullfile(src_patt, 'samp_patt', fname_mask));
X = zeros(size(samp));
X(samp > 0)  = 1;
samp = X; clear X;
idx = find(samp);
nbr_samples = length(idx);

if use_weights
    weights = generate_sparsity_weights(N*N, local_sparsities);    
else
    weights = ones(N*N, 1);
end

%% Sample the image, add noise and create the AV^* operator
if strcmpi(modality, 'fourier')
    y = cil_op_fourier_2d(im(:), 1, N, idx);
    opA = @(x, mode) cil_op_fourier_wavelet_2d(x, mode, N, idx, nres, wname);
elseif strcmpi(modality, 'walsh')
    y = cil_op_fastwht_2d(im, 'sequency');
    y = y(idx);
    y = y(:);
    opA = @(x, mode) cil_op_fastwht_wavelet_2d(x, mode, N, idx, nres, wname);
end

y0=y;

%% Parameters for algorithm
lam = 0.00025;
sigma = 1; 
tau = 1; 
L_A = 1;
eps_0 = norm(y,2);
p_iter = 5;
delta = 10^(-9);
n_iter = 5;

%% Apply the firenet 
x0 = zeros([N*N, 1]);
[rec_wcoeff,PSI] = cp_expo_sr_lasso(y, x0, opA, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta, n_iter, 0);
%[x_final,PSI_slow] = sq_lasso_conv(y, opA, tau, sigma, lam, weights, 40, 0,rec_wcoeff);

% Apply inverse wavelet transform on the recovered wavelet coefficents.
im_rec = cil_op_wavelet_2d(rec_wcoeff, 0, N, nres, wname);
im_rec = reshape(im_rec, N,N);


% Plot image reconstruction
bd = 5;
im_rec = abs(im_rec);
im_rec(im_rec > 1) = 1;

im_rec_u = uint8(round(255*im_rec));
fname = sprintf('im_rec_text_firenet_N_%d_id_%d_int_%d_n_%d_p_%d_srate_%d.png',N, id, int, n_iter, p_iter, round(100*srate));
imwrite(im_rec_u, fullfile(dest,fname))




%im_u = uint8(round(255*im));
%im_u(im_u==0) = uint8(1);
%im_rec_u(im_rec_u==0) = uint8(1);
%
%factor = SCALE/8;
%len1 = 150*factor;
%h1 = 120*factor;
%h2 = 80*factor;
%
%dx_1 = h1*factor:h1*factor+len1;
%dx_2 = h2*factor:h2*factor+len1;
%
%fname_im = sprintf('im_with_red_frame_N_%d.%s', N, im_format);
%fname_im_crop = sprintf('im_with_red_frame_N_%d_crop.%s', N, im_format);
%fname_im_rec = sprintf('im_rec_with_red_frame_N_%d_%s_srate_%d_n_%d_p_%d.%s', N, modality, round(100*srate), n_iter, p_iter, im_format);
%fname_im_rec_crop = sprintf('im_rec_with_red_frame_N_%d_%s_srate_%d_n_%d_p_%d_crop.%s', N, modality, round(100*srate), n_iter, p_iter, im_format);
%
%cmap = gray(256);
%cmap(1,:) = [1,0,0]; % All zero elements will be red
%
%% Image with red box
%im1 = im_u;
%im1(h1-bd:h1-1,           h2-bd:h2+len1+bd) = uint8(0);
%im1(h1+len1+1:h1+len1+bd, h2-bd:h2+len1+bd) = uint8(0);
%im1(h1:h1+len1,           h2+len1+1:h2+len1+bd) = uint8(0);
%im1(h1:h1+len1,           h2-bd:h2-1) = uint8(0);
%
%imwrite(im1, cmap, fullfile(dest, fname_im));
%
%% Cropped image 
%im1 = im_u;
%im1 = im1(dx_1,dx_2);
%
%imwrite(im1, cmap, fullfile(dest, fname_im_crop));
%
%% Reconstructed image with red box
%im1 = im_rec_u;
%im1(h1-bd:h1-1,           h2-bd:h2+len1+bd) = uint8(0);
%im1(h1+len1+1:h1+len1+bd, h2-bd:h2+len1+bd) = uint8(0);
%im1(h1:h1+len1,           h2+len1+1:h2+len1+bd) = uint8(0);
%im1(h1:h1+len1,           h2-bd:h2-1) = uint8(0);
%
%imwrite(im1, cmap, fullfile(dest, fname_im_rec));
%
%% Cropped reconstructed image 
%im1 = im_rec_u;
%im1 = im1(dx_1,dx_2);
%
%imwrite(im1, cmap, fullfile(dest, fname_im_rec_crop));



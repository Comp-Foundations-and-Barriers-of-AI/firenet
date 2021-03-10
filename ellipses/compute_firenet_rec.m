% exp_id, database, and src_metric_data are global variables which need to be set before running this script

addpath('../matlab')
dwtmode('per', 'nodisp')

dest = 'plots';

fname_yaml = 'matlab_config.yml'; 

cgf = yaml.ReadYaml(fname_yaml);

path_data = cgf.DATA.path_data;
path_pattern = cgf.DATA.path_pattern;
srate = cgf.DATA.srate;
sample_nbr = cgf.DATA.sample_nbr;
intensity_diff = cgf.DATA.intensity_diff;

n_iter = cgf.FIRENET.n_iter;
p_iter = cgf.FIRENET.p_iter;
lam    = cgf.FIRENET.lam;
vm     = cgf.FIRENET.vm;
delta  = cgf.FIRENET.delta;

N          = cgf.SAMP_PATT.N;
is_fourier = cgf.SAMP_PATT.is_fourier;
name       = cgf.SAMP_PATT.name;
vm_patt    = cgf.SAMP_PATT.vm;

type_data = 'test';
folder_data = sprintf('raw_data_%d_TF_tumor', N);
src_images = fullfile(path_data, folder_data, type_data);

im_name1 = sprintf('sample_%d_clean.png', sample_nbr);
im_name2 = sprintf('sample_%d_tumor_%d.png', sample_nbr, intensity_diff);

im1 = double(imread(fullfile(src_images, im_name1)))/255;
im2 = double(imread(fullfile(src_images, im_name2)))/255;

fname_mask = sprintf('spf2_DAS_N_%d_srate_%d_db%d.png', N, round(100*srate), vm_patt);
samp = imread(fullfile(path_pattern, fname_mask));
X = zeros(size(samp));
X(samp > 0)  = 1;
samp = X; clear X;
idx = find(samp);
nbr_samples = length(idx);

wname = sprintf('db%d', vm);
nres = wmaxlev([N,N], wname);

sigma = 0.6; 
tau = 0.6; 
L_A = 1;
x0 = zeros([N*N, 1]);
store_hist = logical(0);

weights = ones(N*N, 1);

im_types = {'clean', sprintf('tumor_%d', intensity_diff)};
images = {im1, im2};
for i = 1:length(images)

    im = images{i};
    im_type = im_types{i};

    if is_fourier
        measurements = cil_op_fourier_2d(im(:), 1, N, idx);
        opA = @(x, mode) cil_op_fourier_wavelet_2d(x, mode, N, idx, nres, wname);
    else
        measurements = cil_op_fastwht_2d(im, 'sequency');
        measurements = measurements(idx);
        measurements = measurements(:);
        opA = @(x, mode) cil_op_fastwht_wavelet_2d(x, mode, N, idx, nres, wname);
    end

    eps_0 = norm(measurements, 2);

    % Exectue the exponentially convergent algorithm for square root lasso
    [rec_wcoeff, PSI] = cp_expo_sr_lasso(measurements, x0, opA, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta, n_iter, store_hist);

    im_rec = cil_op_wavelet_2d(rec_wcoeff, 0, N, nres, wname);
    im_rec = reshape(im_rec, N,N);
    im_rec = abs(im_rec);
    im_rec(im_rec > 1) = 1;

    fname = sprintf('fnet_rec_n_%d_p_%d_srate_%d_im_%d_%s.png', n_iter,p_iter,round(100*srate), sample_nbr, im_type);
    imwrite(im2uint8(im_rec), fullfile(dest, fname));
end








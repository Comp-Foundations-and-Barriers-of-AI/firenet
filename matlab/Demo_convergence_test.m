% This script examines the convergence rate of the FIRENET. It also stores the
% reconstructed image, along with a croped version of it. 

clear
close all;
dwtmode('per', 'nodisp'); % Use periodic wavelets

%% Read the image
SCALE=8;
src = '/mn/sarpanitu/ansatte-u4/vegarant/storage_firenet';
fname_core = 'kopp3';
N = 128*SCALE; 
fname = sprintf('%s_%d.png', fname_core, N);
im = double(imread(fullfile(src, 'images', fname)))/255; % scale to the interval [0,1]

%% Fix the wavelet and sampling modality
vm = 1; % Number of vanishing moments
wname = sprintf('db%d', vm); % Wavelet name
nres = wmaxlev([N,N], wname)-1; % Number of wavelet decompositions
modality = 'fourier'; % 'fourier' or 'walsh'
srate = 0.15; % Subsampling rate
nbr_samples = round(N*N*srate); % Number of samples
use_weights = logical(1);

epsilon_threshold = 0.5;
% Estimate local sparsities in the image
local_sparsities = cil_compute_sparsity_of_image(phantom(N), vm, nres, epsilon_threshold);

if strcmpi(modality, 'fourier')
    [idx, str_id] = cil_spf2_DAS(N, nbr_samples, local_sparsities, vm);
elseif strcmpi(modality, 'walsh')
    [idx, str_id] = cil_sph2_DAS(N, nbr_samples, local_sparsities);
else
    disp('Unknown modality');
end

% If you want to look at the sampling pattern, run the lines below;
% Z = zeros([N,N]);
% Z(idx) = 1;
% figure; imagesc(Z); colormap('gray');

if use_weights
    weights = generate_sparsity_weights(N*N, local_sparsities);    
else
    weights = ones(N*N, 1);
end

%% Sample the image, add noise and create the AV^* operator
if strcmpi(modality, 'fourier')
    y = cil_op_fourier_2d(im(:), 1, N, idx);
    opA = @(x, mode) cil_op_fourier_wavelet_2d(x, mode, N, idx, nres, wname);
    noise_v=randn(size(y))+randn(size(y))*1i;
elseif strcmpi(modality, 'walsh')
    y = cil_op_fastwht_2d(im, 'sequency');
    y = y(idx);
    y = y(:);
    opA = @(x, mode) cil_op_fastwht_wavelet_2d(x, mode, N, idx, nres, wname);
    noise_v=randn(size(y));
end

y0=y;
noise_v=noise_v/norm(noise_v(:));
y=y0+noise_v*norm(y(:))*0.02; % add the noise

%% Parameters for algorithm
lam = 0.00025;
sigma = 1; 
tau = 1; 
L_A = 1;
eps_0 = norm(y,2);
p_iter = 5;
delta = 10^(-9);
n_iter = 16;

%% Exectue the exponentially convergent algorithm for square root lasso
x0 = zeros([N*N, 1]);
[rec_wcoeff,PSI] = cp_expo_sr_lasso(y, x0, opA, p_iter, tau, sigma, lam, weights, L_A, eps_0, delta, n_iter, 1);
[x_final, ~] = cp_ergodic_sr_lasso(y, rec_wcoeff, opA, 10000, tau, sigma, lam, weights, 1, 0); % WARNING: the slow non restarted methdo requires several thousand iterates to compute F^*, decrease for reduced time

% Apply inverse wavelet transform on the recovered wavelet coefficents.
im_rec = cil_op_wavelet_2d(rec_wcoeff, 0, N, nres, wname);
im_rec = reshape(im_rec, N,N);
im_rec_slow = cil_op_wavelet_2d(x_final, 0, N, nres, wname);
im_rec_slow = reshape(im_rec_slow, N,N);

% Compute the errors for plot
ct=1;
E1=zeros(n_iter*p_iter,1);
E2=zeros(n_iter*p_iter,1);
for j=1:round(n_iter)
    for jj=1:p_iter
        psi = cil_op_wavelet_2d(PSI{j}{jj}, 0, N, nres, wname);
        psi = reshape(psi, N,N);
        
        % Relative error of the reconstruction in the image domain
        E1(ct)=norm(im(:)-psi(:))/norm(im(:));

        % Difference in the objective function, divided by the norm of the image
        E2(ct)=(lam*norm((PSI{j}{jj}(:)).*weights(:),1)-lam*norm((x_final(:)).*weights(:),1)+norm(opA(PSI{j}{jj}, 1)-y)-norm(opA(x_final, 1)-y))/norm(im(:));
        ct=ct+1;
    end
end
plot_format = 'epsc';
im_format = 'png';
dest = 'plots';
disp_plots = 'on';


if (exist(dest) ~= 7) 
    mkdir(dest);
end


%% Plot the convergence
fig = figure('visible', disp_plots);
semilogy(1:length(E1),E1,'linewidth',2)
hold on
semilogy(1:length(E2),E2,'linewidth',2)
min_error=norm(im(:)-im_rec_slow(:));
err = min_error / norm(im(:));
hold on
semilogy(1:length(E1),err+zeros(1,length(E1)),'--k')
legend({'$\|c_j-c\|_{l^2}/\|c\|_{l^2}$', '$(F(c_j)-F^*)/\|c\|_{l^2}$', '$\|c-c^*\|_{l^2}/\|c\|_{l^2}$'},'interpreter',...
    'latex','fontsize',14,'location','northeast')
ylabel('Error','interpreter','latex','fontsize',14)
xlabel('Total Inner Iterations ($j$)','interpreter','latex','fontsize',14)
axis([0,60,10^(-4),1])
fname = sprintf('conv_rate_N_%d_%s_srate_%d_n_%d_p_%d', N, modality, round(100*srate), n_iter, p_iter);
saveas(fig, fullfile(dest,fname), plot_format);

%% Plot image reconstruction
fig = figure('visible', disp_plots);
imagesc(im); colormap('gray');caxis([0,1]);axis equal;axis off
hold on
plot(round((80+0*(120:270))*SCALE/8),round((120:270)*SCALE/8),'r','linewidth',1)
plot(round((230+0*(120:270))*SCALE/8),round((120:270)*SCALE/8),'r','linewidth',1)
plot(round((80:230)*SCALE/8),round((0*(80:230)+120)*SCALE/8),'r','linewidth',1)
plot(round((80:230)*SCALE/8),round((0*(80:230)+270)*SCALE/8),'r','linewidth',1)
fname = sprintf('image_N_%d_%s_srate_%d_n_%d_p_%d', N, modality, round(100*srate), n_iter, p_iter);
saveas(fig, fullfile(dest,fname), plot_format);

fig = figure('visible', disp_plots);
imagesc(abs(im_rec));
colormap('gray'); caxis([0,1]);axis equal;axis off
hold on
plot(round((80+0*(120:270))*SCALE/8),round((120:270)*SCALE/8),'r','linewidth',1)
plot(round((230+0*(120:270))*SCALE/8),round((120:270)*SCALE/8),'r','linewidth',1)
plot(round((80:230)*SCALE/8),round((0*(80:230)+120)*SCALE/8),'r','linewidth',1)
plot(round((80:230)*SCALE/8),round((0*(80:230)+270)*SCALE/8),'r','linewidth',1)
fname = sprintf('reconstructed_image_N_%d_%s_srate_%d_n_%d_p_%d', N, modality, round(100*srate), n_iter, p_iter);
saveas(fig, fullfile(dest,fname), plot_format);

fig = figure('visible', disp_plots);
xx=round((120:270)*SCALE/8); yy=round((80:230)*SCALE/8);
imagesc(im(min(xx):max(xx),min(yy):max(yy))); colormap('gray');caxis([0,1]);axis equal;axis off
fname = sprintf('image_zoom_N_%d_%s_srate_%d_n_%d_p_%d', N, modality, round(100*srate), n_iter, p_iter);
saveas(fig, fullfile(dest,fname), plot_format);

fig = figure('visible', disp_plots);
imagesc(abs(im_rec(min(xx):max(xx),min(yy):max(yy)))); colormap('gray'); caxis([0,1]);axis equal;axis off
fname = sprintf('reconstructed_image_zoom_N_%d_%s_srate_%d_n_%d_p_%d', N, modality, round(100*srate), n_iter, p_iter);
saveas(fig, fullfile(dest,fname), plot_format);

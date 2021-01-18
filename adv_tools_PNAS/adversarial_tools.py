import numpy as np;
import sys;

l2_norm_of_tensor = lambda x: np.sqrt((abs(x)**2).sum());
l1_norm_of_tensor = lambda x: abs(x).sum();
linf_norm_of_tensor = lambda x: np.amax(abs(x));

def cut_to_01(im):
    """ Truncate all values to the interval [0, 1] """
    new_im = im.copy();
    new_im[im < 0.0] = 0;
    new_im[im > 1.0] = 1.0;
    return new_im

def scale_to_01(im):
    """ Scales all array values to the interval [0,1] using an affine map."""
    ma = np.amax(im);
    mi = np.amin(im);
    new_im = im.copy();
    return (new_im-mi)/(ma-mi);


def compute_psnr(rec, ref):
    """
    Computes the PSNR of the recovery `rec` w.r.t. the reference image `ref`. 
    Notice that these two arguments can not be swapped, as it will yield
    different results. 
    
    More precisely PSNR will be computed between the magnitude of the image 
    |rec| and the magnitude of |ref|

    :return: The PSNR value
    """
    mse = np.mean((abs(rec-ref))**2);
    max_I = np.amax(abs(rec));
    return 10*np.log10((max_I*max_I)/mse);


def pertub_SGA(f, dQ, batch, epoch, r0, v0, momentum=0.9, lr=0.01, verbose=True,
               max_r_norm=float('Inf'), max_diff_norm=float('Inf')):
    """  Search for adversarial perturbation using a gradient ascent algorithm.

    For a neural network ``f`` and a ``batch`` of images this function search 
    for an adversarial perturbation for ``f`` using the gradient ascent direction
    with a Nesterov step. Let A be a sampling matrix and define the function
    
    Q(r) = ||f(y + A r) - f(y)||_{2}^{2} - lambda*||r||_{2}^{2}
    
    The function perform ``epoch`` of the following steps 
        v_{k+1} = momentum*v_{k} + lr* Gradient(Q(r))
        r_{k+1} = r_{k} + v_{k+1}

    :param f: Neural Network.
    :param dQ: Gradident of function Q : R^n -> R. 
    :param batch: Batch of images.
    :param epoch: Number of iterations.
    :param r0: Starting perturbation.
    :param v0: Starting direction.
    :param momentum: Momentum.
    :param lr: Learning rate.
    :param verbose: Whether or not to print information.
    :param max_r_norm: Stop iterations if ||r||_{2} > max_r_norm.
    :param max_diff_norm: Stop iterations if ||f(batch+r) -f(batch)||_{2} > max_diff_norm.
    
    :returns: r_final, v_final, str_log_of_iterations.
    """
    if (verbose):
        print('------------------------------------');
        print('Running SGA with paramters:');
        print('Momentum:      %g' % momentum);
        print('Learning rate: %g' % lr);
        print('max_r_norm:    %g' % max_r_norm);
        print('max_diff_norm: %g' % max_diff_norm);
        print('------------------------------------');
    
    
    fx = f(batch);
    i = 1;
    norm_fx_fxr = 0;
    norm_r = 0;
    backlog = '';
    norm_x0 = l2_norm_of_tensor(batch);
    r = r0;
    v = v0;
    while (i <= epoch and norm_fx_fxr < max_diff_norm and norm_r < max_r_norm):
        
        dr  = dQ(batch, r);
        fxr = f(batch+r);
        v = momentum*v + lr*dr;
        r = r + v;
        
        norm_fx_fxr = l2_norm_of_tensor(fx-fxr);
        norm_r = l2_norm_of_tensor(r);
        
        next_str = \
        '%2d: |f(x)-f(x+r)|: %8g, |r|: %8g, |f(x)-f(x+r)|/|r| : %8g, |r|/|x|: %8g' \
        % (i, norm_fx_fxr, norm_r, norm_fx_fxr/norm_r, norm_r/norm_x0);
        
        backlog = backlog + '\n' + next_str;
        if (verbose):
            print(next_str);
        i = i + 1;
    return r, v, backlog;



def pertub_SGA_data_consistent(f, dQ, batch, epoch, r0, v0, extract_null_space,
                               momentum=0.9, lr=0.01, verbose=True,
                               max_r_norm=float('Inf'), max_diff_norm=float('Inf')):

    if (verbose):
        print('------------------------------------');
        print('Running SGA with paramters:');
        print('Momentum:      %g' % momentum);
        print('Learning rate: %g' % lr);
        print('max_r_norm:    %g' % max_r_norm);
        print('max_diff_norm: %g' % max_diff_norm);
        print('------------------------------------');
    
    
    fx = scale_to_01(f(batch));
    print('max(x): ', np.amax(batch), ', min(x): ', np.amin(batch))
    print('|x-f(x)|: ', l2_norm_of_tensor(batch - fx))
    i = 1;
    norm_fx_fxr = 0;
    norm_r = 0;
    backlog = '';
    norm_x0 = l2_norm_of_tensor(batch);
    r = r0;
    v = v0;
    while (i <= epoch and norm_fx_fxr < max_diff_norm and norm_r < max_r_norm):
       
        # Notice that dr always lies in N(A)^{perp}, for Fourier sampling
        dr  = dQ(batch, r);
        fxr = scale_to_01(f(batch+r));
        delta_x_null = extract_null_space(fxr - (batch + r) );

        v = momentum*v + lr*dr;
        r = r + v + 1*delta_x_null;
        
        norm_fx_fxr = l2_norm_of_tensor(fx-fxr);
        norm_xr_fxr = l2_norm_of_tensor(batch + r - fxr);
        norm_r = l2_norm_of_tensor(r);
        
        next_str = \
        '%2d: |f(x)-f(x+r)|: %8g, |r|: %8g, |f(x)-f(x+r)|/|r| : %8g, |r|/|x|: %8g, |x+r - f(x+r)|: %8g, |Δx_null|: %8g' \
        % (i, norm_fx_fxr, norm_r, norm_fx_fxr/norm_r, norm_r/norm_x0, norm_xr_fxr, l2_norm_of_tensor(delta_x_null));

        backlog = backlog + '\n' + next_str;
        if (verbose):
            print(next_str);
        i = i + 1;
    return r, v, backlog;



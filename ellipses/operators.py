from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

def dtype_to_cdtype(dtype):
    """ Convert tf.float32 or tf.float64 to tf.complex64 or tf.complex128
    """
    if dtype == tf.float32:
        cdtype = tf.complex64
    elif dtype == tf.float64:
        cdtype = tf.complex128
    else:
        cdtype = -1;
        print('dtype: ', dtype, ' is not supported');
    return cdtype;

def proj_l2_ball(x, radius, center=0.0):
    """
    Projects the tensor `x`, onto the l^2-ball centered in `center` with radius `radius`. 
    
    Arguments
    ---------
    x (tf.tensor) : Tensor of shape [batch_size, ?,?,?].
    radius (float) : Radius of the l^2-ball
    center (float or tf.Tensor) : Center of the l^2-ball. Empty dimensions will be broadcasted. 
    
    Returns
    -------
    tf.tensor : The projected version of `x`.
    """
    norm_diff = tf.math.sqrt(tf.reduce_sum(tf.pow(tf.math.abs(x-center), 2), axis=(-3,-2,-1), keepdims=True))
    radius = tf.broadcast_to(radius, norm_diff.shape);

    # `fac` is 1 of norm_diff <= radius, otherwise it is radius/norm_diff (elementwise).
    fac = tf.where(norm_diff <= radius, tf.ones_like(norm_diff), radius/norm_diff)

    return fac*x + (1.0-fac)*center;

class LinearOperator(ABC):
    """Abstract class for Linear opeators"""

    @abstractmethod
    def forward(self, x):
        """Computes Ax"""
        pass

    @abstractmethod
    def adjoint(self, x):
        """Computes A* x"""
        pass

    def __call__(self, x, adjoint=False):
        """Convenience method"""
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)

class Fourier_operator(LinearOperator):
    """ A subsampled discrete Fourier transform

    Arguments
    ---------
    samp_patt (bool, np.array): Tensor of dtype bool with shape [height, width, channels]. 
    dtype (tf.dtype): Precision of the operator.
    """

    def __init__(self, samp_patt, dtype=tf.float32):
        """
        Arguments:
            samp_patt: Tensor of dtype bool with shape [height, width, channels] 
            dtype (tf.dtype): Precision of the operator.
        """
        super().__init__()

        self.dtype = dtype
        self.cdtype = dtype_to_cdtype(dtype)
        self.nbr_samples = np.sum(samp_patt);
        self.samp_patt = tf.transpose(samp_patt, [2,0,1]) # [1,channels, height, width]
        self.samp_patt = tf.expand_dims(self.samp_patt, axis=0)
        self.cN = tf.cast(tf.math.sqrt(tf.cast(tf.math.reduce_prod(self.samp_patt.shape), dtype=self.dtype)), dtype=self.cdtype)

    def _to_real(self, z):
        assert z.shape[-1] == 1
        return tf.concat((tf.math.real(z), tf.math.imag(z)), axis=-1);
    
    def _to_complex(self, z):
        assert z.shape[-1] == 2
        return tf.expand_dims( tf.complex(z[..., 0] , z[..., 1]), -1);

    def __call__(self, x, output_real=False, add_noise=False, max_noise_magnitude=10, noise_magnitude_fixed=None):
        return self.forward(x, output_real=output_real, add_noise=add_noise, max_noise_magnitude=max_noise_magnitude, noise_magnitude_fixed=noise_magnitude_fixed);
        

    def forward(self, x, output_real=False, add_noise=False, max_noise_magnitude=10, noise_magnitude_fixed=None):
        """
        Arguments:
        x: Tensor with shape [batch_size, height, width, channels]
        output_real (bool): If true the output will be real valued with the 
                            real and imaginary component stacked along the 
                            channel dimension. Otherwise it will return a 
                            complex valued tensor.
        add_noise (bool): Add noise to the measurements.
        max_noise_magnitude (float > 0): First a number `mag` is drawn from the 
                            uniform distribution on [0, max_noise_magnitude], then 
                            a noise vector `noise` is drawn from a normal distribution.
                            The noise `mag*noise` are added to the measurements.
        noise_magnitude_fixed (float): If this number is not None, then it overrides the 
                             argument `max_noise_magnitude`. If it is a float, then 
                             we draw a vector `noise` from a normal distribution. The noise
                             `noise_magnitude_fixed*noise`, is then added to the measurements.       
        """
        if x.shape[-1] == 2:
            x = self._to_complex(x)
        z = tf.transpose(x, [0,3,1,2]) # [channels, height, width]
        z = tf.cast(z, self.cdtype)
        z = tf.signal.fftshift(tf.signal.fft2d(z), axes=(2,3))/self.cN


        if add_noise:

            batch_size = x.shape[0];
            if noise_magnitude_fixed is None:
                magn = tf.random.uniform([batch_size,1,1,1], minval=0, maxval=max_noise_magnitude, dtype=self.dtype);
            else:
                magn = noise_magnitude_fixed;
            noise_real = magn*tf.random.normal(z.shape, dtype=self.dtype)/np.sqrt(self.nbr_samples)
            noise_imag = magn*tf.random.normal(z.shape, dtype=self.dtype)/np.sqrt(self.nbr_samples)
            noise = tf.complex(noise_real, noise_imag);

            z = z + noise;

        # Subsampling
        z = tf.where(self.samp_patt, z, tf.zeros_like(z))
        z = tf.transpose(z, [0,2,3,1]) # [height, width, channels]
        if output_real:
            z = self._to_real(z)
        return z

    def adjoint(self, x, output_real=False):
        """ Adjoint of the subsampled discrete Fourier transform. 
        
        Arguments
        ---------
        x (tf.Tensor): Tensor with shape [batch_size, height, width, channels]. If 
                       channels == 2, this will be interpreted as a complex 
                       tensor in the two components.  
        output_real (bool): If true the output will be real valued with the 
                            real and imaginary component stacked along the 
                            channel dimension. Otherwise it will return a 
                            complex valued tensor.

        Return
        ------
        adj (tf.Tensor): The adjoint of the measurements. 
        """
        if x.shape[-1] == 2:
            x = self._to_complex(x)
        z = tf.transpose(x, [0,3,1,2]) # [channels, height, width]
        z = tf.where(self.samp_patt, z, tf.zeros_like(z))
        z = tf.cast(z, self.cdtype)
        z = tf.signal.ifft2d(tf.signal.ifftshift(z, axes=(2,3)))*self.cN
        z = tf.transpose(z, [0,2,3,1]) # [height, width, channels]
        if output_real:
            z = self._to_real(z)
        return z



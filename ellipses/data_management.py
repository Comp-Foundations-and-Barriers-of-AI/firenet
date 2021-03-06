"""
This file contains functions for generating the training data, loading the data, 
and sampling patterns. 

The data generating and loading functions are based on code from 
https://github.com/jmaces/robust-nets, by M. Genzel, J. Macdonald, and M. März
"""


import tensorflow as tf
import glob
import os
from os.path import join
import random

import numpy as np
import odl
import shutil

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


# ----- Dataset creation, saving, and loading -----

class Ellipses_dataset(tf.data.Dataset):
    def _generator(filenames):
        for fname in filenames:
            yield np.expand_dims(np.asarray(Image.open(fname), dtype=np.float32)/255, axis=2);
    def __new__(cls, path, nbr_of_elements, N, intensity_diff=None, dtype=tf.float32):
        if dtype == tf.float32:
            np_dtype = np.float32
        elif dtype == tf.float64:
            np_dtype = np.float64
        else:
            np_dtype = np.float64
            print('Unknow dtype')

        filenames = []
        if intensity_diff is None:
            for i in range(nbr_of_elements):
                fname = join(path, f"sample_{i}_clean.png")
                filenames.append(fname)
        else:
            for i in range(nbr_of_elements):
                fname = join(path, f"sample_{i}_clean.png")
                filenames.append(fname)
                fname = join(path, f"sample_{i}_tumor_{intensity_diff}.png")
                filenames.append(fname)
                

        return tf.data.Dataset.from_generator(
                       generator=cls._generator ,
                       output_types=dtype,
                       output_shapes=(N,N,1),
                       #output_signature = tf.TensorSpec(shape = (1,N,N,1), dtype=tf.float32),
                       args=(filenames,)
                      )
    
            



def load_sampling_pattern(fname):
    """ Load the sampling pattern and format it correctly.

    Arguments
    ---------
    fname (str): path to sampling pattern.
    
    Returns
    -------
    pattern (np.array): The sampling pattern. 
    """
    im = np.asarray(Image.open(fname));
    pattern = im > 0;
    pattern = np.expand_dims(pattern, axis=-1);
    # Need to adjust the right dimensions!
    return pattern


def create_iterable_dataset(
    n, set_params, generator, gen_params,
):
    """ Creates training, validation, and test data sets.

    Samples data signals from a data generator and stores them.

    Parameters
    ----------
    n : int
        Dimension of signals x.
    set_params : dictionary
        Must contain values for the following keys:
        path : str
            Directory path for storing the data sets.
        num_train : int
            Number of samples in the training set.
        num_val : int
            Number of samples in the validation set.
        num_test : int
            Number of samples in the validation set.
    generator : callable
        Generator function to create signal samples x. Will be called with
        the signature generator(n, **gen_params).
    gen_params : dictionary
        Additional keyword arguments passed on to the signal generator.
    
    
    This function is copied from the directory https://github.com/jmaces/robust-nets
    """
    N_train, N_val, N_test = [
        set_params[key] for key in ["num_train", "num_val", "num_test"]
    ]

    def _get_signal():
        x, _ = generator(n, **gen_params)
        x = np.uint8(255*x);
        return x

    os.makedirs(os.path.join(set_params["path"], "train"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "val"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "test"), exist_ok=True)

    for idx in tqdm(range(N_train), desc="generating training signals"):
        x = _get_signal();
        fname = os.path.join(set_params["path"], "train", "sample_{}_clean.png".format(idx));
        im = Image.fromarray(x);
        im.save(fname);

    for idx in tqdm(range(N_test), desc="generating training signals"):
        x = _get_signal();
        fname = os.path.join(set_params["path"], "test", "sample_{}_clean.png".format(idx));
        im = Image.fromarray(x);
        im.save(fname);
    
    for idx in tqdm(range(N_val), desc="generating training signals"):
        x = _get_signal();
        fname = os.path.join(set_params["path"], "val", "sample_{}_clean.png".format(idx));
        im = Image.fromarray(x);
        im.save(fname);


def sample_ellipses(
    n,
    c_min=10,
    c_max=20,
    max_axis=0.5,
    min_axis=0.05,
    margin_offset=0.3,
    margin_offset_axis=0.9,
    grad_fac=1.0,
    bias_fac=1.0,
    bias_fac_min=0.0,
    normalize=True,
    n_seed=None,
):
    """ Creates an image of random ellipses.

    Creates a piecewise linear signal of shape n (two-dimensional) with a
    random number of ellipses with zero boundaries.
    The signal is created as a functions in the box [-1,1] x [-1,1].
    The image is generated such that it cannot have negative values.

    Parameters
    ----------
    n : tuple
        Height x width
    c_min : int, optional
        Minimum number of ellipses. (Default 10)
    c_max : int, optional
         Maximum number of ellipses. (Default 20)
    max_axis : double, optional
        Maximum radius of the ellipses. (Default .5, in [0, 1))
    min_axis : double, optional
        Minimal radius of the ellipses. (Default .05, in [0, 1))
    margin_offset : double, optional
        Minimum distance of the center coordinates to the image boundary.
        (Default .3, in (0, 1])
    margin_offset_axis : double, optional
        Offset parameter so that the ellipses to not touch the image boundary.
        (Default .9, in [0, 1))
    grad_fac : double, optional
        Specifies the slope of the random linear piece that is created for each
        ellipse. Set it to 0.0 for constant pieces. (Default 1.0)
    bias_fac : double, optional
        Scalar factor that upscales the bias of the linear/constant pieces of
        each ellipse. Essentially specifies the weights of the ellipsoid
        regions. (Default 1.0)
    bias_fac_min : double, optional
        Lower bound on the bias for the weights. (Default 0.0)
    normalize : bool, optional
        Normalizes the image to the interval [0, 1] (Default True)
    n_seed : int, optional
        Seed for the numpy random number generator for drawing the jump
        positions. Set to `None` for not setting any seed and keeping the
        current state of the random number generator. (Default `None`)

    Returns
    -------
    torch.Tensor
        Will be of shape n (two-dimensional).
    
    This function is copied from the directory https://github.com/jmaces/robust-nets
    """

    if n_seed is not None:
        np.random.seed(n_seed)

    c = np.random.randint(c_min, c_max)

    cen_x = (1 - margin_offset) * 2 * (np.random.rand(c) - 1 / 2)
    cen_y = (1 - margin_offset) * 2 * (np.random.rand(c) - 1 / 2)
    cen_max = np.maximum(np.abs(cen_x), np.abs(cen_y))

    ax_1 = np.minimum(
        min_axis + (max_axis - min_axis) * np.random.rand(c),
        (1 - cen_max) * margin_offset_axis,
    )
    ax_2 = np.minimum(
        min_axis + (max_axis - min_axis) * np.random.rand(c),
        (1 - cen_max) * margin_offset_axis,
    )

    weights = np.ones(c)
    rot = np.pi / 2 * np.random.rand(c)

    p = np.stack([weights, ax_1, ax_2, cen_x, cen_y, rot]).transpose()
    space = odl.discr.discr_sequence_space(n)

    coord_x = np.linspace(-1.0, 1.0, n[0])
    coord_y = np.linspace(-1.0, 1.0, n[1])
    m_x, m_y = np.meshgrid(coord_x, coord_y)

    X = np.zeros(n)
    for e in range(p.shape[0]):
        E = -np.ones(n)
        while E.min() < 0:
            E = odl.phantom.geometric.ellipsoid_phantom(
                space, p[e : (e + 1), :]
            ).asarray()
            E = E * (
                grad_fac * np.random.randn(1) * m_x
                + grad_fac * np.random.randn(1) * m_y
                + bias_fac_min
                + (bias_fac - bias_fac_min) * np.random.rand(1)
            )
        X = X + E


    if normalize:
        ma = np.amax(X)
        X /= ma

    return X, None 


def load_dataset(path, size=None, np_dtype=np.float64, intensity_diff=None):
    """ Load dataset into memory.

    Arguments
    ---------
    path (str): Path to directory with PNGs files.
    size (None/int): If None, it read all images, if it is an int, it only reads the specified number of elements.
    np_dtype (np.dtype): Precision of the loaded dataset

    Returns
    -------
    data (np.array): Dataset with images, scaled to [0,1].
    
    
    This function is based on code from the directory https://github.com/jmaces/robust-nets
    """
    files = glob.glob(os.path.join(path, "*.png"))
    num_samples = len(files)
    
    if num_samples == 0:
        print('Did not find any images at ', path);
    if size is not None:
        if size <= num_samples:
            files = files[:size];
            num_samples = size
        else:
            print(f'Unable to read {size} images, could only retrive {num_samples} images');
    im = np.asarray(Image.open(files[0]), dtype=np_dtype);
    im /= 255;

    data = np.zeros((num_samples,) + im.shape + (1,), dtype=np_dtype);
    data[0,:,:,0] = im;

    for i in tqdm(range(1,num_samples), desc="Loading data"):
        im = np.asarray(Image.open(files[0]), dtype=np_dtype);
        data[i, :, :, 0] = im/255;

    return data

def add_text_to_image(image, font_size, font_file, intensity_diff):
    """Adds the word 'Tumor' to the image at a random location.

    Arguments
    ---------
    image (ndarray): Array of with shape [N,N], and all elements lying in the 
                     interval [0, 1].
    font_size (int): Font size
    font_file (str): Path to the font file. Usually these files has the extension .ttf
    intensity_diff (int): All image pixels are integers in the interval [0, 255]. 
                          We sample a pixel value 'pix_int' close to where the text is placed and add a text with the intensity value `pix_int + intensity_diff`. If `pix_int+intensity_diff > 255`, we use the intensity `pix_int-intensity_diff`.
    
    Returns
    -------
    Return a copy of the input image with the text `Tumor` added in the image. All values of this 
    image lies in the interval [0,1].
"""
    N = image.shape[0]
    pil_image = Image.fromarray(np.uint8(255*image))
    
    pos_x, pos_y = np.random.randint(low=int(N/4), high=int(3*N/4), size=2);
    pos_x = int(pos_x)
    pos_y = int(pos_y)
    pixel_intensity = pil_image.getpixel((pos_x, pos_y))   

    draw = ImageDraw.Draw(pil_image);
    font = ImageFont.truetype(font_file, font_size)
    
    if pixel_intensity + intensity_diff > 255:
        intensity = pixel_intensity - intensity_diff
    else: 
        intensity = pixel_intensity + intensity_diff

    draw.text((pos_x, pos_y), 'Tumor', intensity, font=font)
    return np.asarray(pil_image, 'float64')/255

# ---- run data generation -----
if __name__ == "__main__":
    """ Generate dataset
    """
    N = 512;
    dest = f'/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data_{N}_TF_tumor';

    fname = 'data_management.py'
    #shutil.copyfile(fname, join(dest, fname))

    data_params = {  # additional data generation parameters
    "c_min": 20,
    "c_max": 45,
    "max_axis": 0.25,
    "min_axis": 0.03,
    "margin_offset": 0.3,
    "margin_offset_axis": 0.9,
    "grad_fac": 0.9,
    "bias_fac": 1.0,
    "bias_fac_min": 0.3,
    "normalize": True,
    }

    set_params = {
        "num_train": 5000,
        "num_val": 500,
        "num_test": 500,
        "path": dest,
    }

    numpy_seed = 4;
    np.random.seed(numpy_seed)

    
    ell_im, _ = sample_ellipses([N,N], **data_params);
    print(ell_im)


    font_size = 16;
    font_file = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
    intensity_diff = 30;
    im_with_text = add_text_to_image(ell_im, font_size, font_file, intensity_diff);
    
    data_gen = sample_ellipses  # data generator function
    create_iterable_dataset(
        [N, N], set_params, sample_ellipses, data_params,
    )
    
    print('Starting to add text to images')
    name_dataset = ['train', 'test', 'val'];
    for dir_name in name_dataset:
        path = join(dest, dir_name)
        files = glob.glob(os.path.join(path, "*clean.png"))
        for i in range(len(files)):
            filename = files[i];
            filename_core = filename[:-9]
            filename_with_text = filename_core + f"tumor_{intensity_diff}.png";
            
            # Load image
            im = np.asarray(Image.open(filename), dtype=np.float64)
            im /= 255
            im_with_text = add_text_to_image(im, font_size, font_file, intensity_diff)
            pil_im = Image.fromarray(np.uint8(255*im_with_text));
            pil_im.save(filename_with_text)





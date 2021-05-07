# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import normalize


def generate_gaussian2D(image_shape, sigma_x, sigma_y, mu_x, mu_y):
    """
    Function returning an image of shape image_shape containing a 2D gaussian of unitary amplitude and standard deviations 
    and center passed as parameters.
    
    Args:
        image_shape::[tuple]
            Tuple describing the desired shape of the output image in the form (n_lines, n_columns, n_channels).
        sigma_x::[float]
            The horizontal standard deviation of the 2D gaussian.
        sigma_y::[float]
            The vertical standard deviation of the 2D gaussian.
        mu_x::[float]
            The horizontal centering of the 2D gaussian.
        mu_y::[float]
            The vertical centering of the 2D gaussian.
    Returns:
        gaussian::[np.array]
            Numpy array of shape (n_lines, n_columns, n_channels) containing the desired 2D gaussian.
    
    """
    pixels_range = range(-image_shape[0]//2, image_shape[1]//2)
    x, y = np.meshgrid(pixels_range, pixels_range)
    gaussian_x = np.exp(- (x - mu_x)**2 / (2 * sigma_x**2))
    gaussian_y = np.exp(- (y - mu_y)**2 / (2 * sigma_y**2))
    gaussian = gaussian_x * gaussian_y
    
    return gaussian.reshape(image_shape)


def add_gaussian(image, amplitude, sigma_x=None, sigma_y=None, mu_x=None, mu_y=None, nb_bits=8):
    """
    Function distorting the image by adding a 2D gaussian with amplitude, standard deviations and centering passed as parameter.
    The obtained image is then normalised for it to be in the range [0, 2**nb_bits-1].
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        amplitude::[np.array]
            The amplitude of the gaussian we want to add to the image (before normalization of distorted image).
        sigma_x::[float]
            The horizontal standard deviation of the 2D gaussian. If None, it is computed as one fifth of horizontal image shape.
        sigma_y::[float]
            The vertical standard deviation of the 2D gaussian. If None, it is computed as one fifth of vertical image shape.
        mu_x::[float]
            The horizontal centering of the 2D gaussian. If None, it is chosen at random within the image shape.
        mu_y::[float]
            The vertical centering of the 2D gaussian. If None, it is chosen at random within the image shape.
        nb_bits::[int]
            Normalization range of output image: [0, 2**nb_bits-1].
    Returns:
        new_image::[np.array]
            Numpy array of same shape as image containing the image distorted by addition of desired 2D gaussian.
    
    """
    if sigma_x is None:
        sigma_x = (image.shape[0])/5
    if sigma_y is None:
        sigma_y = (image.shape[1])/5
    if mu_x is None:
        mu_x = np.random.randint(-image.shape[0] // 2, image.shape[0] // 2)
    if mu_y is None:
        mu_y = np.random.randint(-image.shape[1] // 2, image.shape[1] // 2)
        
    gaussian = generate_gaussian2D(image.shape, sigma_x, sigma_y, mu_x, mu_y)
    new_image = gaussian * amplitude + image
    
    return normalize(new_image, nb_bits)


def add_gaussian_noise(image, mean, sigma, nb_bits=8):
    """
    Function distorting the image by adding gaussian noise of desired mean and standard deviation.
    The obtained image is then normalised for it to be in the range [0, 2**nb_bits-1].
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        mean::[float]
            The mean of the gaussian noise.
        sigma::[float]
            The standard deviation of the gaussian noise.
        nb_bits::[int]
            Normalization range of output image: [0, 2**nb_bits-1].
    Returns:
        new_image::[np.array]
            Numpy array of same shape as image containing the image distorted by addition gaussian noise.
    
    """
    gaussian  = np.random.normal(mean, sigma, image.shape)
    new_image = normalize(image + gaussian, nb_bits)
    return new_image


def zoom_image(image, zoom_factor, val_padding=None):
    """
    Function distorting the image by zooming it in towards its center if zoom_factor > 1 or by zooming it out and padding
    with val_padding to keep same shape if 0 < zoom_factor < 1. Cubic interpolation is used to preserve quality. 
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        zoom_factor::[float]
            The zoom factor to apply to the image. If zoom_factor > 1, the image is zoomed in on its center. 
            If 0 < zoom_factor < 1, the image is zoomed out and consecutively padded on the borders to keep same shape.
            If zoom_factor == 1, the original image is returned.
            If zoom_factor < 0, an exception is raised.
        val_padding::[float]
            The value to use to pad the borders of the image when zoom_factor < 1. If None, it is computed as the min
            value found in image.
            
    Returns:
        output::[np.array]
            Numpy array of same shape as image containing the image distorted by zooming in or out.
    
    """
    if zoom_factor <= 0:
        raise ValueError("The zoom factor must be strictly positive")
    
    elif zoom_factor == 1:
        output = image
        
    else:
        # Resize
        new_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        
        height, width, _      = image.shape
        new_height, new_width = new_image.shape
    
        if zoom_factor > 1:
            # Crop central portion of correct size
            line_start = (new_height - height) // 2
            col_start  = (new_width - width) // 2
            line_end   = line_start + height
            col_end    = col_start  + width

            output = new_image[line_start:line_end, col_start:col_end]

        elif zoom_factor < 1:  
            if val_padding is None:
                val_padding = image.min()

            # Padding with val_padding
            pad_up    = (height - new_height) // 2
            pad_down  = height - pad_up - new_height
            pad_left  = (width - new_width) // 2
            pad_right = width - pad_left - new_width

            output = np.pad(new_image, ((pad_up, pad_down), (pad_left, pad_right)), 'constant', constant_values = val_padding)

    return output.reshape(image.shape)


def zoom_image_to_meet_shape(image, shape):
    """
    Function which takes the input image and resizes it to meet desired shape using a cubic interpolation algorithm to
    preserve quality. 
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        shape::[tuple]
            Tuple describing the desired shape of the output image in the form (new_n_lines, new_n_columns, new_n_channels).
    
    Returns:
        new_image::[np.array]
            Numpy array of desired shape containing the image given as input resized via cubic interpolation.
    
    """
    return cv2.resize(image, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC).reshape(shape)
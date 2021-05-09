# -*- coding: utf-8 -*-
import cv2
import numpy as np


def apply_random_distortion_from_range(function, image, params_ranges={}):
    """
    Apply a distortion to the image by the use of function that takes as parameter a randomly chosen value in a given range.
    
    Args:
        function::[function]
            The function to distort the image.
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        params_ranges::[dict]
            The range of parameters to give to function between which we will choose a random value
    Returns:
        distorted_image::[np.array]
            Numpy array containing the distorted image of shape (n_lines, n_columns, n_channels).
    
    """
    random_params = {}
    for param, val_range in params_ranges.items():
        random_params[param] = np.random.uniform(*val_range)

    distorted_image = function(image, **random_params)
    
    return distorted_image


def normalize(image, nb_bits=1):
    """
    Normalize the image on a given number of bits n_bits.

    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        nb_bits::[int]
            The number of bits by which we want to normalize. By default it normalizes the
            image to be in the range [0,1].
    Returns:
        normalized_image::[np.array]
            Numpy array containing the normalized version of the image.
       
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return (2**nb_bits-1)*(image-min_val)/(max_val-min_val)


def binarize(image, thr = None):
    if thr is None:
        thr = image.max() / 2
        
    return (image > thr).astype(np.uint8)


def gaussian_blur(image, sigma_horizontal, sigma_vertical = 0.):
    # If sigma_vertical = 0, opencv puts it equal to sigma_horizontal
    return cv2.GaussianBlur(image, ksize = (0, 0), sigmaX = sigma_horizontal, sigmaY = sigma_vertical)



def add_gaussian_noise(image, mean, sigma, nb_bits = 1):
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


def rotation(image, deg, border_value = None):
    """
    Function returning image rotated by deg degrees counter-clockwise around its center.
    
    Args:
        image::[np.array]
            Image we wish to rotate.
        deg::[float]
            Rotation angle in degrees. Positive values mean counter-clockwise rotation.
        border_value::[float]
            The value to use to pad the borders of the new image which do not correspond to points in the original image.
            If None, it is computed as the min value found in image.
    Returns:
        result::[np.array]
            Image rotated by deg degrees counter-clockwise around its center.
    """
    if border_value is None:
        border_value = image.min()
        
    max_value = image.max()
        
    rows, cols = image.shape
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
    result = cv2.warpAffine(image, matrix, (cols, rows), flags = cv2.INTER_CUBIC, borderValue = border_value)

    return np.clip(result, a_min = 0, a_max = max_value)


def translate(image, dx, dy, border_value = None):
    """
    Function returning image translated by dx pixels horizontally and dy pixels vertically.
    
    Args:
        image::[np.array]
            Image we wish to translate.
        dx::[int]
            Number of pixels we want to translate image horizontally. Positive values mean translation to the right.
        dy::[int]
            Number of pixels we want to translate image vertically. Positive values mean translation downwards.
        border_value::[float]
            The value to use to pad the borders of the new image which do not correspond to points in the original image.
            If None, it is computed as the min value found in image.
    Returns:
        result::[np.array]
            Image translated by dx pixels horizontally and dy pixels vertically.
    """
    if border_value is None:
        border_value = image.min()
        
    rows, cols = image.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    result = cv2.warpAffine(image, matrix, (cols, rows), borderValue = border_value)
    
    return result


def zoom_image(image, zoom_factor, val_padding = None):
    """
    Function distorting the image by zooming it in towards its center if zoom_factor > 1 or by zooming it out and padding
    with val_padding to keep same shape if 0 < zoom_factor < 1. Cubic interpolation is used to preserve quality when 
    zoom_factor > 1 and Area interpolation is used when zoom_factor < 1. 
    
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
        if zoom_factor > 1:
            new_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        
            height, width         = image.shape
            new_height, new_width = new_image.shape
            
            # Crop central portion of correct size
            line_start = (new_height - height) // 2
            col_start  = (new_width - width) // 2
            line_end   = line_start + height
            col_end    = col_start  + width

            output = new_image[line_start:line_end, col_start:col_end]

        elif zoom_factor < 1:
            new_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_AREA)
        
            height, width         = image.shape
            new_height, new_width = new_image.shape
        
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
    interpolation = cv2.INTER_CUBIC if image.shape[0] < shape[0] else cv2.INTER_AREA
    
    return cv2.resize(image, (shape[0], shape[1]), interpolation = interpolation).reshape(shape)
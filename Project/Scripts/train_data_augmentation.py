# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from distortions import add_gaussian, add_gaussian_noise



def make_train_generator(train_images, train_masks, batch_size=1, custom_distortions=False, seed=2):
    """
    Function returning a generator of batches of images and their corresponding masks. Random data augmentation is applied
    to the images and, if needed, the masks are also changed in the same way to guarantee coherence with images. 
    
    Args:
        train_images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the training images.
        train_masks::[np.array]
            Numpy array of same shape as train_images containing the corresponding training masks.
        batch_size::[int]
            The number of images returned for each iteration on the generator.
        custom_distortion::[bool]
            If True, non-uniform illumination (add_gaussian) and gaussian noise (add_gaussian_noise) distortions are
            applied to the images in addition to random rotation and flipping.
        seed::[int]
            Seed for the random number generator. Needed to guarantee that same rotation and flipping is applied to
            images and masks.
    Returns:
        train_generator::[generator]
            Generator that, for every iteration, returns a tuple of (images, masks) where images and masks are np.array
            of size (batch_size, n_lines, n_columns, n_channels).

    """
    if custom_distortions:
        images_datagen = ImageDataGenerator(preprocessing_function=data_augmentation_with_distortions_for_images)
    else:
        images_datagen = ImageDataGenerator(preprocessing_function=data_augmentation_no_distortions)
        
    masks_datagen  = ImageDataGenerator(preprocessing_function=data_augmentation_no_distortions)

    # Provide the same seed and keyword arguments to the flow method so that images and masks transformed the same way.
    images_generator = images_datagen.flow(train_images, batch_size=batch_size, seed=seed)
    masks_generator  = masks_datagen .flow(train_masks , batch_size=batch_size, seed=seed)

    # Combine generators into one which yields image and masks.
    train_generator = zip(images_generator, masks_generator)

    return train_generator



def data_augmentation_no_distortions(image):
    """
    Augment the dataset by randomly rotating by multiples of 90 degrees and randomly flipping.
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
    Returns:
        new_image::[np.array]
            Numpy array containing the new image of shape (n_lines, n_columns, n_channels) randomly rotated by multiples of 90
            degrees and randomly flipped.
    
    """
    random_rotation    = np.random.choice([0,1,2,3])
    possible_flip_axis = np.array([(), (1), (2), (1,2)], dtype=object)
    random_flip_axis   = np.random.choice(possible_flip_axis)

    new_image = np.rot90(image    , random_rotation)
    new_image = np.flip (new_image, random_flip_axis)
    
    return new_image


def data_augmentation_with_distortions_for_images(image):
    """
    Augment the dataset by randomly applying data_augmentation_no_distortions function and additionnaly distorting the image 
    (addition of a 2D gaussian and gaussian noise) randomly within a range of parameters.
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
    Returns:
        new_image::[np.array]
            Numpy array containing the new image of shape (n_lines, n_columns, n_channels) randomly distorted.
    """
    new_image = data_augmentation_no_distortions(image)
    new_image = apply_random_distortion_from_range(add_gaussian, new_image, {"amplitude": (0, 300)})
    new_image = apply_random_distortion_from_range(add_gaussian_noise, new_image, {"mean": (0, 0), "sigma": (0, 30)})

    return new_image


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
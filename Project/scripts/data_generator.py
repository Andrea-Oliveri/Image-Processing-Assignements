# -*- coding: utf-8 -*-
import warnings
import numpy as np
import tensorflow
from .distortions import *


class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, mnist_images, mnist_labels, figures_images, figures_labels, resolution, batch_size, augment_mnist_data = False, 
                 augment_figure_data = False):
        
        # Storing options parameters for later use.
        self.augment_mnist_data  = augment_mnist_data
        self.augment_figure_data = augment_figure_data

        # Storing images and labels.
        self.mnist_images   = mnist_images
        self.mnist_labels   = mnist_labels
        self.figures_images = figures_images
        self.figures_labels = figures_labels
                
        # Storing number of unique classes.
        _, mnist_counts   = np.unique(self.mnist_labels  , return_counts = True)
        _, figures_counts = np.unique(self.figures_labels, return_counts = True)

        self.n_mnist_classes = len(mnist_counts)
        self.n_figures_classes = len(figures_counts)
        self.n_classes = self.n_mnist_classes + self.n_figures_classes
        
        # Store batch size, resolution of desired output (training input) and number of batches per epoch (accounting
        # for class equilibrium in each batch).
        if batch_size % self.n_classes:
            warnings.warn(f"Batch Size must be a multiple of {self.n_classes} to guarantee class equilibrium. " + \
                          f"Rounding to closest multiple larger than asked batch size ({batch_size}).", UserWarning)

            batch_size = int(np.ceil(batch_size / self.n_classes) * self.n_classes)
        
        self.batch_size           = batch_size
        self.n_exemples_each_class_per_batch = self.batch_size // self.n_classes
        self.resolution           = resolution
        self.n_batches_per_epoch  = min(mnist_counts) // self.n_exemples_each_class_per_batch
        
        # Generate shuffled indices for first epoch.
        self.on_epoch_end()

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches_per_epoch

    
    def __getitem__(self, index):
        'Generate one batch of data'
        X = np.empty((self.batch_size, *self.resolution), dtype = np.float32)
        y = np.zeros(self.batch_size, dtype = np.uint8)
                       
        # Add randomly picked MNIST digits to batch, applying data augmentation on them if necessary.
        n_tot_mnist_exemples_per_batch = self.n_exemples_each_class_per_batch * self.n_mnist_classes

        index_mnist_batches = self.random_index_mnist[index * n_tot_mnist_exemples_per_batch:(index + 1) * n_tot_mnist_exemples_per_batch]
        
        i = 0
        for index_mnist in index_mnist_batches:
            mnist_image = self.mnist_images[index_mnist]
            mnist_label = self.mnist_labels[index_mnist]
            
            X[i, :] = self.pretreat_image(mnist_image, self.augment_mnist_data, can_image_be_rotated_180 = False)
            y[i   ] = mnist_label
            
            i += 1
              
        # Add figures to batch, applying data augmentatin on them if necessary.
        for _ in range(self.n_exemples_each_class_per_batch):
            for figure_image, figure_label in zip(self.figures_images, self.figures_labels):
                X[i, :] = self.pretreat_image(figure_image, self.augment_figure_data, can_image_be_rotated_180 = True)
                y[i   ] = figure_label
        
                i += 1
        
        return np.expand_dims(X, axis = -1), y
    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        mnist_classes = np.unique(self.mnist_labels)
        idx_class_mnist = []
        for c in mnist_classes:
            idx_same_class, = np.where(self.mnist_labels == c)
            np.random.shuffle(idx_same_class)
            idx_class_mnist.append(idx_same_class)
            
        # Makes list of list rectangular by removing elements from classes having more than others. 
        min_len = min(map(len, idx_class_mnist))
        idx_class_mnist = [elem[:min_len] for elem in idx_class_mnist]
        
        # Organizes indices of dataset ensuring that in each batch same number of classes is present.
        self.random_index_mnist = np.transpose(idx_class_mnist).flatten()
        
        
    def pretreat_image(self, image, random_augmentation, can_image_be_rotated_180 = False):
        new_image = image
                        
        new_image = zoom_image_to_meet_shape(new_image, self.resolution)

        new_image = normalize(new_image)
        
        new_image = self.random_data_augmentation(new_image, can_image_be_rotated_180) if random_augmentation else new_image

        new_image = binarize(new_image)

        return new_image


        
    def random_data_augmentation(self, image, can_image_be_rotated_180 = False):
        """Randomly performs a data augmentation on the data passed as parameter and returns the new data."""
        new_image = image
        
        #new_image = apply_random_distortion_from_range(add_gaussian_noise, new_image, 
        #                                               {"mean": (0, 0), "sigma": (0, 0.3)})
        
        #new_image = apply_random_distortion_from_range(gaussian_blur, new_image,
        #                                               {"sigma_horizontal": (1e-6, 1e-1), "sigma_vertical": (1e-6, 1e-1)})
        
        new_image = rotate_180(new_image) if can_image_be_rotated_180 and np.random.choice([True, False]) else new_image
        
        new_image = apply_random_distortion_from_range(zoom_image, new_image,
                                                       {"zoom_factor": (0.8, 1)})
        
        new_image = apply_random_distortion_from_range(rotation, new_image, 
                                                       {"deg": (-30, 30)})
        
        new_image = apply_random_distortion_from_range(translate, new_image, 
                                                       {"dx": (-4, 4), "dy": (-4, 4)})
        
        return new_image
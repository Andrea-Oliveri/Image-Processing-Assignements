# -*- coding: utf-8 -*-
import warnings
import numpy as np
import tensorflow 


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
        y = np.zeros((self.batch_size, self.n_classes  ), dtype = np.uint8)
                       
        # Add randomly picked MNIST digits to batch, applying data augmentation on them if necessary.
        n_tot_mnist_exemples_per_batch = self.n_exemples_each_class_per_batch * self.n_mnist_classes

        index_mnist_batches = self.random_index_mnist[index * n_tot_mnist_exemples_per_batch:(index + 1) * n_tot_mnist_exemples_per_batch]
        
        i = 0
        for index_mnist in index_mnist_batches:
            mnist_image = self.mnist_images[index_mnist]
            mnist_label = self.mnist_labels[index_mnist]
                        
            X[i, :] = self.random_data_augmentation(mnist_image) if self.augment_mnist_data else mnist_image
            y[i, mnist_label] = 1
            
            i += 1
        
        
        assert i % 10 == 0 and self.batch_size > i, f"Problem: index i not correct: got {i}, not % 10 and not larger than batch size"

        assert (self.batch_size - i) // 3 == (self.batch_size - i) / 3 == self.n_exemples_each_class_per_batch, f"Problem with // vs /: {(self.batch_size - i) // 3} vs {(self.batch_size - i) / 3}"
        
        import cv2
        
        # Add figures to batch, applying data augmentatin on them if necessary.
        for _ in range(self.n_exemples_each_class_per_batch):
            for figure_image, figure_label in zip(self.figures_images, self.figures_labels):
                X[i, :] = cv2.resize(self.random_data_augmentation(figure_image) if self.augment_figure_data else figure_image, self.resolution)
                y[i, figure_label] = 1
                
                i += 1

        
        
        
        
        assert i == self.batch_size, f"Problem: index i not correct: {self.batch_size} expected, got {i}"
        assert np.all(np.sum(y, axis = 1) == 1), f"Problem: got one line with multiple 1"
        assert np.all(np.sum(y, axis = 0) == np.sum(y, axis = 0)[0]), f"Problem: got one different number of classes per batch"

        
        return np.expand_dims(X / 255, axis = -1), y
    
    
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
        
        
        
        
        test = self.mnist_labels[self.random_index_mnist]
        _, counts = np.unique(test, return_counts = True)
        assert len(counts) == 10 and min(counts) == max(counts), "Error on epoch end"
        
        
    def harvest(self):
        temp_batch_size = self.batch_size
        self.batch_size = len(self.index_measures)
        X, y = self[0]
        self.batch_size = temp_batch_size
        return X, y       
        
        
    def random_data_augmentation(self, x):
        """Randomly performs a data augmentation on the data passed as parameter and returns the new data. For now, the only
        data augmentation which is known should not impact the label is flipping the measure over the snapshot axis (measurement
        time axis) as an activity which is performed in one direction and in the opposite direction should still be the same action
        (for the activities considered in this dataset)."""
        
        return x
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, SpatialDropout2D


def convolutional_network(input_shape = (28, 28, 1), show_summary = True):
    """
    Returns an instance of a tensorflow.keras model implementing a Convolutional Network network.

    Args:
        input_shape::[tuple]
            The shape of the input layer.
        show_summary::[bool]
            if true, a summary showing the different layers in the model is printed.

    Returns:
        model::[tensorflow.keras model]
            Instance of tensorflow.keras model implementing a Convolutional Network architecture.

    """
    model = Sequential([ Input(input_shape), 
                         Conv2D(filters = 32 , kernel_size = 3, padding = 'same' , kernel_initializer = 'he_uniform', activation = 'relu'),
                         MaxPool2D(pool_size = 2),
                         Conv2D(filters = 64 , kernel_size = 3, padding = 'same' , kernel_initializer = 'he_uniform', activation = 'relu'),
                         MaxPool2D(pool_size = 2),
                         Conv2D(filters = 128, kernel_size = 7, padding = 'valid', kernel_initializer = 'he_uniform', activation = 'relu'),
                         SpatialDropout2D(rate = 0.5),
                         Conv2D(filters = 13 , kernel_size = 1, padding = 'valid', activation = 'softmax'),
                         Flatten() ])
    
    if show_summary:
        model.summary()
    
    return model
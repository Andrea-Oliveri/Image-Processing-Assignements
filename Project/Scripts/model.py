# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D


def Unet(input_shape=(28, 28, 1), show_summary=True):
    """
    Returns an instance of a tensorflow.keras model implementing a Fully Convolutional Network network.

    Args:
        input_shape::[tuple]
            The shape of the input layer.
        show_summary::[bool]
            if true, a summary showing the different layers in the model is printed.

    Returns:
        model::[tensorflow.keras model]
            Instance of tensorflow.keras model implementing a Fully Convolutional Network architecture.

    """
    raise NotImplementedError
    
    if show_summary:
        model.summary()
    
    return model
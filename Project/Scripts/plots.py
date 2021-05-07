# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Plots the evolution of accuracy and loss over the epochs as collected in history dictionary.
    
    Args:
        history::[dict]
            Dictionary of form {metric: metric_val_list, ...}
            
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
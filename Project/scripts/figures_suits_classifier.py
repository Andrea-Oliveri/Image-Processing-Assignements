import cv2
import numpy as np
import tensorflow as tf
import sklearn
import os
import pickle

from .extract import get_color_pixels
from .distortions import *


class FiguresSuitsClassifier():
    
    def __init__(self, 
                 figure_classifier_save_path = os.path.join("Models", "Convolutional Network"), 
                 suits_classifier_save_path  = os.path.join("Models", "suits_classifier.pkl"), 
                 figure_classifier_input_resolution = (28, 28)):
        
        # Load figure and suit classifiers.
        self.figure_classifier = tf.keras.models.load_model(figure_classifier_save_path)
        with open(suits_classifier_save_path, 'rb') as file:
            self.suits_classifier       = pickle.load(file)
            self.n_fourier_coefficients = pickle.load(file)

        self.figure_classifier_input_resolution = figure_classifier_input_resolution
        
        # Lookup table to convert numeric predictions to string.
        self.suit_lookup   = {0: 'S', 1: 'C', 2: 'H', 3: 'D'}
        self.figure_lookup = {i: str(i) for i in range(10)}
        self.figure_lookup[10] = 'J'
        self.figure_lookup[11] = 'Q'
        self.figure_lookup[12] = 'K'
       
    
    def predict_figure(self, image, color):
        preprocessed_image = self._preprocess_figure(image, color)
        
        prediction = self.figure_classifier.predict(preprocessed_image[None, :, :, None])
        
        return self.figure_lookup[np.argmax(prediction)]



    def predict_suit(self, image, color):
        preprocessed_image = self.preprocess_suit(image, color)
        
        features = self.get_fourier_descriptor(preprocessed_image, n_coefficients_to_keep = self.n_fourier_coefficients)
                
        prediction, = self.suits_classifier.predict([features])
        
        
        return self.suit_lookup[prediction]
    
    
    def _preprocess_figure(self, image, color):
        mask = get_color_pixels(image, color)

        height, width = mask.shape
        max_size = max(height, width)

        pad_left   = (max_size - width ) // 2
        pad_right  = max_size - width - pad_left
        pad_top    = (max_size - height) // 2
        pad_bottom = max_size - height - pad_top

        mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, borderType = cv2.BORDER_CONSTANT, value = 0)
        mask = zoom_image_to_meet_shape(mask, self.figure_classifier_input_resolution)
        mask = normalize(mask)
        mask = binarize(mask)

        return mask.astype(np.float32)
    
    
    @staticmethod
    def preprocess_suit(image, color):
        mask = get_color_pixels(image, color)
        
        # Only take the largest component (except background).
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest_component).astype(np.uint8)

        return mask
    

    @staticmethod
    def get_fourier_descriptor(image, n_coefficients_to_keep = 2):
        """
        Function returning a Fourier descriptor of image made by keeping the first n_coefficients_to_keep coefficients
        (not including the bias coefficient).

        Args:
            image::[np.array]
                Image we want to compute the Fourier descriptor of.
            n_coefficients_to_keep::[int]
                Number of coefficients (not including the bias coefficient) to keep to make up the Fourier descriptor.
        Returns:
            fourier_descriptor::[np.array]
                Array of size (n_coefficients_to_keep, ) where each element is one Fourier descriptor coefficient.
        """        
        # Compute outer contours of image.
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Isolating longest contour from the rest
        len_contours = [len(contour) for contour in contours]
        idx_longest_contour = np.argmax(len_contours)
        contour = contours[idx_longest_contour].squeeze()    

        # Make complex signal from contour array.
        complex_contour_signal = contour[:, 0] + 1j * contour[:, 1]

        # Compute fourier coefficients.
        fourier_coefficients = np.fft.fft(complex_contour_signal)

        # To make fourier coefficient resistant to translation, first coefficient discarded.
        fourier_coefficients = fourier_coefficients[1:]

        # To make fourier coefficient resistant to scaling, ratio between coefficients is used instead of actual magnitude.
        fourier_coefficients = fourier_coefficients / fourier_coefficients[n_coefficients_to_keep+1] 

        # To make fourier coefficient resistant to rotation, phase is discarded.
        fourier_coefficients = np.abs(fourier_coefficients)  

        return fourier_coefficients[:n_coefficients_to_keep]

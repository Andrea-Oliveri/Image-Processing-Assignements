import cv2
import numpy as np
from scripts.extract import *


########### FUNC ############

def pre_process(img):
	"""
	image: colored input image
	"""
	mask_red   = get_color_pixels(img, 'red')
	mask_black = get_color_pixels(img, 'black')
	image = mask_red if mask_red.sum() > mask_black.sum() else mask_black
	
	# adds connected components and take largest one minus background
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 
																			4 ,
																			cv2.CV_32S)
	max_idx = np.argsort(stats[:,-1])[::-1][1]
	image = (labels == max_idx).astype(np.uint8) * 255

	return image

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


########### MAIN ############


def suit_predict(img, model):
    image = pre_process(img)

    huMoments = np.asarray([np.array(cv2.HuMoments(cv2.moments(image))[0:6])])
    huMoments = huMoments.squeeze(-1)
    FD = np.asarray([get_fourier_descriptor(image, n_coefficients_to_keep = 10)])

    X = np.concatenate([huMoments, FD],axis=1)

    return model.predict(X)

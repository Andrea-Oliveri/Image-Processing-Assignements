from collections import defaultdict
from scipy.misc import imread
from skimage.filter import canny
from scipy.ndimage.filters import sobel


# Methods to extract card and coin for classification 














# GENERAL HOUGH 

'''
Created on May 19, 2013

@author: vinnie
'''

# Good for the b/w test images used
MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50

def gradient_orientation(image):
	'''
	Calculate the gradient orientation for edge point in the image
	'''
	dx = sobel(image, axis=0, mode='constant')
	dy = sobel(image, axis=1, mode='constant')
	gradient = np.arctan2(dy,dx) * 180 / np.pi
	
	return gradient
	
def build_r_table(image, origin):
	'''
	Build the R-table from the given shape image and a reference point
	'''
	edges = canny(image, low_threshold=MIN_CANNY_THRESHOLD, 
				  high_threshold=MAX_CANNY_THRESHOLD)
	gradient = gradient_orientation(edges)
	
	r_table = defaultdict(list)
	for (i,j),value in np.ndenumerate(edges):
		if value:
			r_table[gradient[i,j]].append((origin[0]-i, origin[1]-j))

	return r_table

def accumulate_gradients(r_table, grayImage):
	'''
	Perform a General Hough Transform with the given image and R-table
	'''
	edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, 
				  high_threshold=MAX_CANNY_THRESHOLD)
	gradient = gradient_orientation(edges)
	
	accumulator = np.zeros(grayImage.shape)
	for (i,j),value in np.ndenumerate(edges):
		if value:
			for r in r_table[gradient[i,j]]:
				accum_i, accum_j = i+r[0], j+r[1]
				if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
					accumulator[accum_i, accum_j] += 1
					
	return accumulator

def general_hough_closure(reference_image):
	'''
	Generator function to create a closure with the reference image and origin
	at the center of the reference image
	
	Returns a function f, which takes a query image and returns the accumulator
	'''
	referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2)
	r_table = build_r_table(reference_image, referencePoint)
	
	def f(query_image):
		return accumulate_gradients(r_table, query_image)
		
	return f

def n_max(a, n):
	'''
	Return the N max elements and indices in a
	'''
	indices = a.ravel().argsort()[-n:]
	indices = (np.unravel_index(i, a.shape) for i in indices)
	return [(a[i], i) for i in indices]


# # top 5 results in red
# m = n_max(accumulator, 5)
# y_points = [pt[1][0] for pt in m]
# x_points = [pt[1][1] for pt in m] 
# plt.scatter(x_points, y_points, marker='o', color='r')


def reconstruct():
	pass
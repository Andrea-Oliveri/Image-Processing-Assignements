import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Method to extract card and coin for classification 
def get_color_pixels(image, color):
	image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	if color == "black":
		return cv2.inRange(image_hsv, (0, 0, 0), (180, 255, 30))

	elif color == "green":
		return cv2.inRange(image_hsv, (40, 50, 30), (75, 255, 255))
		
	elif color == "red":
		mask1 = cv2.inRange(image_hsv, (0, 70, 50), (10, 255, 255))
		mask2 = cv2.inRange(image_hsv, (170, 70, 50), (180, 255, 255))
		return cv2.bitwise_or(mask1, mask2)
	
	else:
		raise ValueError(f"Color parameter must be one of 'black', 'green', 'red'. Got: {color}")

def red_color_isolation(img) : 
	output = img.copy()
	red_filtered = (output[:,:,0] > 150) & (output[:,:,1] < 100) & (output[:,:,2] < 110)
	output[:, :, 0] = output[:, :, 0] * red_filtered
	output[:, :, 1] = output[:, :, 1] * red_filtered
	output[:, :, 2] = output[:, :, 2] * red_filtered
	output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
	# plt.imshow(output[:,:,0])
	return output[:,:,0]

def green_color_isolation(img) : 
	output = img.copy()
	output = get_color_pixels(output,"green")
	return output

def black_color_isolation(img) : 
	output = img.copy()
	output = get_color_pixels(output,"black")
	return output


def get_rectanglecoords(img):
	# binary_img simply get (from connected components image but masked by label)
	activ_coord = np.where(img)
	ymin,ymax = np.min(activ_coord[0]), np.max(activ_coord[0])
	xmin,xmax = np.min(activ_coord[1]), np.max(activ_coord[1])
	
	return xmin,xmax,ymin,ymax


# Main Func
def extract(img):
	# take as input a bgr image 

	# get contour images through two ways 
	print('Contour Expressing...')

	# 1.Gradient Image 
	gradient_img = cv2.Canny(img,25,100)
	
	# 2.hsv Image 
	output_red = red_color_isolation(img)
	output_green = green_color_isolation(img)
	output_black = black_color_isolation(img)
	output = output_red + output_green + output_black


	# 3.Extract Circle
	print('Extracting Circle Dealer...')
	min_votes = 5000

	# output -> contour made very obvious by color extract
	circles = cv2.HoughCircles(output,cv2.HOUGH_GRADIENT,1,min_votes,
							param1=50,param2=30)

	circles = np.uint16(np.around(circles))

	# take one and only circle, since we put a high minimum vote
	x,y,r = circles[0][0]
	cropped_circ = img[y-r:y+r,x-r:x+r]

	# 4. Extract Cards
	print('Extracting Cards...')
	no_circ = cv2.circle(gradient_img, (x,y), int(r*1.1), (0,0,0), -1)
	contours, _ = cv2.findContours(gaussian_filter(no_circ,sigma=5), 
		cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	array_contour = [contour.reshape(np.array(contour.shape)[[0,2]]) for contour in contours]
	# take top 25% contours (top in terms of size of array)
	top_four = np.asarray([(i,array_contour[i].shape[0]) for i in range(len(array_contour))])
	top_four = np.array(sorted(top_four,key=lambda x:x[1],reverse=True))
	top_four = top_four[:len(top_four)//4,0] #we take half of the largest

	# got top four contours  
	four_contour = np.array(array_contour)[list(top_four)]

	blank = np.zeros_like(no_circ)

	# sequential filling, since it's possible that two contours express the same rectangle and therefore
	# for some reason it does not fill inside
	for one_contour in four_contour:
		cv2.drawContours(blank, [one_contour], -1, (255,255,255), thickness=cv2.FILLED)
	

	# 4 represents minimum number of pixels for a component to be kept
	num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(blank, 4)

	# keep 4 largest components (excluding background) supposedly represents the cards
	_, r1, r2, r3, r4 = np.array(sorted([(idx,val) for idx,val 
							in enumerate(stats[:,-1])],
							key=lambda x:x[1],reverse=True))[:,0][:5]

	card1 = (labels_im == r1)
	card2 = (labels_im == r2)
	card3 = (labels_im == r3)
	card4 = (labels_im == r4)

	xmin1, xmax1, ymin1, ymax1 = get_rectanglecoords(card1)
	xmin2, xmax2, ymin2, ymax2 = get_rectanglecoords(card2)
	xmin3, xmax3, ymin3, ymax3 = get_rectanglecoords(card3)
	xmin4, xmax4, ymin4, ymax4 = get_rectanglecoords(card4)


	return (cropped_circ,
			img[ymin1:ymax1,xmin1:xmax1], 
			img[ymin2:ymax2,xmin2:xmax2], 
			img[ymin3:ymax3,xmin3:xmax3],
			img[ymin4:ymax4,xmin4:xmax4])

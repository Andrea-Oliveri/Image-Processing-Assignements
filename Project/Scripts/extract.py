import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


################################################## UTILs ##############################################################

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

def associate_point_to_player(image_shape, point):
    image_rows, image_columns = image_shape
    point_row , point_column  = point
    
    player_points = [(image_rows, image_columns / 2), (image_rows / 2, image_columns), (0, image_columns / 2), (image_rows / 2, 0)]
    
    distances = cdist(player_points, [point])
    
    player = np.argmin(distances) + 1
    
    return player

def get_rectanglecoords(img):
	# binary_img simply get (from connected components image but masked by label)
	activ_coord = np.where(img)
	ymin,ymax = np.min(activ_coord[0]), np.max(activ_coord[0])
	xmin,xmax = np.min(activ_coord[1]), np.max(activ_coord[1])
	
	return xmin,xmax,ymin,ymax

################################################## MAIN-CLASS ##############################################################

class Extractor():
    
    def __init__(self, canny_thresholds = (25, 100), hough_circles_parameters = (50, 30), sigma_gaussian_blur = 5,
                 smaller_card_side_range = (400, 700), larger_card_size_range = (600, 900), nms_threshold = 0.3):
        self.canny_thresholds = canny_thresholds
        self.hough_circles_parameters = hough_circles_parameters
        self.sigma_gaussian_blur = sigma_gaussian_blur
        
        self.smaller_card_side_range = (min(smaller_card_side_range), max(smaller_card_side_range))
        self.larger_card_size_range  = (min(larger_card_size_range ), max(larger_card_size_range ))
        
        self.nms_threshold = nms_threshold
        
        
        
    def _extract(self, image):
        dealer_circle, dealer_player = self._extract_dealer(image)
        
        cards = self._extract_cards(image, dealer_circle)
        
        return dealer_player, cards
        
        
        
    def _extract_dealer(self, image):
        green_mask  = get_color_pixels(image, "green")

        # Extract circle.
        circles = cv2.HoughCircles(green_mask, cv2.HOUGH_GRADIENT, dp = 1,
                                   minDist = np.inf, 
                                   param1  = self.hough_circles_parameters[0], 
                                   param2  = self.hough_circles_parameters[1])

        circles = np.uint16(np.around(circles))

        # Take the only circle detected, as we put a high minimum distance.
        column, row, radius = circles[0][0]
        
        # Show detection.
        #plt.imshow(image[row-radius:row+radius,column-radius:column+radius][:,:,::-1])
        #plt.show()
        
        # Determine which player is dealer.
        dealer_player = self._associate_point_to_player(image, (row, column))
        
        return (column, row, radius), dealer_player
        
    
    def _extract_cards(self, image, dealer_circle):
        column, row, radius = dealer_circle
        
        gradient = cv2.Canny(image, *self.canny_thresholds) 

        gradient_no_circle         = cv2.circle(gradient, (column, row), int(radius * 1.1), (0, 0, 0), cv2.FILLED)
        gradient_no_circle_blurred = cv2.GaussianBlur(gradient_no_circle, None, sigmaX = self.sigma_gaussian_blur)
        
        # Here as we don't use hierarcy returned parameter we don't need RETR_TREE and we can use RETR_LIST
        contours, _ = cv2.findContours(gradient_no_circle_blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            if self._bbox_can_be_card(bbox):
                bounding_boxes.append(bbox)
        
        indices = cv2.dnn.NMSBoxes(bounding_boxes, [w*h for _, _, w, h in bounding_boxes], 0, self.nms_threshold)
        bounding_boxes = [bbox for idx, bbox in enumerate(bounding_boxes) if idx in indices]
        

        cards = {}
        for bbox in bounding_boxes:
            player = self._associate_bbox_to_player(image, bbox)
            column, row, width, height = bbox
            card_image = image[row:row+height, column:column+width]
            card_image = np.rot90(card_image, 1-player)
            
            cards[player] = card_image
        
        return cards
        
    
    
    def _bbox_can_be_card(self, bbox):
        _, _, width, height = bbox
        
        # width_height_list.append(width)
        # width_height_list.append(height)

        width_in_smaller_range = self.smaller_card_side_range[0] <= width <= self.smaller_card_side_range[1]
        width_in_larger_range  = self.larger_card_size_range [0] <= width <= self.larger_card_size_range [1]

        height_in_smaller_range = self.smaller_card_side_range[0] <= height <= self.smaller_card_side_range[1]
        height_in_larger_range  = self.larger_card_size_range [0] <= height <= self.larger_card_size_range [1]
        
        return (width_in_smaller_range and height_in_larger_range) or (width_in_larger_range and height_in_smaller_range)
    

    
    
    
    def _associate_point_to_player(self, image, point):
        image_rows, image_columns, _ = image.shape
        point_row , point_column     = point

        player_points = [(image_rows, image_columns / 2), (image_rows / 2, image_columns), (0, image_columns / 2), (image_rows / 2, 0)]

        distances = cdist(player_points, [point])

        player = np.argmin(distances) + 1

        return player
    
    
    def _associate_bbox_to_player(self, image, bbox):
        column, row, width, height = bbox
        center_row    = row    + height / 2
        center_column = column + width  / 2

        return self._associate_point_to_player(image, (center_row, center_column))
    
    
    def __call__(self, image):
        return self._extract(image)
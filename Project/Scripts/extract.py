import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist



################################################## UTILs ##############################################################

def get_color_pixels(image, color):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if color == "black":
        return cv2.inRange(image_hsv, (0, 0, 0), (180, 255, 45))

    elif color == "green":
        return cv2.inRange(image_hsv, (40, 50, 30), (75, 255, 255))
        
    elif color == "red":
        mask1 = cv2.inRange(image_hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(image_hsv, (170, 70, 50), (180, 255, 255))
        return cv2.bitwise_or(mask1, mask2)
    
    else:
        raise ValueError(f"Color parameter must be one of 'black', 'green', 'red'. Got: {color}")


################################################## MAIN-CLASS ##############################################################


class Extractor():
    
    def __init__(self, canny_thresholds = (25, 100), hough_circles_parameters = (50, 30), sigma_gaussian_blur = 5,
                 smaller_card_side_range = (400, 700), larger_card_size_range = (600, 900), 
                 nms_threshold = 0.3, tolerance = 0, min_size = 2000, median_filter_size = 7,
                 figure_suits_crop_margin = 70):
        """
        tolerance::[float]
            Tolerance in inequality check to decide whether components are close enough.
        min_size::[int]
                Minimum size in number of pixels of component to be considered a valid object.
        """
        
        # Parameters used to extract dealer circle.
        self.hough_circles_parameters = hough_circles_parameters
        
        # Parameters used to extract cards.
        self.canny_thresholds = canny_thresholds
        self.sigma_gaussian_blur = sigma_gaussian_blur
        self.smaller_card_side_range = (min(smaller_card_side_range), max(smaller_card_side_range))
        self.larger_card_size_range  = (min(larger_card_size_range ), max(larger_card_size_range ))
        self.nms_threshold = nms_threshold
        
        # Parameters used to extract figures and suits.
        self.tolerance = tolerance
        self.min_size  = min_size
        self.median_filter_size = median_filter_size
        self.figure_suits_crop_margin = figure_suits_crop_margin
        
        
    def __call__(self, image):
        return self._extract(image) 
        
        
    def _extract(self, image):
        dealer_circle, dealer_player = self._extract_dealer(image)
        
        cards = self._extract_cards(image, dealer_circle)
        
        figures_suits = self._extract_figures_suits(cards)
        
        return dealer_player, cards, figures_suits
        
        
        
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
        
        contours, _ = cv2.findContours(gradient_no_circle_blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
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
        
        # Debug only ---------
#         width_height_list.append(width)
#         width_height_list.append(height)
        # --------------------
        
        width_in_smaller_range = self.smaller_card_side_range[0] <= width <= self.smaller_card_side_range[1]
        width_in_larger_range  = self.larger_card_size_range [0] <= width <= self.larger_card_size_range [1]

        height_in_smaller_range = self.smaller_card_side_range[0] <= height <= self.smaller_card_side_range[1]
        height_in_larger_range  = self.larger_card_size_range [0] <= height <= self.larger_card_size_range [1]
        
        return (width_in_smaller_range and height_in_larger_range) or (width_in_larger_range and height_in_smaller_range)    
    
    
    def _associate_point_to_player(self, image, point):       
        image_rows, image_columns = image.shape[0], image.shape[1]
        point_row , point_column  = point

        player_points = [(image_rows, image_columns / 2), (image_rows / 2, image_columns), (0, image_columns / 2), (image_rows / 2, 0)]

        distances = cdist(player_points, [point])

        player = np.argmin(distances) + 1

        return player
    
    
    def _associate_bbox_to_player(self, image, bbox):
        column, row, width, height = bbox
        center_row    = row    + height / 2
        center_column = column + width  / 2

        return self._associate_point_to_player(image, (center_row, center_column))
    
    
    def _extract_figures_suits(self, cards):
        figures_suits = {}
        
        for player, card in cards.items():
            
            red_mask   = get_color_pixels(card, "red")
            black_mask = get_color_pixels(card, "black")
            color      = "red" if red_mask.sum() > black_mask.sum() else "black"
            
            mask       = red_mask if color == 'red' else black_mask
            
            mask = cv2.medianBlur(mask, self.median_filter_size)
            
            new_mask = self._get_objects_from_mask(mask)
            labels   = np.unique(new_mask)
            
            stats = []
            for label in labels:
                
                # Background
                if label == 0:
                    continue
                    
                rows, cols       = np.where(new_mask == label)
                centroid         = np.mean(rows), np.mean(cols)
                min_row, max_row = np.min (rows), np.max (rows)
                min_col, max_col = np.min (cols), np.max (cols)
                
                stats.append({'distance': np.sum(np.square(centroid)), 'area': len(rows),
                              'min_row': min_row, 'max_row': max_row, 
                              'min_col': min_col, 'max_col': max_col})
            
                        
            object_top_left     = min(stats, key = lambda elem: elem['distance'])
            object_bottom_right = max(stats, key = lambda elem: elem['distance'])

            suit_object = object_top_left if object_top_left['area'] > object_bottom_right['area'] else object_bottom_right
            suit_symbol = self._crop_element_with_margins(card, 
                                                          (suit_object['min_row'], suit_object['max_row']),
                                                          (suit_object['min_col'], suit_object['max_col']),
                                                          self.figure_suits_crop_margin)    

            card = card[object_top_left['max_row']:object_bottom_right['min_row'], :]

            figure_object = {'min_row': np.inf, 'max_row': -np.inf, 'min_col': np.inf, 'max_col': -np.inf}
            for elem in stats:
                if elem not in [object_top_left, object_bottom_right]:
                    figure_object['min_row'] = min(figure_object['min_row'], elem['min_row'])
                    figure_object['min_col'] = min(figure_object['min_col'], elem['min_col'])
                    figure_object['max_row'] = max(figure_object['max_row'], elem['max_row'])
                    figure_object['max_col'] = max(figure_object['max_col'], elem['max_col'])

            figure_object['min_row'] -= object_top_left['max_row']
            figure_object['max_row'] -= object_top_left['max_row']


            figure_symbol = self._crop_element_with_margins(card, 
                                                            (figure_object['min_row'], figure_object['max_row']),
                                                            (figure_object['min_col'], figure_object['max_col']),
                                                            self.figure_suits_crop_margin)    

            figures_suits[player] = {"figure": figure_symbol, "suit": suit_symbol, "color": color}   
            
        return figures_suits
        
        
    def _crop_element_with_margins(self, image, row_range, col_range, margin):
        min_row, max_row = row_range
        min_col, max_col = col_range

        image_rows, image_cols, _ = image.shape

        min_row = max(min_row - margin, 0)
        min_col = max(min_col - margin, 0)
        max_row = min(max_row + margin, image_rows - 1)
        max_col = min(max_col + margin, image_cols - 1)
        
        return image[min_row:max_row, min_col:max_col]

        
    def _detect_same_object(self, component1, component2):
        """
        Function returning whether two components are sufficiently close to be classified as belonging
        to same object. 

        Args:
            component1::[dict]
                Dictionary of form {"centroid": np.array, "width": float, "height": float} containing
                information about the first connected component.
            component2::[dict]
                Dictionary of form {"centroid": np.array, "width": float, "height": float} containing
                information about the second connected component.        
        Returns:
            output::[boolean]
                True if the components are close enough to probably be part of the same object, False
                otherwise.
        """
        distance = np.linalg.norm(component1["centroid"] - component2["centroid"])
        radius_component1 = (component1["width"] + component1["height"]) / 4 
        radius_component2 = (component2["width"] + component2["height"]) / 4 

        return distance <= radius_component1 + self.tolerance or distance <= radius_component2 + self.tolerance


    def _get_objects_from_mask(self, mask):
        """
        Function returning segmentation mask of objects obtained by cleaning the color mask passed
        as parameter. 

        Args:
            mask::[np.array]
                Image mask obtained by color thresholding.
        Returns:
            objects_mask::[np.array]
                Mask of same size as input mask segmenting the objects which are large enough and
                grouped into one object if they are close enough.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # Merging close components that belong to same object, skipping background labeled 0.
        new_labels = {}
        for component1 in range(1, num_labels):
            parameters1 = {"centroid": np.array(centroids[component1]), "width": stats[component1, cv2.CC_STAT_WIDTH], "height": stats[component1, cv2.CC_STAT_HEIGHT]}
            for component2 in range(component1 + 1, num_labels):
                parameters2 = {"centroid": np.array(centroids[component2]), "width": stats[component2, cv2.CC_STAT_WIDTH], "height": stats[component2, cv2.CC_STAT_HEIGHT]}
                if self._detect_same_object(parameters1, parameters2):
                    new_labels[component2] = component1

        # Solves conflicts caused by values in dictionary which are also keys.
        key_value_conflicts = [key for key, value in new_labels.items() if value in new_labels.keys()]
        while key_value_conflicts:
            for conflict in key_value_conflicts:
                new_labels[conflict] = new_labels[new_labels[conflict]]
            key_value_conflicts = [key for key, value in new_labels.items() if value in new_labels.keys()]

        # Merges close components that belong to same object.
        for old_component, new_component in new_labels.items():
            labels[labels == old_component] = new_component
        
        # Removes too small groups (merges them with background), assuring there are always at least 3 components 
        # plus background which are kept, and which would be the two suits and the figure.
        unique, counts = np.unique(labels, return_counts = True)
        max_groups_that_can_be_removed = len(unique) - 3 - 1
        for idx_smallest_group in np.argsort(counts):
            if max_groups_that_can_be_removed <= 0:
                break
                
            component = unique[idx_smallest_group]
            n_pixels  = counts[idx_smallest_group]
            if n_pixels < self.min_size:
                labels[labels == component] = 0
                max_groups_that_can_be_removed -= 1
        
        return labels
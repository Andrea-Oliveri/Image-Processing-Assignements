import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist



def get_color_pixels(image, color):
    """
    Function returning a mask containing all pixels in image with a desired color.

    Args:
        image::[np.array]
            Image we want to create the color mask of.
        color::[str]
            String containing either 'red' or 'black' or 'green' and describing which pixels we want to mask in the image. 
    Returns:
        mask::[np.array]
            Binary mask containing 0 for pixels which do not match the desired color and 1 for those which match.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if color == "black":
        return cv2.inRange(image_hsv, (0, 0, 0), (180, 255, 50))

    elif color == "green":
        return cv2.inRange(image_hsv, (40, 50, 30), (75, 255, 255))
        
    elif color == "red":
        mask1 = cv2.inRange(image_hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(image_hsv, (170, 70, 50), (180, 255, 255))
        return cv2.bitwise_or(mask1, mask2)
    
    else:
        raise ValueError(f"Color parameter must be one of 'black', 'green', 'red'. Got: {color}")





class Extractor():
    """
    Class collecting all necessary steps to extract dealer, cards, figures and suits bounding boxes and cropped images
    from the whole round input image.
    """
        
    def __init__(self, hough_circles_parameters = (50, 30), canny_thresholds = (25, 100), sigma_gaussian_blur = 5,
                 smaller_card_side_range = (400, 700), larger_card_size_range = (600, 900), 
                 nms_threshold = 0.3, same_component_tolerance = 0, min_size = 2000, median_filter_size = 7,
                 figure_suits_crop_margin = 70, min_distance_figure_component = 0.1, max_distance_figure_component = 0.5):
        """
        Constructor of FiguresSuitsClassifier. Stores plently of parameters needed to be tuned to effectively extract the 
        dealer, cards, figures and suits of interest from the input image.
        
        Args:
            hough_circles_parameters::[tuple]
                Tuple containing (param1, param2) of cv2.HoughCircles.
            canny_thresholds::[tuple]
                Thresholds used by the Canny Edge Detector when isolating the cards from the rest of the image.
            sigma_gaussian_blur::[float]
                Standard deviation of the gaussian blur used on the output of the Canny Edge Detector 
                when isolating the cards from the rest of the image. Needed to close the cards contours.
            smaller_card_side_range::[tuple]
                Tuple containing (min_smaller_size_length, max_smaller_size_length) describing the range of width or
                height that a bounding box can have on its smaller size to be considered a card. 
            larger_card_size_range::[tuple]
                Tuple containing (min_larger_size_length, max_larger_size_length) describing the range of width or
                height that a bounding box can have on its larger size to be considered a card. 
            nms_threshold::[float]
                Non-Maximum Suppression threshold to used when removing several bounding boxes for the same card.
            same_component_tolerance::[float]
                Tolerance in number of pixels in the inequality check to decide if components are close enough to
                be merged into one component.
            min_size::[int]
                Min are in number of pixels that a component must span for it to be considered not noise.
            median_filter_size::[int]
                Size of the median filter used to remove imperfections from color mask generated when segmenting suits
                and figures.
            figure_suits_crop_margin::[int]
                Number of additional pixels to include as margin when cropping a bounding box for the suit or figure from
                the image.
            min_distance_figure_component::[float]
                Minumum distance that the centroid of a connected component must be at for it to be considered part of the
                figure rather than part of the suit.
            max_distance_figure_component::[float]
                Maximum distance that the centroid of a connected component must be at for it to be considered part of the
                figure rather than part of the suit.
        Returns:
            None
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
        self.same_component_tolerance = same_component_tolerance
        self.min_size  = min_size
        self.median_filter_size = median_filter_size
        self.figure_suits_crop_margin = figure_suits_crop_margin
        self.min_distance_figure_component = min_distance_figure_component
        self.max_distance_figure_component = max_distance_figure_component
        
        
    def __call__(self, image):
        """
        Special function simply calling self._extract.
        Args:
            image::[np.array]
                See image parameter in _extract method.
        Returns:
            dealer_data::[dict]
                See returned value in _extract method.
            cards::[dict]
                See returned value in _extract method.
            figures_suits::[dict]
                See returned value in _extract method.
        """
        return self._extract(image) 
        
        
    def _extract(self, image):
        """
        Function extracting the dealer, the cards, the figures and suits from the image. Both cropped images and
        bounding boxes are returned.
        
        Args:
            image::[np.array]
                Image we want to extract the dealer, the cards, the figures and suits from.
        Returns:
            dealer_data::[dict]
                Dictionary containing the position and bounding box of the dealer circle and prediction for the 
                dealer player.
            cards::[dict]
                Dictionary containing, for each player, the bounding box and cropped image of the his card.
            figures_suits::[dict]
                Dictionary containing, for each player, the bounding box, color and cropped image of the suits and
                figure in his card.                
        """
        dealer_data   = self._extract_dealer(image)
        cards         = self._extract_cards(image, dealer_data['circle'])
        figures_suits = self._extract_figures_suits(cards)
        
        return dealer_data, cards, figures_suits
        
        
        
    def _extract_dealer(self, image):
        """
        Function extracting the dealer from the image. The position and bounding box of the dealer circle as well
        as prediction for the dealer player are returned.
        
        Args:
            image::[np.array]
                Image we want to extract the dealer from.
        Returns:
            dealer_data::[dict]
                Dictionary containing the position and bounding box of the dealer circle and prediction for the 
                dealer player.        
        """
        green_mask  = get_color_pixels(image, "green")

        # Extract circle.
        circles = cv2.HoughCircles(green_mask, cv2.HOUGH_GRADIENT, dp = 1,
                                   minDist = np.inf, 
                                   param1  = self.hough_circles_parameters[0], 
                                   param2  = self.hough_circles_parameters[1])

        circles = np.uint16(np.around(circles))

        # Take the only circle detected, as we put a high minimum distance.
        column, row, radius = circles[0][0]
        
        # Determine which player is dealer.
        dealer_player = self._associate_point_to_player(image, (row, column))
        
        return {'circle': (column, row, radius), 'player': dealer_player, 
                'bbox': (column - radius, row - radius, 2 * radius, 2 * radius)}
        
    
    def _extract_cards(self, image, dealer_circle):
        """
        Function extracting the cards from the image. The bounding box and cropped image of the cards are returned.
        
        Args:
            image::[np.array]
                Image we want to extract the dealer from.
        Returns:
            cards::[dict]
                Dictionary containing, for each player, the bounding box and cropped image of the his card.
        """
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
            
            cards[player] = {'image': card_image, 'bbox': bbox}
        
        return cards
        
    
    
    def _bbox_can_be_card(self, bbox):
        """
        Function returning a boolean describing if the bounding box passed as parameter has width and height compatible
        with that of a card.
        
        Args:
            bbox::[tuple]
                Tuple with (col, row, width, height) describing the bounding box we want to test if it can be a card.
        Returns:
            can_be_card::[boolean]
                Boolean describing if the bounding box passed as parameter has width and height compatible with that of a card.
        """
        _, _, width, height = bbox
        
        width_in_smaller_range = self.smaller_card_side_range[0] <= width <= self.smaller_card_side_range[1]
        width_in_larger_range  = self.larger_card_size_range [0] <= width <= self.larger_card_size_range [1]

        height_in_smaller_range = self.smaller_card_side_range[0] <= height <= self.smaller_card_side_range[1]
        height_in_larger_range  = self.larger_card_size_range [0] <= height <= self.larger_card_size_range [1]
        
        return (width_in_smaller_range and height_in_larger_range) or (width_in_larger_range and height_in_smaller_range)    
    
    
    def _associate_point_to_player(self, image, point):
        """
        Function associating for any point in an image a player by computing the distance of the point from the middle of
        each edge of the image.
        
        Args:
            image::[np.array]
                Image containing the point that we want to associate to a player.
            point::[tuple]
                Tuple of (row, col) containing the coordinates of the point we want to associate to a player.
        Returns:
            player::[int]
                The player closest to the point to which the point got assigned. 
        """
        image_rows, image_columns = image.shape[0], image.shape[1]
        point_row , point_column  = point

        player_points = [(image_rows, image_columns / 2), (image_rows / 2, image_columns), (0, image_columns / 2), (image_rows / 2, 0)]

        distances = cdist(player_points, [point])

        player = np.argmin(distances) + 1

        return player
    
    
    def _associate_bbox_to_player(self, image, bbox):
        """
        Function associating a bounding box in an image a player by computing the distance of the centroid of the box
        from the middle of each edge of the image.
        
        Args:
            image::[np.array]
                Image containing the bounding box that we want to associate to a player.
            bbox::[tuple]
                Tuple with (col, row, width, height) describing the bounding box we want to associate to a player.
        Returns:
            player::[int]
                The player closest to the bounding box center to which the bounding box got assigned. 
        """
        column, row, width, height = bbox
        center_row    = row    + height / 2
        center_column = column + width  / 2

        return self._associate_point_to_player(image, (center_row, center_column))
    
    
    def _extract_figures_suits(self, cards):
        """
        Function extracting the figures and suits from the each player's card. Both cropped images and bounding boxes 
        are returned, as well as the color of the suits and figure.
        
        Args:
            cards::[dict]
                Dictionary containing, for each player, the bounding box and cropped image of the his card.
        Returns:
            figures_suits::[dict]
                Dictionary containing, for each player, the bounding box, color and cropped image of the suits and
                figure in his card.                
        """
        figures_suits = {}
        
        for player, card_data in cards.items():
            card = card_data["image"]
            
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
                
                diagonal_length = card.shape[0] ** 2 + card.shape[1] ** 2
                
                stats.append({'distance': np.sum(np.square(centroid)) / diagonal_length, 'area': len(rows),
                              'min_row': min_row, 'max_row': max_row, 
                              'min_col': min_col, 'max_col': max_col})
            
                        
            object_top_left     = min(stats, key = lambda elem: elem['distance'])
            object_bottom_right = max(stats, key = lambda elem: elem['distance'])

            suit_object = object_top_left if object_top_left['area'] > object_bottom_right['area'] else object_bottom_right
            suit_symbol = self._crop_element_with_margins(card, 
                                                          (suit_object['min_row'], suit_object['max_row']),
                                                          (suit_object['min_col'], suit_object['max_col']),
                                                          self.figure_suits_crop_margin)    

            figure_object = {'min_row': np.inf, 'max_row': -np.inf, 'min_col': np.inf, 'max_col': -np.inf}
            for elem in stats:
                if elem not in [object_top_left, object_bottom_right] and \
                   self.min_distance_figure_component < elem['distance'] < self.max_distance_figure_component:
                    figure_object['min_row'] = min(figure_object['min_row'], elem['min_row'])
                    figure_object['min_col'] = min(figure_object['min_col'], elem['min_col'])
                    figure_object['max_row'] = max(figure_object['max_row'], elem['max_row'])
                    figure_object['max_col'] = max(figure_object['max_col'], elem['max_col'])


            figure_symbol = self._crop_element_with_margins(card, 
                                                            (figure_object['min_row'], figure_object['max_row']),
                                                            (figure_object['min_col'], figure_object['max_col']),
                                                            self.figure_suits_crop_margin)    

            figures_suits[player] = {"figure": figure_symbol, "suit": suit_symbol, "color": color}
            
            for key, object_to_bbox in zip(["bbox_figure", "bbox_suit_1", "bbox_suit_2"], [figure_object, object_top_left, object_bottom_right]):
                width  = object_to_bbox["max_col"] - object_to_bbox["min_col"]
                height = object_to_bbox["max_row"] - object_to_bbox["min_row"]
                figures_suits[player][key] = (object_to_bbox["min_col"], object_to_bbox["min_row"], width, height)

        return figures_suits
        
        
    def _crop_element_with_margins(self, image, row_range, col_range, margin):
        """
        Crops a region of the image described by row_range and col_range leaving a margin on each side.
        
        Args:
            image::[np.array]
                Image we want to crop the region of interest from.
            row_range::[tuple]
                Tuple containing (min_row, max_row) of the region of image we want to crop.
            col_range::[tuple]
                Tuple containing (min_col, max_col) of the region of image we want to crop.
            margin::[int]
                The number of pixels to leave around each side of the region of interest when cropping.
        Returns:
            cropped_image::[np.array]
                Region of interest cropped with margins from image. 
        """
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

        return distance <= radius_component1 + self.same_component_tolerance or \
               distance <= radius_component2 + self.same_component_tolerance


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
            parameters1 = {"centroid": np.array(centroids[component1]), 
                           "width": stats[component1, cv2.CC_STAT_WIDTH], 
                           "height": stats[component1, cv2.CC_STAT_HEIGHT]}
            for component2 in range(component1 + 1, num_labels):
                parameters2 = {"centroid": np.array(centroids[component2]), 
                               "width": stats[component2, cv2.CC_STAT_WIDTH],
                               "height": stats[component2, cv2.CC_STAT_HEIGHT]}
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
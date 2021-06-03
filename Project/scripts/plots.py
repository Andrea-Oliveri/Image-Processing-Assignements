# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    
    
    
def draw_and_plot_overlay(image, bboxes_and_predictions, game, round_):
    
    def draw_bbox(image, bbox, bbox_type, thickness = 30, bbox_title = ""):
        if bbox_type == "card" or bbox_type == "dealer":
            color = (200, 72, 16)
        else:
            color = (48, 225, 240)
        
        col, row, width, height = bbox
        
        image = cv2.rectangle(image, \
                              (col - thickness, row - thickness), \
                              (col + width + thickness, row + height + thickness), \
                              color = color, \
                              thickness = thickness)
        
        image = cv2.putText(image, \
                            bbox_title, \
                            (col - thickness, row - thickness - 60), \
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 5, \
                            color = color, \
                            thickness = 15)

        return image
    
    
    def map_bbox_inside_card(bbox_inside, bbox_card, player):
        """
        Function converting bbox_inside from the set of coordinates relative to the inside of the card
        into the set of coordinates relative to the whole image.
        
        Args:
            bbox_inside::[tuple]
                Tuple with (col, row, width, height) describing the bounding box with coordinates relative
                to the inside of the card (inside of bbox_card). 
            bbox_card::[tuple]
                Tuple with (col, row, width, height) describing the bounding box of the card containing 
                bbox_inside with coordinates relative to the image.
            player::[int]
                Which player does bbox_card belong to.
                
        Returns:
            bbox_transformed::[tuple]
                Tuple with (col, row, width, height) describing the same region as bbox_inside inside bbox_card,
                but relative to the image coordinates.
        """
        col_inside, row_inside, width_inside, height_inside = bbox_inside
        col_card  , row_card  , width_card  , height_card   = bbox_card
        
        if player == 1:
            col    = col_card + col_inside
            row    = row_card + row_inside
            width  = width_inside
            height = height_inside
            
        elif player == 2:
            col    = col_card + row_inside
            row    = row_card + height_card - col_inside - width_inside
            width  = height_inside
            height = width_inside

        elif player == 3:
            col    = col_card + width_card - col_inside - width_inside
            row    = row_card + height_card - row_inside - height_inside
            width  = width_inside
            height = height_inside
            
        else:
            col    = col_card + width_card - row_inside - height_inside
            row    = row_card + col_inside
            width  = height_inside
            height = width_inside
            
            
        return (col, row, width, height)

    
    image = image.copy()
    
    dealer_bbox = bboxes_and_predictions.pop("dealer_bbox")
    dealer_pred = bboxes_and_predictions.pop("dealer_pred")

    image = draw_bbox(image, dealer_bbox, "dealer", bbox_title = f"Player {dealer_pred}")
    
    for player, data in bboxes_and_predictions.items():
        card_bbox = data["card_bbox"]
        card_pred = data["card_pred"]
        
        image = draw_bbox(image, card_bbox, "card", bbox_title = card_pred)

        image = draw_bbox(image, map_bbox_inside_card(data["figure_bbox" ], card_bbox, player), "figure")
        image = draw_bbox(image, map_bbox_inside_card(data["suits_bbox_1"], card_bbox, player), "suit")
        image = draw_bbox(image, map_bbox_inside_card(data["suits_bbox_2"], card_bbox, player), "suit")

        
    plt.figure(figsize = (10, 7))
    plt.imshow(image[:,:,::-1])
    plt.title(f"Game: {game}, Round: {round_}")
    plt.xticks([])
    plt.yticks([])
    plt.show()
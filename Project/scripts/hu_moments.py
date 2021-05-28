import cv2
import numpy as np


spade = cv2.imread('./media/spade.jpg',0)
clover = cv2.imread('./media/clover.jpg',0)
diamond = cv2.imread('./media/diamond.jpg',0)
heart = cv2.imread('./media/heart.jpg',0)

SUITS = [spade,clover,diamond,heart]
TEMPLATES = [cv2.HuMoments(cv2.moments(suit))[:2] for suit in SUITS]


def classify(mask, templates=TEMPLATES):
    
    # mask is suit image from extractor 
    features = cv2.HuMoments(cv2.moments(mask))[:2]
    return np.argmin([(center[0]-features[0])**2 + (center[1]-features[1])**2
                      for center in templates])
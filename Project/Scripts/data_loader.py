import os
import cv2
import numpy as np
import pandas as pd
import re


class DataLoader():

    def __init__(self, data_dir = './train_games/'):
        self.data_dir = data_dir
        self._load_data()

    
    def __getitem__(self, indices):
        # indices must have form (n_game, n_round) and returns both image and labels.

        assert type(indices) in [tuple, list] and len(indices) == 2, \
               "__getitem__ expects a tuple of 2 indices of form (n_game, n_round)."
        
        n_game, n_round = [int(idx) for idx in indices]
        
        image_row  = self.images[(self.images['game'] == n_game) & (self.images['round'] == n_round)]
        labels_row = self.labels[(self.labels['game'] == n_game) & (self.labels['round'] == n_round)]
        
        if image_row.empty or labels_row.empty:
            raise KeyError(f'Round {n_round} in Game {n_game} does not exist in this dataset')
        
        image, = image_row['image']
        labels = labels_row.drop(columns = ['game', 'round']).to_dict('list')
        labels = {k: v[0] for k, v in labels.items()}
        
        return image, labels


    def _load_data(self):

        self.labels = []
        self.images = []
        
        for game_dir in os.listdir(self.data_dir):
            game_path    = os.path.join(self.data_dir, game_dir)
            game_number, = re.findall(r'\d+', game_dir)
            
            # Contains names of images for each round and of labels .csv
            game_files = os.listdir(game_path)
            
            # Separates names of image files and .csv file.
            game_csv, = [file for file in os.listdir(game_path) if file.endswith('.csv')]
            game_files.remove(game_csv)
            
            # Load .csv file.
            game_labels = pd.read_csv(os.path.join(game_path, game_csv), delimiter = ',') \
                            .assign(game = game_number) \
                            .rename(columns = {'Unnamed: 0': 'round'}) \
                            .astype({'game': int, 'round': int, 'D': int})
            
            self.labels.append(game_labels)
            
            # Load images.
            for image_file in game_files:
                image_path    = os.path.join(game_path, image_file)
                round_number, = re.findall(r'\d+', image_file)
                
                self.images.append( {'game': game_number, 'round': round_number, 'image': cv2.imread(image_path)} )               
        
        self.labels = pd.concat(self.labels, axis = 0, ignore_index = True)        .sort_values(['game', 'round'])
        self.images = pd.DataFrame(self.images).astype({'game': int, 'round': int}).sort_values(['game', 'round'])
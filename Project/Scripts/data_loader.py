import os
import cv2
import numpy as np
import pandas as pd
import re


class DataLoader():

    def __init__(self, data_dir = 'train_games'):
        self.data_dir = data_dir
        self._load_data()

    
    def __getitem__(self, indices):
        # indices must have form (n_game, n_round) and function returns both image and labels.

        assert type(indices) in [tuple, list] and len(indices) == 2, \
               "__getitem__ expects a tuple of 2 indices of form (n_game, n_round)."
        
        n_game, n_round = [int(idx) for idx in indices]
        
        dataframe_row = self.dataframe[(self.dataframe['game'] == n_game) & (self.dataframe['round'] == n_round)]
        
        if dataframe_row.empty:
            raise KeyError(f'Round {n_round} in Game {n_game} does not exist in this dataset')
        
        image_path, = dataframe_row['image']
        image       = cv2.imread(image_path)
        
        labels = dataframe_row.drop(columns = ['game', 'round', 'image']).to_dict('list')
        labels = {k: v[0] for k, v in labels.items()}
        
        return image, labels


    def _load_data(self):

        self.dataframe = []
        
        for game_dir in os.listdir(self.data_dir):
            game_path    = os.path.join(self.data_dir, game_dir)
            game_number, = re.findall(r'(?<=game)\d+', game_dir)
            
            # Contains names of images for each round and of labels .csv
            game_files = os.listdir(game_path)
            
            # Separates names of image files and .csv file.
            game_csv, = [file for file in os.listdir(game_path) if file.endswith('.csv')]
            game_files.remove(game_csv)
            
            # Load image paths.
            game_image_paths = {}
            for image_file in game_files:
                image_path    = os.path.join(game_path, image_file)
                round_number, = re.findall(r'\d+(?=.jpg)', image_file)
                
                game_image_paths[int(round_number)] = os.path.join(image_path)
                
            
            # Load .csv file.
            game_data = pd.read_csv(os.path.join(game_path, game_csv), delimiter = ',') \
                          .assign(game = game_number) \
                          .rename(columns = {'Unnamed: 0': 'round'}) \
                          .astype({'game': int, 'round': int, 'D': int})
            
            game_data['image'] = game_data.apply(lambda x: game_image_paths[x['round']], axis = 1)            
            
            self.dataframe.append(game_data)
                   
        
        self.dataframe = pd.concat(self.dataframe, axis = 0).sort_values(['game', 'round']).reset_index(drop = True)
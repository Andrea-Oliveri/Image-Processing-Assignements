import os
import cv2
import numpy as np
import pandas as pd
import re


class DataLoader():

    def __init__(self, data_dir = os.path.join('.', 'train_games')):
        self.data_dir = data_dir

    
    def __getitem__(self, indices):
        # Indices must have form (n_game, n_round) and function returns both image and labels.

        assert type(indices) in [tuple, list] and len(indices) == 2, \
               "__getitem__ expects a tuple of 2 indices of form (n_game, n_round)."
        
        n_game, n_round = [int(idx) for idx in indices]
        
        
        # Generating and testing existence of paths for game folder, image round and game csv.
        game_path = os.path.join(self.data_dir, f'game{n_game}')
        
        if not os.path.isdir(game_path):
            raise ValueError(f"Folder for game {n_game} can't be found in directory {self.data_dir}")
            
        image_path = os.path.join(game_path, f'{n_round}.jpg')
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Image for round {n_round} can't be found in directory {game_path}")
        
        csv_path = os.path.join(game_path, f'game{n_game}.csv')

        
        # Loading requested data.
        image = cv2.imread(image_path)

        if os.path.isfile(csv_path):
            game_data = pd.read_csv(csv_path, delimiter = ',') \
                          .rename(columns = {'Unnamed: 0': 'round'}) \
                          .astype({'round': int, 'D': int})
        
            game_data_row = game_data[game_data['round'] == n_round]
        
            if game_data_row.empty:
                raise KeyError(f'Round {n_round} does not exist in {csv_path}')
                
            labels = game_data_row.drop(columns = ['round']).to_dict('list')
            labels = {k: v[0] for k, v in labels.items()}
            
            return image, labels
        
        return image, None
    
        
    def get_available_games(self):
        game_numbers = []
        
        for game_dir in os.listdir(self.data_dir):
            game_number = re.findall(r'(?<=game)\d+', game_dir)[-1]
        
            game_numbers.append( int(game_number) )
            
        return sorted(game_numbers)
    
    
    def get_available_rounds(self, n_game):
        game_path = os.path.join(self.data_dir, f'game{n_game}')
        
        if not os.path.isdir(game_path):
            return []
        
        image_files   = [file for file in os.listdir(game_path) if file.endswith('.jpg')]
        round_numbers = []
        
        for image_file in image_files:
            round_number = re.findall(r'\d+(?=.jpg)', image_file)[-1]
            
            round_numbers.append( int(round_number) )
            
        return sorted(round_numbers)
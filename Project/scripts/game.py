import numpy as np
import pandas as pd
import warnings
import os
import re

from .data_loader import DataLoader
from .extract import Extractor
from .figures_suits_classifier import FiguresSuitsClassifier
from .plots import draw_and_plot_overlay
from .utils import print_results, evaluate_game

    
def predict_game(game_dir_path = None, game_number = None, data_loader = None, extractor = None, classifier = None,
                 draw_overlay = True, compute_accuracy = False):
    """
    For all rounds of one game, extract the dealer, figures and suits of each player, predict their values, then compute the 
    score of each player at the end of the game using both standard and advanced rules. Optionally shows the overlay and computes
    the accuracy of the predictions. 
    
    The game to analyse can be specified in one of two ways. Either game_dir_path is set while game_number and data_loader are
    left to None, or game_dir_path is left to None and game_number and data_loader are set. If everything is provided, 
    game_dir_path is used and data_loader and game_number are ignored.
    
    The extractor and classifier parameters can be ignored irrespective of all the others, in which case a new Extractor and 
    FiguresSuitsClassifier instance will be created with default parameters.
    
    
    Args:
        game_dir_path::[str]
            The path of the directory containing the images of each round, plus the ground truth .csv (the latter is only needed
            if compute_accuracy = True).
        game_number::[int]
            Number of the game, as contained in the data_loader.data_dir folder.
        data_loader::[DataLoader]
            The instance initialized with a data_dir containing game with number game_number.
        extractor::[Extractor]
            Extractor instance to be used to extract dealer, cards, figures and suits from the image of each round.
            If not provided, an instance with default parameters is created.
        classifier::[FiguresSuitsClassifier]
            FiguresSuitsClassifier instance to be used to predict figure and suits extracted from images of each round. 
            If not provided, an instance with default parameters is created.
        draw_overlay::[boolean]
            Whether for each round an overlay showing the extracted regions of interest for the dealer, cards, figures, suits
            as well as the predictions should be shown.
        compute_accuracy::[boolean]
            Whether when all predictions of all rounds are available, the overall prediction accuracy over the game should be
            computed by using the ground truth .csv file contained in the same folder as the round images.
    Returns:
        None
    """
    # Can be called in two ways: game_dir_path alone and no game_number and data_laoder or data_loader and game_number.
    # extractor and classifier are independent: can be set or left to None independently of all the rest.

    if game_dir_path is None:
        if data_loader is None or game_number is None:
            raise RuntimeError("Game constructor requires both data_loader and game_number to be provided when game_dir_path isn't.")

    else:
        if data_loader is not None or game_number is not None:
            warnings.warn("Provided game_dir_path and either data_loader or game_number (or both) to Game constructor." + \
                          "Ignoring data_loader and game_number parameters.")

        # Get path of parent directory to game directory.
        parent_dir = os.path.abspath(os.path.join(game_dir_path, os.pardir))

        # Create instance of DataLoader for given game_dir_path and store game number.
        data_loader = DataLoader(parent_dir)
                
        game_number = int( re.findall(r'(?<=game)\d+', game_dir_path)[-1] )
            
    
    # Instanciate Extractor and FiguresSuitsClassifier with default values if not provided.
    extractor  = Extractor() if extractor is None else extractor
    classifier = FiguresSuitsClassifier() if classifier is None else classifier    
    
    game_results = []
    
    for round_ in data_loader.get_available_rounds(game_number):
        # Load round image and labels (if available). Labels are discarded here.
        image, _ = data_loader[game_number, round_]
        
        # Extract from image dealer, cards, figure and suit of each player.
        dealer, cards, figures_suits = extractor(image)
        
        # Store results for score count and overlay.
        round_results = {'round': round_, 'D': dealer['player']}
        round_overlay_data = {'dealer_bbox': dealer['bbox'], 'dealer_pred': dealer['player']}
        
        for player, images in figures_suits.items():
            # Predict figure value and suit for each player and merge them into a card prediction.
            pred_figure = classifier.predict_figure(images['figure'], images['color'])
            pred_suit   = classifier.predict_suit  (images['suit']  , images['color'])
            card_pred   = pred_figure + pred_suit
            
            # Store results for score count and overlay.
            round_results['P' + str(player)] = card_pred
            round_overlay_data[player] = {'card_bbox'   : cards[player]['bbox'], 
                                          'card_pred'   : card_pred,
                                          'figure_bbox' : figures_suits[player]['bbox_figure'], 
                                          'suits_bbox_1': figures_suits[player]['bbox_suit_1'],
                                          'suits_bbox_2': figures_suits[player]['bbox_suit_2']}
        
        if draw_overlay:
            draw_and_plot_overlay(image, round_overlay_data, game_number, round_)
            
        # Store results for score count.
        game_results.append( round_results )
        
        
    # Calculate score for standard and advanced game rules.
    scores = {mode: {key: 0 for key in round_results.keys() if 'P' in key} for mode in ['standard', 'advanced']}
    figure_to_weight = {**{str(digit): digit for digit in range(10)}, **{'J': 10, 'Q': 11, 'K': 12}}
    players = ['P1', 'P2', 'P3', 'P4']
    
    game_results = pd.DataFrame(game_results).sort_values('round', ascending = True)
    for _, round_results in game_results.iterrows():        
        player_figures = {key: figure_to_weight[value[0]] for key, value in round_results.items() if key in players}
        player_suits   = {key: value[1]                   for key, value in round_results.items() if key in players}
        dealer_suit    = player_suits['P' + str(round_results['D'])]
        
        # Standard rules:
        max_figure = max(player_figures.values())
        winning_players = [player for player, figure in player_figures.items() if figure == max_figure]
        for player in winning_players:
            scores['standard'][player] += 1
            
        # Advanced rules:
        valid_player_figures = {key: value for key, value in player_figures.items() if player_suits[key] == dealer_suit}
        winning_player =  max(valid_player_figures, key = valid_player_figures.get)
        scores['advanced'][winning_player] += 1
        
    # Print results of game.   
    print_results(game_results[players].values,
                  game_results['D'].values,
                  [scores['standard'][player] for player in players], 
                  [scores['advanced'][player] for player in players])
    
    # Compute accuracy.
    if compute_accuracy:
        ground_truth_path = os.path.join(data_loader.data_dir, f'game{game_number}', f'game{game_number}.csv')
        assert os.path.isfile(ground_truth_path), \
               "To predict accuracy over game ground truth file must be available. No file named " + ground_truth_path
        
        ground_truth = pd.read_csv(ground_truth_path)
        
        assert not ground_truth.isnull().values.any(), \
               f"To predict accuracy over game ground truth file must be filled entirely. {ground_truth_path} contains missing entries."

        accuracy_standard = evaluate_game(game_results[players].values, ground_truth[players].values, mode_advanced = False)
        accuracy_advanced = evaluate_game(game_results[players].values, ground_truth[players].values, mode_advanced = True)
        print("Model Accuracy on Game {} is: Standard={:.3f}, Advanced={:.3f}".format(game_number, accuracy_standard, accuracy_advanced))
    
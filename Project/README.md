# IAPR: Project¶

### Description
Your task is to ready yourself for the final evaluation. The day of the exam we will give you a new folder with a new game. ! The digits on the cards differ from the one of the traning set. When given a new data folder with 13 images your should be able to:
#### Task 0
	- Plot an overlay for each round image that shows your detections and classification. You can for example plot bounding boxes around the cards/dealer token and add a text overlay with the name of the classes.
#### Task 1
	- Predict the rank of the card played by each player at each round (Standard rules).
	- Predict the number of points of each player according to Standard rules
#### Task 2
	- Detect which player is the selected dealer for each round.
	- Predict the rank and the suit of the card played by each player at each round (Advanced rules).
	- Predict the number of points of each player according to Advanced rules

### Install

- tensorflow==2.1.0
- numpy==1.19
- opencv-python==4.5
- pandas==1.2.4
- scikit-learn==0.24.1
- scipy==1.6.2
- matplotlib
- jupyter

### Code

In this github you will find:

   - train_games: Python code that run the test 
   - Dataset: Python code that run the test 
   - media: example images
   - scripts: 
      - data_generator.py: data augmentation for figures
      - data_loader.py: load input images
      - distortions.py: image transformations
      - extract.py: main class for cards, figures and suits extraction
      - figure_suits_classifier.py: main class for figure and suits classification
      - game.py: class to obtain game results
      - model.py: CNN model
      - plots.py: plot utils functions
      - utils.py: result printing
   - Models: Saved weights from training (both Linear SVC and CNN)
   - requirements.txt : required to be installed libraries
   - project.ipynb : Full project notebook
   - Model Train.ipynb: notebook where we train CNN model for figure predictions
   - Classification Suits.ipynb: notebook where we train Linear SVC for suits predictions


### Run


```
jupyter notebook project.ipynb
```  



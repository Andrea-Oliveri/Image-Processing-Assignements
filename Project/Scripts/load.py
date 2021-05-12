import os
import cv2	
import numpy as np
from tqdm import tqdm
import pandas as pd


class Dataset():

	def __init__(self,root='./train_games/'):
		# Initialize data, download, etc.
		# read with numpy or pandas
		self.root = root
		self.n_games = os.listdir(root)
		dfs = self._get_df()
		self.labels = pd.concat(dfs)


	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_games

	# not __getitem__
	def getitem(self,*game):
		# args are strings of folder/filenames
		# essentially getting the image and labels for a specific game and round 
		# if game one arg -> take entire game

		if not game: return

		spec = [g for g in game]
		if len(spec) == 1:
			#get full game
			g = spec[0]
			if not(g in os.listdir(self.root)): raise IndexError("file does not exist")

			path = self.root+'/'+g
			label = self.labels.loc[self.labels['game']==g.strip('/')]
			img = np.asarray([cv2.imread(path+'/'+str(rd)+'.jpg') for rd in label['round']])
			
		else:
			#consider two args
			g,rd = spec
			if not(g in os.listdir(self.root)): raise IndexError("file does not exist")
			path = self.root+'/'+g

			if not(rd in os.listdir(path)): raise IndexError("file does not exist")
			img = cv2.imread(path+'/'+rd)
			label = self.labels.loc[(self.labels['game']==g.strip('/')) & (self.labels['round']==int(rd.strip('.jpg')))]
		#np.array and dataframe
		return img,label


	def _get_df(self):
		# find the only csv in all game folders
		dfs = []
		for game in os.listdir(self.root):

			path = self.root+game 
			game_csv = [rd for rd in os.listdir(path) if rd.endswith('.csv')][0]

			df = pd.read_csv(path+'/'+game_csv,delimiter=',')
			df['game'] = game
			df = df.rename(columns={'Unnamed: 0':'round'})
			dfs.append(df)

		return dfs
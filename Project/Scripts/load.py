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

		
		# in order, read images arrays
		# call the public func below  (cannot do that on loading, images too large so takes time)
		self.images = []

	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_games


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

	def get_images(self):
		images = []
		for game in tqdm(os.listdir(self.root)):
			path = self.root+game 
			round_images = [cv2.imread(path+'/'+rd) for rd in os.listdir(path) if rd.endswith('.jpg')]
			images.append(round_images)

		return np.asarray(images)
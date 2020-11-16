# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:48:54 2020

@author: cpak1
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

masters = pd.read_csv("Master_Ranked_Games.csv")
grandmasters = pd.read_csv("GrandMaster_Ranked_Games.csv")
challenger = pd.read_csv("Challenger_Ranked_Games.csv")
combined = pd.concat((masters, grandmasters, challenger))
#lets drop information we don't need: we will only look at it from one team's perspective; it is pointless to look
#at variables like redWins, because it is a two player game if one wins, the other loses. similarly, we will take out
#all the red 'first variables', because if blue has obtained it first, the opposite value will be shown for red.
masters_adj = masters.drop(['redWins','redFirstBlood','redFirstTower', 'redFirstInhibitor', 'redFirstBaron'], axis = 1)
grandmasters_adj = grandmasters.drop(['redWins','redFirstBlood','redFirstTower', 'redFirstInhibitor', 'redFirstBaron'], axis = 1)
challenger_adj = challenger.drop(['redWins','redFirstBlood','redFirstTower', 'redFirstInhibitor', 'redFirstBaron'], axis = 1)
combined_adj = combined.drop(['redWins','redFirstBlood','redFirstTower', 'redFirstInhibitor', 'redFirstBaron'], axis = 1)

#from here, im only going to consider the data stuff for masters, so you know how it looks. -to Helios -from Chris

y = masters['blueWins']
X = masters.drop('blueWins', axis = 1)
#lets do some understanding of our data and visualization.
numeric = [cat for cat in combined.columns if combined[cat].dtype != 'O']
#we find that all of our data has already been processed/ or is all numeric
#looking at our variables
'''
for cat in combined.columns:
        plt.figure()
        hist, binedges = np.histogram(masters[cat], bins = 'auto')
        plt.hist(masters[cat], binedges)
        plt.title(cat)
'''        
#preprocessing:
print("Missing values in masters: ",masters.isnull().values.any())
#no missing values




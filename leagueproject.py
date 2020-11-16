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
#We have to split continous variables from binary variables - rip...
continous = [cat for cat in masters.columns if len(masters[cat].unique()) > 2]
#these continuous variables gotta be one hot encoded i think
binary = [cat for cat in masters.columns if len(masters[cat].unique()) == 2]
y = masters_adj['blueWins']
X = masters_adj.drop('blueWins', axis = 1)
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
#no missing values
print("Missing values in masters: ",masters.isnull().values.any())

#some feature engineering
blueVisionScore = masters["blueWardPlaced"] + masters["blueWardkills"]
redVisionScore = masters["redWardPlaced"] + masters["redWardkills"]
masters["blueVisionScore"] = blueVisionScore
masters["redVisionScore"] = redVisionScore

#train/test split:
from sklearn.model_selection import train_test_split

masters_train, masters_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
#model stuff:
nb_model = GaussianNB()
nb_model.fit(masters_train, y_train)

nb_y_pred = nb_model.predict(masters_test)
'''
rf_model = RandomForestRegressor()
rf_model.fit(masters_train, y_train)
rf_y_pred = rf_model.predict(masters_test)
'''
from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, rf_y_pred))
print(accuracy_score(y_test, nb_y_pred))
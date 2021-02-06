# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:48:54 2020

@author: cpak1
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
masters = pd.read_csv("Master_Ranked_Games.csv")
grandmasters = pd.read_csv("GrandMaster_Ranked_Games.csv")
challenger = pd.read_csv("Challenger_Ranked_Games.csv")
combined = pd.concat((masters, grandmasters, challenger))

#lets drop information we don't need: we will only look at it from one team's perspective; it is pointless to look
#at variables like redWins, because it is a two player game if one wins, the other loses. similarly, we will take out
#all the red 'first variables', because if blue has obtained it first, the opposite value will be shown for red.
numeric = [cat for cat in combined.columns if combined[cat].dtype != 'O']
nominal = [cat for cat in combined.columns if combined[cat].dtype == 'O']
plt.bar(["nominal","numeric"], [len(nominal),len(numeric)+1])
plt.title("data types")
#fig = plt.figure(figsize=(15.20,10.80))

'''
sn.set(font_scale=0.6)
sn.heatmap(corr, annot = True, annot_kws={"size":6})
plt.savefig("corr.eps", format = 'eps')
plt.show()
'''
y = combined['blueWins']
combined = combined.drop(['redWins', 'blueWins', 'gameId'], axis = 1)
combined1 = combined.copy()
#lets do some understanding of our data and visualization.
#we find that all of our data has already been processed/ or is all numeric
#looking at our variables

#preprocessing:
#no missing values
print("Missing values in masters: ",masters.isnull().values.any())
#incorrect values

#X["blueTotalLevel"].mask(X["blueTotalLevel"] > 90, np.mean(X["blueTotalLevel"]), inplace = True)
'''
plt.figure()
plt.hist(combined.blueDeath)
plt.title("Blue Deaths")
plt.figure()
plt.hist(combined.blueKills)
plt.title("Blue Kills")
plt.figure()
plt.hist(combined.blueAssist)
plt.title("Blue Assists")
plt.figure()
'''
#some feature engineering
plt.figure()
from scipy.stats import skew  
from scipy.stats import boxcox
skews = {}
for col in combined.columns:
        skews[col] = skew(combined[col])
plt.bar(skews.keys(), skews.values())
plt.xticks(rotation = 90)
plt.title("skew of attributes")
plt.figure()       
blueVisionScore = combined1["blueWardPlaced"] + combined1["blueWardkills"]
redVisionScore =combined1["redWardPlaced"] + combined1["redWardkills"]
combined1["blueDeath"] = combined1["blueDeath"].replace(0, 1)
combined1["redDeath"] = combined1["redDeath"].replace(0, 1)
combined1["blue_kda"] = ((combined1["blueKills"] + combined1["blueAssist"])/combined1["blueDeath"]).astype('float64')
combined1["red_kda"] = (combined1["redKills"] + combined1["redAssist"])/combined1["redDeath"]
combined1 = combined1.drop(['blueKills', 'redKills', 'blueDeath', 'redDeath', 'blueAssist', 'redAssist'], axis = 1)
combined1 = combined1.drop(["blueWardPlaced", "blueWardkills", "redWardPlaced", "redWardkills"], axis = 1)
combined1["blueVisionScore"] = blueVisionScore
combined1["redVisionScore"] = redVisionScore
normed = boxcox(np.array(combined1.blue_kda)+.001)[0]

'''
hist, binedges = np.histogram(combined1.blue_kda, bins='auto')
plt.hist(combined1.blue_kda, bins= binedges)
plt.title("Blue KDA Ratio")
'''
plt.figure()
continuous = {cat:len(combined[cat].unique()) for cat in combined.columns}
plt.bar(continuous.keys(), continuous.values())
plt.xticks(rotation = 90)
plt.title("unique values in attributes")
plt.figure()
'''
#binning
for cat in continuous:
        combined[cat] = pd.qcut(combined[cat],q=4)
        #print(X[cat])
        le = LabelEncoder()
        combined[cat] = le.fit_transform(combined[cat])
#train/test split:
from sklearn.model_selection import train_test_split
combined_train, combined_test, y_train, y_test = train_test_split(combined, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
#model stuff:
nb_model = GaussianNB()
nb_model.fit(combined_train, y_train)
nb_y_pred = nb_model.predict(combined_test)
rf_model = RandomForestClassifier()
rf_model.fit(combined_train, y_train)
rf_y_pred = rf_model.predict(combined_test)
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
#print(accuracy_score(y_test, rf_y_pred))
print("naive bayes: ", accuracy_score(y_test, nb_y_pred))
print("random forest: ", accuracy_score(y_test, rf_y_pred))
print("naive bayes rmse: ", math.sqrt(mean_squared_error(y_test, nb_y_pred)))
print("random forest rmse: ", math.sqrt(mean_squared_error(y_test, rf_y_pred)))


nb_cv = cross_val_score(nb_model, combined, y, cv = 8)
rf_cv = cross_val_score(rf_model, combined, y, cv = 8)
plt.figure()
plt.boxplot(nb_cv)
plt.title("naive_bayes accuracy boxplot")
plt.figure()
plt.boxplot(rf_cv)
plt.title("random forest accuracy boxplot")
plt.figure()
'''
'''
nbcm = confusion_matrix(y_test, nb_y_pred) 
sn.heatmap(nbcm, annot=True)
plt.show()
plt.figure()
rfcm = confusion_matrix(y_test,rf_y_pred,annot=True)
sn.heatmap(rfcm)
plt.show()
'''

'''
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
cv= ShuffleSplit(n_splits=10, test_size=0.2,random_state=0)
plot_learning_curve(rf_model, "Random Forest", combined, y, cv=cv, n_jobs=4)
'''
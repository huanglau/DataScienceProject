#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:54:03 2019

@author: vector
"""
import pandas as pd
import os
import sys
import sklearn
import numpy as np


# add current directory to path if not already in it
#if os.getcwd not in sys.path:
#    sys.path.append(os.getcwd())

#%%
#
#EFeatScores, ETFeat = ExtraTrees(Xtrain, ytrain, 100)
#
#XtestExtraTrees = pd.DataFrame([Xtest[col] for col in ETFeat], index = [title for title in ETFeat])
#
#XtestRFE = Xtest.iloc[:, ETFeat.support_]
#XvalRFE = Xval.iloc[:, RFEFeat.support_]
#XtrainRFE = Xtrain.iloc[:, RFEFeat.support_]
#%% feature selection method 2
# filter method. Uses chi2 to select the best features. This is a filter method
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif 
# chi2 must get only positive numbers

def KBest (X, y, numFeat, sOutDir, score_func=chi2):
    """
     filter method. Uses chi2 to select the best features. This is a filter method
     https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
     
     output:
         list of to numFeat featurenames
    """
    BestFeatures = SelectKBest(score_func=score_func, k = numFeat)
    fitSelectKBest = BestFeatures.fit(X.values, y.values)
    dfscores = pd.DataFrame(fitSelectKBest.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score'] 
    
    print(featureScores.nlargest(numFeat,'Score'))
    SelectKBestFeatures = featureScores.nlargest(numFeat,'Score').Specs.values
    
    # drop all other features
    XFeatures = X.drop([col for col in X.columns if col not in SelectKBestFeatures], axis='columns')
    
    # merge back with Y
    data = pd.concat([y, XFeatures], axis = 1)
    data.to_csv(sOutDir, index=False)

    return SelectKBestFeatures, XFeatures, data

#%%  Backwards feature selection, wrapper method


# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def LogRegFeat(X, y, numFeat, sOutDir):
    """
    uses backwards feature selection. Wrapper method. Involves fitting a model.
    """
    # Feature extraction
    model = LogisticRegression(random_state = 0)
    rfe = RFE(model, numFeat)
    fit = rfe.fit(X.values, y.values)
    features = np.array(X.columns)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    print(" the selected features are")
    print(features[fit.support_])
    Features = features[fit.support_]
    # drop all other features
    XFeatures = X.drop([col for col in X.columns if col not in Features], axis='columns')
    
    # merge back with Y
    data = pd.concat([y, XFeatures], axis = 1)
    data.to_csv(sOutDir, index=False)    

    return features[fit.support_], XFeatures, data

#%%

print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           #n_redundant=2, n_repeated=0, n_classes=8,
                           #n_clusters_per_class=1, random_state=0)

def BackwardFSCV(X, y):
    """
    Cross validation using bakwards feature selection. Does this over 5 folds and
    over multiple numbers of features. I.e. select the best 5 features, best 10 features, 
    best 15 features, etc
    
    RFE stands for recursive feature elimination
    """
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=15, min_features_to_select = 10, 
                  cv=StratifiedKFold(5),
                  scoring='accuracy', verbose = True, n_jobs = -1)
    rfecv.fit(X, y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
       # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    
    plt.plot(np.arange(0, np.shape(X)[1]+1, 5), rfecv.grid_scores_)
    plt.show()
    return rfecv

#%% Load data 
dfModel1 = pd.read_csv("../data/Model1WOStratum.csv")

y = pd.DataFrame(dfModel1['Ever used ilicit drugs'])
X = dfModel1.drop(['Ever used ilicit drugs'], axis='columns')


#for i in range(0, 150, 10):
#    KBestFeaturesNames, XFeatures, data = KBest (X, y, numFeat=i,
#                                             sOutDir='../data/SelectedFeatures/Model1_{}Feat_KBest_f_classif.csv'.format(i), score_func=chi2)
#for i in range(0, 150, 10):
#    KBestFeaturesNames, XFeatures, data = KBest (X, y, numFeat=i,
#                                             sOutDir='../data/SelectedFeatures/Model1_{}Feat_KBest_chi2.csv'.format(i), score_func=chi2)


rfecv = BackwardFSCV(X, y)

#%%  Backwards feature selection, wrapper method

    
#RFEFeat, RFEFit = LogRegFeat(X, y, numFeat=100, sOutDir='../data/SelectedFeatures/Model1_50Feat_KBest_f_classif.csv')


#%% feature selection

## Feature Importance with Extra Trees Classifier
#from sklearn.ensemble import ExtraTreesClassifier
#
#def ExtraTrees(X, y, numFeat):
#    """
#    Runs extra trees classifier on data. This
#    "This class implements a meta estimator that fits a number of randomized decision trees
#    (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
#    
#    Inputs: X: xvalue. dataframe
#            Y: yvalues. dataframe
#            numFeat: integer. Number of features to be selected
#    outputs:
#        -features with importance. 
#        -list of questions
#    
#    """
#    # feature extraction
#    model = ExtraTreesClassifier(random_state = 0)
#    model.fit(X.values, y.values)
#    print(model.feature_importances_)
#    print(pdQuestionsMode.columns[model.feature_importances_ > 0])
#    print(pdQuestionsMode.columns[model.feature_importances_ == 0])
#    feat_importances = pd.Series(model.feature_importances_, index=pdQuestionsMode.columns)
#    feat_importances.nlargest(numFeat).plot(kind='barh')
#    plt.show()
#    return feat_importances.nlargest(numFeat), (feat_importances.nlargest(numFeat)).axes
#    

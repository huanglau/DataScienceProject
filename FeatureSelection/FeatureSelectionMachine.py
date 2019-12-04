#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:54:03 2019

@author: vector
"""
import pandas as pd
import os
import sys
# for gpu usage
#import h2o4gpu as sklearn
#from h2o4gpu.linear_model import LogisticRegression
#from h2o4gpu.feature_selection import RFECV

#import sklearn
import numpy as np
import pickle
import matplotlib.pyplot as plt
print(__doc__)
#
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression




# chi2 must get only positive numbers

# add current directory to path if not already in it
#if os.getcwd not in sys.path:
#    sys.path.append(os.getcwd())

#%% feature selection method 2
# filter method. Uses chi2 to select the best features. This is a filter method
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# chi2 must get only positive numbers

#def KBest (X, y, numFeat, sOutDir, score_func=chi2):
#    """
#     filter method. Uses chi2 to select the best features. This is a filter method
#     https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
#     
#     output:
#         list of to numFeat featurenames
#    """
#    BestFeatures = SelectKBest(score_func=score_func, k = numFeat)
#    fitSelectKBest = BestFeatures.fit(X.values, y.values)
#    dfscores = pd.DataFrame(fitSelectKBest.scores_)
#    dfcolumns = pd.DataFrame(X.columns)
#    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#    featureScores.columns = ['Specs','Score'] 
#    
#    print(featureScores.nlargest(numFeat,'Score'))
#    SelectKBestFeatures = featureScores.nlargest(numFeat,'Score').Specs.values
#    
#    # drop all other features
#    XFeatures = X.drop([col for col in X.columns if col not in SelectKBestFeatures], axis='columns')
#    
#    # merge back with Y
#    data = pd.concat([y, XFeatures], axis = 1)
#    data.to_csv(sOutDir, index=False)
#
#    return SelectKBestFeatures, XFeatures, data
#
##%%  Backwards feature selection, wrapper method
#
#
#def LogRegFeat(X, y, numFeat, sOutDir):
#    """
#    uses backwards feature selection. Wrapper method. Involves fitting a model.
#    """
#    # Feature extraction
#    model = LogisticRegression(random_state = 0)
#    rfe = RFE(model, numFeat)
#    fit = rfe.fit(X.values, y.values)
#    features = np.array(X.columns)
#    print("Num Features: %s" % (fit.n_features_))
#    print("Selected Features: %s" % (fit.support_))
#    print("Feature Ranking: %s" % (fit.ranking_))
#    print(" the selected features are")
#    print(features[fit.support_])
#    Features = features[fit.support_]
#    # drop all other features
#    XFeatures = X.drop([col for col in X.columns if col not in Features], axis='columns')
#    
#    # merge back with Y
#    data = pd.concat([y, XFeatures], axis = 1)
#    data.to_csv(sOutDir, index=False)    
#
#    return features[fit.support_], XFeatures, data

#%%



def BackwardFSCV(X, y, sOutDir, step=1, min_features_to_select = 10, kFolds = 5):
    """
    Cross validation using bakwards feature selection. Does this over 5 folds and
    over multiple numbers of features. I.e. select the best 5 features, best 10 features, 
    best 15 features, etc
    
    RFE stands for recursive feature elimination
    """
    os.makedirs(sOutDir, exist_ok = True)
    
    # Create the RFE object and compute a cross-validated score.  
    LogReg = LogisticRegression()
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=LogReg, step=step,
                  scoring='accuracy', verbose = True, n_jobs = -1)
    rfecv.fit(X, y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
       # Plot number of features VS. cross-validation scores
    fig = plt.figure()

    plt.xlabel("Number of features selected", figure=fig)
    plt.ylabel("Cross validation score (nb of correct classifications)", figure=fig)
    
    plt.plot(np.array(range(1, len(rfecv.grid_scores_) + 1))*step, rfecv.grid_scores_, figure=fig)
    fig.savefig(os.path.join(sOutDir, 'FeatureSelectionRFECV.png'), dpi = 300)
    plt.show()
    
    with open(os.path.join(sOutDir, 'objs.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([rfecv], f)
    
    # drop variables not selected
    XFeatures = X.drop(X.columns[rfecv.support_==False], axis='columns')

    XFeatures.to_csv(os.path.join(sOutDir, 'RFECV.csv'))
    return rfecv


def Backwards(dfModel, sXColumn , sOutDir):
    y = pd.DataFrame(dfModel[sXColumn])
    X = dfModel.drop([sXColumn], axis='columns')
    BackwardFSCV(X, y, sOutDir)
    
    
#%%
#import pandas as pd
#import os
#import sys
#import sklearn
#import numpy as np
#import pickle
#import matplotlib.pyplot as plt
#rfecv = pickle.load( open( "/home/vector/Documents/Western/Courses/CS4414/project/data/DataScienceProject/FeatureSelection/RFECV_Model2.pkl", "rb" ) )
#plt.plot(np.flip(np.arange(133, 0, -15)), rfecv[0].grid_scores_[1:])
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")


#%% Load data 
#dfModel1 = pd.read_csv("../data/Model1WOStratum.csv")
#
#y = pd.DataFrame(dfModel1['Ever used ilicit drugs'])
#X = dfModel1.drop(['Ever used ilicit drugs'], axis='columns')

#for i in range(0, 150, 10):
#    KBestFeaturesNames, XFeatures, data = KBest (X, y, numFeat=i,
#                                             sOutDir='../data/SelectedFeatures/Model1_{}Feat_KBest_f_classif.csv'.format(i), score_func=chi2)
#for i in range(0, 150, 10):
#    KBestFeaturesNames, XFeatures, data = KBest (X, y, numFeat=i,
#                                             sOutDir='../data/SelectedFeatures/Model1_{}Feat_KBest_chi2.csv'.format(i), score_func=chi2)

#dfModel2 = pd.read_csv("../data/Model2WOStratum.csv")
#
#y = pd.DataFrame(dfModel2['Freq Drug Use'])
#X = dfModel2.drop(['Freq Drug Use'], axis='columns')
#sOutDir = "../data/SelectedFeatures/Model2WOStratum_RFECV135.csv"
#rfecv = BackwardFSCV(X, y, sOutDir)

#dfModel = pd.read_csv("../data/Model1NoDrugQs.csv")
#dfModel = dfModel.drop([dfModel.columns[i] for i in range(1,100) ], axis='columns')
#Backwards(dfModel, 'Ever used ilicit drugs' , "../data/SelectedFeatures/Model1NoDrugQs")

#%% model 1
    

dfModel = pd.read_csv("../data/Model1.csv")
Backwards(dfModel, 'Ever used ilicit drugs' , "../data/SelectedFeatures/Model1")

dfModel = pd.read_csv("../data/Model1NoDrugQs.csv")
Backwards(dfModel, 'Ever used ilicit drugs' , "../data/SelectedFeatures/Model1NoDrugQs")

dfModel = pd.read_csv("../data/Model1NoDrugQsNoDummies.csv")
Backwards(dfModel, 'Ever used ilicit drugs' , "../data/SelectedFeatures/Model1NoDrugQsNoDummies")

dfModel = pd.read_csv("../data/Model1NoDummies.csv")
Backwards(dfModel, 'Ever used ilicit drugs' , "../data/SelectedFeatures/Model1NoDummies")


#%% model 2

dfModel = pd.read_csv("../data/Model2.csv")
Backwards(dfModel, 'Freq Drug Use' , "../data/SelectedFeatures/Model2")

dfModel = pd.read_csv("../data/Model2NoDrugQs.csv")
Backwards(dfModel, 'Freq Drug Use' , "../data/SelectedFeatures/Mode2NoDrugQs")

dfModel = pd.read_csv("../data/Model2NoDrugQsNoDummies.csv")
Backwards(dfModel, 'Freq Drug Use' , "../data/SelectedFeatures/Model2NoDrugQsNoDummies")

dfModel = pd.read_csv("../data/Model2NoDummies.csv")
Backwards(dfModel, 'Freq Drug Use', "../data/SelectedFeatures/Mode2NoDummies")
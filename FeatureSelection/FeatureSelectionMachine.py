#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:54:03 2019

@author: vector
"""


EFeatScores, ETFeat = ExtraTrees(Xtrain, ytrain, 100)

XtestExtraTrees = pd.DataFrame([Xtest[col] for col in ETFeat], index = [title for title in ETFeat])

XtestRFE = Xtest.iloc[:, ETFeat.support_]
XvalRFE = Xval.iloc[:, RFEFeat.support_]
XtrainRFE = Xtrain.iloc[:, RFEFeat.support_]
#%% feature selection method 2
# filter method. Uses chi2 to select the best features. This is a filter method
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def KBest (X, y, numFeat, score_func=chi2):
    """
     filter method. Uses chi2 to select the best features. This is a filter method
     https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
     
     output:
         list of to numFeat featurenames
    """
    BestFeatures = SelectKBest(score_func=score_func, k = numFeat)
    fitSelectKBest = BestFeatures.fit(X.values, y.values)
    dfscores = pd.DataFrame(fitSelectKBest.scores_)
    dfcolumns = pd.DataFrame(Xtrain.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score'] 
    
    print(featureScores.nlargest(numFeat,'Score'))
    SelectKBestFeatures = featureScores.nlargest(numFeat,'Score').Specs.values
    return SelectKBestFeatures

# as expected most of the features are drug related 
KBestFeatures = KBest (Xtrain, ytrain, numFeat=100, score_func=chi2)

#%% reanalyze with no drug wuestins at all
#pdIlicitDrugEverUsed, pdQuestionsNoDrugs = preprocessingNoDrugs(data)
#pdQuestionsNoDrugsMode = pdQuestionsNoDrugs.fillna(value = Mode)
#
#model = ExtraTreesClassifier(seed = 0)
#model.fit(pdQuestionsNoDrugsMode.values, pdIlicitDrugEverUsed.values)
#print(model.feature_importances_)
#print(pdQuestionsNoDrugsMode.columns[model.feature_importances_ > 0])
#print(pdQuestionsNoDrugsMode.columns[model.feature_importances_ == 0])
#feat_importances = pd.Series(model.feature_importances_, index=pdQuestionsNoDrugsMode.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()
#
#BestFeatures = SelectKBest(score_func=chi2, k = 10)
#fitSelectKBest = BestFeatures.fit(pdQuestionsNoDrugsMode.values, pdIlicitDrugEverUsed.values)
#dfscores = pd.DataFrame(fitSelectKBest.scores_)
#dfcolumns = pd.DataFrame(pdQuestionsNoDrugsMode.columns)
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score'] 
#
#print(featureScores.nlargest(10,'Score'))
# as expected most of the features are drug related 

#%%  Backwards feature selection, wrapper method

# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def LogRegFeat(X, y, numFeat):
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
    

    return features[fit.support_], fit
    
RFEFeat, RFEFit = LogRegFeat(Xtrain, ytrain, numFeat=100)

# apply select only the selected features in xtrain test and val
XtestRFE = Xtest.iloc[:, RFEFit.support_]
XvalRFE = Xval.iloc[:, RFEFeat.support_]
XtrainRFE = Xtrain.iloc[:, RFEFeat.support_]
XtestRFE.to_csv("FeatureSelection/XtestRFE100.csv")

#%% feature selection

# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

def ExtraTrees(X, y, numFeat):
    """
    Runs extra trees classifier on data. This
    "This class implements a meta estimator that fits a number of randomized decision trees
    (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
    
    Inputs: X: xvalue. dataframe
            Y: yvalues. dataframe
            numFeat: integer. Number of features to be selected
    outputs:
        -features with importance. 
        -list of questions
    
    """
    # feature extraction
    model = ExtraTreesClassifier(random_state = 0)
    model.fit(X.values, y.values)
    print(model.feature_importances_)
    print(pdQuestionsMode.columns[model.feature_importances_ > 0])
    print(pdQuestionsMode.columns[model.feature_importances_ == 0])
    feat_importances = pd.Series(model.feature_importances_, index=pdQuestionsMode.columns)
    feat_importances.nlargest(numFeat).plot(kind='barh')
    plt.show()
    return feat_importances.nlargest(numFeat), (feat_importances.nlargest(numFeat)).axes
    

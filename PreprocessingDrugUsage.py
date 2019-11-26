#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:36:19 2019

@author: vector
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# load up data
data = pd.read_csv('nationalYRBS20172015')#, delimiter=' ')
data.columns

# show off some example data
data['q49'] # how many times used any cocaine
data['q51'] # how many times ever used heroin
data['q52'] # how many times ever used methaphetamines

data['qcurrentcocaine']
data['qcurrentheroin']
data['qcurrentmeth']

# data that is y/n questions
lYN = ['q19', 'q23', 'q24','q25', 'q26', 'q27', 'q29', 'q30', 'q34','q39', 
       'qn39', 'q58', 'q59','q63', 'qn63', 'q64', 'qn64']
lCatagorical = ['q36', 'qn36', 'q43', 'qn43', 'q65', 'qn65', 'q66', 'q67', 
                'q69', 'qn69', 'q85', 'q87'] # maybe Q68
lDemographics = ['Unnamed: 0', 'Unnamed: 0.1', 'sitecode', 'sitename', 'sitetype', 
                 'sitetypenum', 'year','survyear', 'weight', 'stratum', 'PSU', 'record']
# 49, 50, 51, 52, 53, and 57
lDrugs = ['q49', 'q50', 'q51', 'q52', 'q53', 'q57', 'qcurrentcocaine',
                             'qcurrentmeth','qcurrentheroin', 'qhallucdrug',
                             'qnhallucdrug', 'qn49', 'qn50','qn57',
                             'qn51', 'qn52', 'qn53']
print(data.columns)
#%% Experimentation for predicting if teen used ANY of the ilicit drugs 
# 0, if never used, 1 if used

# data preprocessing
def preprocessingIlicitDrug(data, lDemographics, Mode = True):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    """
    #clean out rows (persons) with more than 150 NAN values
    data = data.dropna(thresh=150, axis='rows')
    
    # remove obvious variables that relate too closely to the prediction
    pdQuestions = data.drop(lDrugs, axis = 'columns') 

    # get prediction variable
    #49, 50, 51, 52, 53, and 57
    pdIlicitDrugEverUsed = pd.DataFrame((data['q49'].values >= 2) + (data['q50'].values  >= 2) + (data['q51'].values  >= 2)
                            + (data['q52'].values>= 2) + (data['q53'].values  >= 2) + (data['q57'].values >= 2), columns=['Ever used ilicit drugs'])

    # remove demographic data
    pdQuestions = pdQuestions.drop(lDemographics, axis='columns')

    # process dependent variables
    # make dummies
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = lCatagorical, drop_first=True)
    # convert y/n answers to dummies lYN
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = lYN, drop_first=True)

    # remove columns (questions) with more than N nans
    NanThresh = 10000
    NanColumns = np.array( [[sColumn, np.sum(pd.isnull(pdQuestions[sColumn].values[:]))] for sColumn in pdQuestions if np.sum(pd.isnull(pdQuestions[sColumn].values[:])) > NanThresh])
    pdQuestions = pdQuestions.drop(NanColumns[:,0], axis = 'columns')

    if Mode == True:
        # fills in Nans with most frequent catagory
        Mode = {i:pdQuestions[i].value_counts().index[0] for i in pdQuestions.columns}
        pdQuestions = pdQuestions.fillna(value = Mode)
    return NanColumns, pdIlicitDrugEverUsed, pdQuestions

# data preprocessing
def preprocessingNoDrugs(data, Mode = True):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    """
    #clean out rows (persons) with more than 150 NAN values
    data = data.dropna(thresh=150, axis='rows')

    # get prediction variable
    #49, 50, 51, 52, 53, and 57
    pdIlicitDrugEverUsed = pd.DataFrame((data['q49'].values >= 2) + (data['q50'].values  >= 2) + (data['q51'].values  >= 2)
                            + (data['q52'].values>= 2) + (data['q53'].values  >= 2) + (data['q57'].values >= 2), columns=['Ever used ilicit drugs'])

    # remove all drug related factors (any drugs at all)
    pdQuestions = data.drop(['q{}'.format(str(i)) for i in range(30,58)], axis = 'columns') 
    # drug questions for alcohol, cigarettes etc
    pdQuestions = pdQuestions.drop(['qn{}'.format(str(i)) for i in range(30,58)], axis = 'columns') 
    pdQuestions = pdQuestions.drop(['qcurrentcocaine','qcurrentmeth','qcurrentheroin', 'qhallucdrug',
                             'qnhallucdrug'], axis = 'columns') 

    # remove demographic data
    pdQuestions = pdQuestions.drop(lDemographics, axis='columns')

    # process dependent variables
    # make dummies
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = [ 'q65', 'qn65', 'q66', 'q67', 
                'q69', 'qn69', 'q85', 'q87'] , drop_first=True)

    # remove columns (questions) with more than N nans
    NanThresh = 10000
    NanColumns = np.array( [[sColumn, np.sum(pd.isnull(data[sColumn].values[:]))] for sColumn in pdQuestions if np.sum(pd.isnull(pdQuestions[sColumn].values[:])) > NanThresh])
    pdQuestions = pdQuestions.drop(NanColumns[:,0], axis = 'columns')
    
    if Mode == True:
        # fills in Nans with most frequent catagory
        Mode = {i:pdQuestions[i].value_counts().index[0] for i in pdQuestions.columns}
        pdQuestions = pdQuestions.fillna(value = Mode)

    return pdIlicitDrugEverUsed, pdQuestions

NanColumns, pdIlicitDrugEverUsed, pdQuestions = preprocessingIlicitDrug(data, lDemographics)


# drop all qn* questions. Drop compound questions
#pdQuestions = pdQuestions.drop([question for question in pdQuestions.columns if 'n' in question], axis='columns')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# split into train test
Xtrain, Xtest, ytrain, ytest = train_test_split( pdQuestions,
                                                pdIlicitDrugEverUsed, 
                                                test_size=0.015173780800382761, # to get 444 in test set
                                                random_state=0)
# split train into train val
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain,
                                                ytrain, 
                                                test_size=0.29496477773536456, # to get 8500 in val set
                                                random_state=0)
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
    
RFEFeat RFEFit = LogRegFeat(Xtrain, ytrain, numFeat=100)

# apply select only the selected features in xtrain test and val
XtestRFE = Xtest.iloc[:, RFEFit.support_]
XvalRFE = Xval.iloc[:, RFEFeat.support_]
XtrainRFE = Xtrain.iloc[:, RFEFeat.support_]
XtestRFE.to_csv("FeatureSelection/XtestRFE100.csv")

#%%


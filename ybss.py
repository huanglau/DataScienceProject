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
print(data.columns)
#%% Experimentation for predicting if teen used ANY of the ilicit drugs 
# 0, if never used, 1 if used

# data preprocessing
def preprocessing(data, lDemographics):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    """
    #clean out rows (persons) with more than 150 NAN values
    data = data.dropna(thresh=150, axis='rows')
    
    # remove obvious variables that relate too closely to the prediction
    pdQuestions = data.drop(['q49', 'q51', 'q52', 'q53', 'q57', 'qcurrentcocaine',
                             'qcurrentmeth','qcurrentheroin', 'qhallucdrug',
                             'qnhallucdrug', 'qn49', 'qn57',
                             'qn51', 'qn52', 'qn53'], axis = 'columns') 

    # get prediction variable
    pdIlicitDrugEverUsed = pd.DataFrame((data['q49'].values > 1) *  (data['q51'].values  >= 1) * (data['q52'].values>= 1) * (data['q59'].values >= 1), columns=['Ever used ilicit drugs'])

    # remove demographic data
    pdQuestions = pdQuestions.drop(lDemographics, axis='columns')

    # process dependent variables
    # make dummies
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = lCatagorical, drop_first=True)

    # remove columns (questions) with more than N nans
    NanThresh = 10000
    NanColumns = np.array( [[sColumn, np.sum(pd.isnull(data[sColumn].values[:]))] for sColumn in pdQuestions if np.sum(pd.isnull(pdQuestions[sColumn].values[:])) > NanThresh])
    pdQuestions = pdQuestions.drop(NanColumns[:,0], axis = 'columns')

    return NanColumns, pdIlicitDrugEverUsed, pdQuestions

# data preprocessing
def preprocessingNoDrugs(data):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    """
    #clean out rows (persons) with more than 150 NAN values
    data = data.dropna(thresh=150, axis='rows')

    # get prediction variable
    #TODO: check if this is correct
    pdIlicitDrugEverUsed = pd.DataFrame((data['q49'].values >= 1) *  (data['q51'].values  >= 1) * (data['q52'].values>= 1) * (data['q59'].values >= 1), columns=['Ever used ilicit drugs'])


    # remove all drug related factors (any drugs at all)
    pdQuestions = data.drop(['q{}'.format(str(i)) for i in range(30,58)], axis = 'columns') 
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

    return pdIlicitDrugEverUsed, pdQuestions

NanColumns, pdIlicitDrugEverUsed, pdQuestions = preprocessing(data, lDemographics)
# fills in Nans with most frequent catagory
Mode = {i:pdQuestions[i].value_counts().index[0] for i in pdQuestions.columns}
pdQuestionsMode = pdQuestions.fillna(value = Mode)


# drop all qn* questions. Drop compound questions
#pdQuestions = pdQuestions.drop([question for question in pdQuestions.columns if 'n' in question], axis='columns')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# split into train test
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

# feature extraction
model = ExtraTreesClassifier()
model.fit(pdQuestionsMode.values, pdIlicitDrugEverUsed.values)
print(model.feature_importances_)
print(pdQuestionsMode.columns[model.feature_importances_ > 0])
print(pdQuestionsMode.columns[model.feature_importances_ == 0])
feat_importances = pd.Series(model.feature_importances_, index=pdQuestionsMode.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#%% feature selection method 2
# filter method. Uses chi2 to select the best features
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

BestFeatures = SelectKBest(score_func=chi2, k = 100)
fitSelectKBest = BestFeatures.fit(Xtrain.values, ytrain.values)
dfscores = pd.DataFrame(fitSelectKBest.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 

print(featureScores.nlargest(100,'Score'))
SelectKBestFeatures = featureScores.nlargest(100,'Score').Specs.values
# as expected most of the features are drug related 


#%% reanalyze with no drug wuestins at all
pdIlicitDrugEverUsed, pdQuestionsNoDrugs = preprocessingNoDrugs(data)
pdQuestionsNoDrugsMode = pdQuestionsNoDrugs.fillna(value = Mode)

model = ExtraTreesClassifier()
model.fit(pdQuestionsNoDrugsMode.values, pdIlicitDrugEverUsed.values)
print(model.feature_importances_)
print(pdQuestionsNoDrugsMode.columns[model.feature_importances_ > 0])
print(pdQuestionsNoDrugsMode.columns[model.feature_importances_ == 0])
feat_importances = pd.Series(model.feature_importances_, index=pdQuestionsNoDrugsMode.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

BestFeatures = SelectKBest(score_func=chi2, k = 10)
fitSelectKBest = BestFeatures.fit(pdQuestionsNoDrugsMode.values, pdIlicitDrugEverUsed.values)
dfscores = pd.DataFrame(fitSelectKBest.scores_)
dfcolumns = pd.DataFrame(pdQuestionsNoDrugsMode.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 

print(featureScores.nlargest(10,'Score'))
# as expected most of the features are drug related 

#%%  Backwards feature selection, wrapper method

# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 100)
fit = rfe.fit(Xtrain, ytrain)
features = np.array(Xtrain.columns)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
print(" the selected features are")
print(features[fit.support_])

# apply select only the selected features in xtrain test and val
XtestRFE = Xtest.iloc[:, fit.support_]
XvalRFE = Xval.iloc[:, fit.support_]
XtrainRFE = Xtrain.iloc[:, fit.support_]

XtestRFE.to_csv("XtestRFE100.csv")

#%%


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
       'q58', 'q59','q63',  'q64']
lCatagorical = ['q36', 'q43', 'qn43', 'q65',  'q66', 'q67', 
                'q69', 'q85', 'q87'] # maybe Q68
# remove stheight. Only 2885 users answered this. Removed bmipct. This is the bmi percentail
lDemographics = ['Unnamed: 0', 'Unnamed: 0.1', 'sitecode', 'sitename', 'sitetype', 
                 'sitetypenum', 'year','survyear',  'PSU', 'record', 'stheight', 'bmipct','stratum'] # stratum
lNotInBoth20172015 = ['grade', 'bmipct', 'q10', 'q16', 'q18', 'q23', 'q35', 'q63']
# 49, 50, 51, 52, 53, and 57
lDrugs = ['q49', 'q50', 'q51', 'q52', 'q53', 'q57', 'qcurrentcocaine',
                             'qcurrentmeth','qcurrentheroin', 'qhallucdrug']
#                             'qnhallucdrug', 'qn49', 'qn50','qn57',
#                            'qn51', 'qn52', 'qn53']
print(data.columns)
#%% Experimentation for predicting if teen used ANY of the ilicit drugs 
# 0, if never used, 1 if used

# data preprocessing
def preprocessingIlicitDrug(data, lDemographics, sOutDir = 'data/Model1.csv', DropAllDrugs = False):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    """
    # get prediction variable
    #49, 50, 51, 52, 53, and 57
    pdIlicitDrugEverUsed = pd.DataFrame((data['q49'].values >= 2) + (data['q50'].values  >= 2) + (data['q51'].values  >= 2)
                            + (data['q52'].values>= 2) + (data['q53'].values  >= 2) + (data['q57'].values >= 2), columns=['Ever used ilicit drugs'])
    
    # remove questions that werrn't asked in both years
    df15 = data[data['year'] == 2015]
    df17 = data[data['year'] == 2017]
    TooManyNans15 =[ Col for Col in df15.columns if df15[Col].isna().sum() >= len(df15)]
    TooManyNans17 =[ Col for Col in df17.columns if df17[Col].isna().sum() >= len(df17)]
    TooManyNans = list(set(TooManyNans15) | set(TooManyNans17))
    pdQuestions = data.drop(TooManyNans, axis='columns')
    
    # drop out all 'qn*' data
    pdQuestions = pdQuestions.drop([i for i in pdQuestions.columns if 'qn' in i], axis = 'columns') 

    # remove demographic data, and useless data
    pdQuestions = pdQuestions.drop(lDemographics, axis='columns')

    # remove columns (questions) with more than N nans
#    NanThresh = 20000#len(pdQuestions)-1
#    NanColumns = np.array( [[sColumn, np.sum(pd.isnull(pdQuestions[sColumn].values[:]))] for sColumn in pdQuestions if np.sum(pd.isnull(pdQuestions[sColumn].values[:])) > NanThresh])
#    pdQuestions = pdQuestions.drop(NanColumns[:,0], axis = 'columns')
    
    # remove obvious variables that relate too closely to the prediction
    # only remove if they weren't removed alreadt
    if len([drug for drug in lDrugs if drug in pdQuestions.columns]) > 0:
        pdQuestions = pdQuestions.drop([drug for drug in lDrugs if drug in pdQuestions.columns], axis = 'columns') 
    if DropAllDrugs == True:
        # remove all drug related factors (any drugs at all)
        pdQuestions = pdQuestions.drop(['q{}'.format(str(i)) for i in range(30,58) if 'q{}'.format(str(i))  in pdQuestions.columns], axis = 'columns') 
    
    #clean out rows (persons) with more than half the answers missing
    pdQuestions = pdQuestions.dropna(thresh=int(len(pdQuestions.columns)/2), axis='rows')
    
    # standardize height and weight
    numeric_variables = [ 'weight', 'bmi']
    #Subtract the mean
    pdQuestions.loc[:, numeric_variables] = (pdQuestions.loc[:, numeric_variables] - pdQuestions.loc[:, numeric_variables].mean())

<<<<<<< HEAD
XtestExtraTrees = pd.DataFrame([Xtest[col] for col in ETFeat], index = ETFeat[0])
=======
    #Divide by the standard deviation
    pdQuestions.loc[:, numeric_variables] = pdQuestions.loc[:, numeric_variables]/pdQuestions.loc[:,numeric_variables].std()
>>>>>>> master

    # fills in Nans with 0 for non-answer
    pdQuestions = pdQuestions.fillna(value = 0)

    # process dependent variables, make dummies
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = [col for col in lCatagorical if col in pdQuestions.columns], drop_first=True, dummy_na=True)
    # convert y/n answers to dummies lYN
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = [col for col in lYN if col in pdQuestions.columns],  drop_first=True, dummy_na=True)

    # save as csv
    # stack the two pandas into one file
    pdQuestions = pdQuestions.fillna(value = 0)

    # combine the data to for something to ouput to csv
    dfAllData = pd.concat([pdIlicitDrugEverUsed, pdQuestions], axis = 1)

    dfAllData.to_csv(sOutDir, index=False)    
    return pdIlicitDrugEverUsed, pdQuestions

# run to output cleaned data
preprocessingIlicitDrug(data, lDemographics, sOutDir = 'data/Model1WOStratum.csv', DropAllDrugs = False)
preprocessingIlicitDrug(data, lDemographics, sOutDir = 'data/Model1NoDrugQsInXWOStratum.csv', DropAllDrugs = True)
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

# data that is y/n questions
lYN = ['q19', 'q23', 'q24','q25', 'q26', 'q27', 'q29', 'q30', 'q34','q39', 
       'q58', 'q59','q63',  'q64']
lCatagorical = ['q36', 'q43', 'qn43', 'q65',  'q66', 'q67', 
                'q69', 'q85', 'q87'] # maybe Q68
# remove stheight. Only 2885 users answered this. Removed bmipct. This is the bmi percentail
lDemographics = ['Unnamed: 0', 'Unnamed: 0.1', 'sitecode', 'sitename', 'sitetype', 
                 'sitetypenum', 'year','survyear',  'PSU', 'record', 'stheight', 'bmipct']#, 'stratum'] # stratum
lNotInBoth20172015 = ['grade', 'bmipct', 'q10', 'q16', 'q18', 'q23', 'q35', 'q63']
# questions about drug frequency
# 32, 35, 37, 38, 42
lFreq = ['q32', 'q35', 'q37', 'q38', 'q42']
#                             'qnhallucdrug', 'qn49', 'qn50','qn57',
#                            'qn51', 'qn52', 'qn53']
print(data.columns)
#%% Experimentation for predicting if teen used ANY of the ilicit drugs 
# 0, if never used, 1 if used

# data preprocessing
def preprocessingDrugFreq(data, lDemographics, sOutDir, DropAllDrugs = False):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    
    
    The highest frequency of response is used from student answers to questions
    32 (cigarettes), 35 (electronic vapor product), 37 (chewing tobacco, snuff, dip, snus, or dissolvable tobacco products),
    38 (cigars, cigarettes or little cigars), and 42 (alcohol). 

    """
    nanRows = np.arange(0,len(data))[np.isnan(data['q32'].values)*np.isnan(data['q35'].values)*np.isnan(data['q37'].values)*np.isnan(data['q38'].values)*np.isnan(data['q42'].values)]
    
    # drop rows where they didn't answer all of the questions for pd Freq
    data = data.drop(nanRows, axis='rows')

    # get prediction variable
    # 32, 35, 37, 38, 42
    freqThresh = 2
    pdFreq = pd.DataFrame((data['q32'].values >= freqThresh) + (data['q35'].values  >= freqThresh) 
                            + (data['q37'].values  >= freqThresh)
                            + (data['q38'].values>= freqThresh)
                            + (data['q42'].values  >= freqThresh), columns=['Freq Drug Use'])
    
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
   
    # remove obvious variables that relate too closely to the prediction
    # only remove if they weren't removed alreadt
    if len([freq for freq in lFreq if freq in pdQuestions.columns]) > 0:
        pdQuestions = pdQuestions.drop([freq for freq in lFreq if freq in pdQuestions.columns], axis = 'columns') 
    if DropAllDrugs == True:
        # remove all drug related factors (any drugs at all)
        pdQuestions = pdQuestions.drop(['q{}'.format(str(i)) for i in range(30,58) if 'q{}'.format(str(i))  in pdQuestions.columns], axis = 'columns') 
    
    # standardize height and weight
    numeric_variables = ['weight', 'bmi']
    #Subtract the mean
    pdQuestions.loc[:, numeric_variables] = (pdQuestions.loc[:, numeric_variables] - pdQuestions.loc[:, numeric_variables].mean())

    #Divide by the standard deviation
    pdQuestions.loc[:, numeric_variables] = pdQuestions.loc[:, numeric_variables]/pdQuestions.loc[:,numeric_variables].std()

    #clean out rows (persons) with more than half the answers missing
    pdQuestions = pdQuestions.dropna(thresh=int(len(pdQuestions.columns)/2), axis='rows')

    # process dependent variables, make dummies
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = [col for col in lCatagorical if col in pdQuestions.columns], drop_first=True, dummy_na=True)
    # convert y/n answers to dummies lYN
    pdQuestions = pd.get_dummies(pdQuestions, prefix_sep = "_", columns = [col for col in lYN if col in pdQuestions.columns],  drop_first=True, dummy_na=True)

    # save as csv
    # stack the two pandas into one file
    pdQuestions = pdQuestions.fillna(value = 0)

    # combine the data to for something to ouput to csv
    dfAllData = pd.concat([pdFreq, pdQuestions], axis = 1)
    #clean out rows (persons) with more than half the answers missing
#    dfAllData = dfAllData.dropna(thresh=int(len(pdQuestions.columns)/2), axis='rows')
    dfAllData.to_csv(sOutDir, index=False)    
    return pdFreq, pdQuestions

# run to output cleaned data
preprocessingDrugFreq(data, lDemographics, sOutDir = 'data/Model2Wtratum.csv', DropAllDrugs = False)
preprocessingDrugFreq(data, lDemographics, sOutDir = 'data/Model2NoDrugQsInXWStratum.csv', DropAllDrugs = True)
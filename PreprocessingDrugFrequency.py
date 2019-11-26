"""

The second model is intended to help identify youth at risk of or with addictive 
patterns of tobacco and/or alcohol consumption, and will predict the frequency 
of alcohol and/or tobacco product use in the past 30 days. The outcome is operationalized 
as a categorical variable with seven response categories ranging from zero to all 30 days.
 The highest frequency of response is used from student answers to questions 32 (cigarettes),
 35 (electronic vapor product), 37 (chewing tobacco, snuff, dip, snus, or dissolvable tobacco products),
 38 (cigars, cigarettes or little cigars), and 42 (alcohol). 

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# load up data
data = pd.read_csv('nationalYRBS20172015')#, delimiter=' ')
data.columns

# data that is y/n questions
lYN = ['q19', 'q23', 'q24','q25', 'q26', 'q27', 'q29', 'q30', 'q34','q39', 
       'qn39', 'q58', 'q59','q63', 'qn63', 'q64', 'qn64']
lCatagorical = ['q36', 'qn36', 'q43', 'qn43', 'q65', 'qn65', 'q66', 'q67', 
                'q69', 'qn69', 'q85', 'q87'] # maybe Q68
lDemographics = ['Unnamed: 0', 'Unnamed: 0.1', 'sitecode', 'sitename', 'sitetype', 
                 'sitetypenum', 'year','survyear', 'weight', 'stratum', 'PSU', 'record']
lDrugs = ['q49', 'q50', 'q51', 'q52', 'q53', 'q57', 'qcurrentcocaine',
                             'qcurrentmeth','qcurrentheroin', 'qhallucdrug',
                             'qnhallucdrug', 'qn49', 'qn50','qn57',
                             'qn51', 'qn52', 'qn53']
print(data.columns)
#%% Experimentation for predicting drug frequency

# data preprocessing
def preprocessingDrugFreq(data, lDemographics, Mode = True):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    
    
    The highest frequency of response is used from student answers to questions
    32 (cigarettes), 35 (electronic vapor product), 37 (chewing tobacco, snuff, dip, snus, or dissolvable tobacco products),
    38 (cigars, cigarettes or little cigars), and 42 (alcohol). 

    """
    #clean out rows (persons) with more than 150 NAN values
    data = data.dropna(thresh=150, axis='rows')
    
    # remove obvious variables that relate too closely to the prediction
    pdQuestions = data.drop(lDrugs, axis = 'columns') 

    # get prediction variable. Max freq. Ignored Nan's
    #32, 35, 37, 38, 42
    pdFreq = pd.DataFrame(np.nanmax(np.array([data['q32'].values, data['q35'].values , data['q37'].values, 
                                         data['q38'].values, data['q42'].values]) , axis = 0), columns=['Freq Drug Use'])

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

    return NanColumns, pdFreq, pdQuestions

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
    
    # get prediction variable. Max freq. Ignored Nan's
    #32, 35, 37, 38, 42
    pdFreq = pd.DataFrame(np.nanmax(np.array([data['q32'].values, data['q35'].values , data['q37'].values, 
                                         data['q38'].values, data['q42'].values]) , axis = 0), columns=['Freq Drug Use'])

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
    return pdFreq, pdQuestions

NanColumns, pdIlicitDrugEverUsed, pdQuestions = preprocessingDrugFreq(data, lDemographics)


# drop all qn* questions. Drop compound questions
#pdQuestions = pdQuestions.drop([question for question in pdQuestions.columns if 'n' in question], axis='columns')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# split into train test
Xtrain, Xtest, ytrain, ytest = train_test_split(pdQuestions,
                                                pdIlicitDrugEverUsed, 
                                                test_size=0.015173780800382761, # to get 444 in test set
                                                random_state=0)
# split train into train val
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain,
                                                ytrain, 
                                                test_size=0.29496477773536456, # to get 8500 in val set
                                                random_state=0)
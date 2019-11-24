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
print(data.columns)
#%% Experimentation for predicting drug frequency

# data preprocessing
def preprocessingIlicitDrug(data, lDemographics):
    """ 
    function cleans data. First it removes rows with more tha N nans,
    then it removes the drug questinos from the X variables. Then it creates the Y var,
    then it drops demographic variables, then creates dummy vars for catagorical data,
    then removes columns with N nans
    
    The Y output is illicit drug frequency
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

    return pdIlicitDrugEverUsed, pdQuestions

NanColumns, pdIlicitDrugEverUsed, pdQuestions = preprocessingIlicitDrug(data, lDemographics)
# fills in Nans with most frequent catagory
Mode = {i:pdQuestions[i].value_counts().index[0] for i in pdQuestions.columns}
pdQuestionsMode = pdQuestions.fillna(value = Mode)


# drop all qn* questions. Drop compound questions
#pdQuestions = pdQuestions.drop([question for question in pdQuestions.columns if 'n' in question], axis='columns')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# split into train test
Xtrain, Xtest, ytrain, ytest = train_test_split( pdQuestionsMode,
                                                pdIlicitDrugEverUsed, 
                                                test_size=0.015173780800382761, # to get 444 in test set
                                                random_state=0)
# split train into train val
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain,
                                                ytrain, 
                                                test_size=0.29496
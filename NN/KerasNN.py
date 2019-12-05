#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:54:20 2019

@author: vector
"""

import keras
import os
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import metrics
import sklearn
from sklearn.model_selection import KFold   
import errorMetrics as Error
from keras import optimizers
from keras.optimizers import Adam


#%% other accuracy models
def Precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    Implimented in a previous version of keras
    """
#    y_pred = K.cast(y_pred > 0.5, y_true.dtype)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def Recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    

    
    In binary it's the sensitivity TP/(TP+FN)
    """
#    y_pred = K.cast(y_pred > 0.5, y_true.dtype) # need this for siamese networks
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


#%% error metrics for classifications
def GenAUC(npPred, npTruth):
    """
    generates auc, false pos rate, true pos rate and thresholds of each given a prediction and
    truth numpt array. Should work in any dimentional data. 
    Assumes binary classification and that a positive result is a 1
    """
    fpr, tpr, thresholds = metrics.roc_curve(npTruth, npPred, pos_label =1)
    return metrics.auc(fpr, tpr), fpr, tpr, thresholds
#%% build model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def fcNet(iInputSize = 100):
    """ 
    """
    model = Sequential()
    model.add(Dense(128, input_dim=iInputSize, activation='relu'))
    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.5))    
#    model.add(Dense(256, activation='relu'))
#    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(128, activation='relu')) 
#    model.add(Dense(128, activation='relu')) 
    
#    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model



#%%
        
import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats
def SplitData(sdf1, sdf2, sColumn1, sColumn2):

    #import data - change filename in future
    df1 = pd.read_csv(sdf1)
    df2 = pd.read_csv(sdf2)
    
    print("shape of first df", df1.shape)
    print("shape of second df", df2.shape) 
    if 'modOneOutcome' in df1.columns:
        nanRows = np.arange(0,len(df1))[np.isnan(df1['modOneOutcome'].values)]
        df1.drop(nanRows, axis='rows', inplace=True)
        #normalize weight
        Norm = ['weight', 'bmi']
        df1.loc[:, Norm] = (df1.loc[:, Norm] - df1.loc[:, Norm].min())
        df1.loc[:, Norm] = df1.loc[:, Norm]/df1.loc[:, Norm].max()

        
        nanRows = np.arange(0,len(df2))[np.isnan(df2['modTwoOutcome'].values)]
        df2.drop(nanRows, axis='rows', inplace=True)
        #normalize weight
        df2.loc[:, Norm] = (df2.loc[:, Norm] - df2.loc[:, Norm].min())
        df2.loc[:, Norm] = df2.loc[:, Norm]/df2.loc[:, Norm].max()
    
    #note this script assumes names: 'modOneOutcome', 'modTwoOutcome'
    # one is illicit drugs; two is tobacco and alcohol 
    
    ###################################
    # GET SET SIZES
    ###################################
    #test set
    #bounded loss apporach
    # d = 0.015
    testSize = int(((2 * 0.5) / 0.015)**2) #4444
    
    #validation set
    #to check the probability of correct model selection, with different validation test sizes
    # loc is mean = -0.01 for 1% model difference 
    # scale is std deviation = depends on n; using 0.5 for bound on loss standard deviation in formula
    # x is the value to go up to = 0 for getting the negative quadrant
    #tried the below probability calculation for several SD values; n = 8500 gives about 90% confidence of selecting better of two models 
    
    #  stats.norm.cdf(x = 0, loc = -0.01, scale = 0.0076696)
    
    validationSize = 8500
    
    
    
    ###################################
    #data splitting
    ###################################
    
    
    #multiply weight column by 100
    #range is currently from 0.044200 to 10.678100
    #need > 0 values for the duplication
    #this is going to make a lot of duplicate observations
    #there is some error introduced because only getting so many  decimals
    #we could recover all decimals; the data become really large though (computing shape of data took noticeable time) 
    
    df1['weightMultiplied'] = (df1['weight'] * 100).astype(int)
    df2['weightMultiplied'] = (df2['weight'] * 100).astype(int)
    
    #check 
    print("multiplied weights 1", df1['weightMultiplied'].describe())
    print("multiplied weights 2", df2['weightMultiplied'].describe())
    print("Shape of data 1", df1.shape)
    print("Shape of data 2", df2.shape)
    # print("sum of all weights  adn other stuff", df['weight'].describe())
    
    
    # SEPARATE OUTCOMES FROM PREDICTORS
    
    #extract outcomes - names may differ
    y1 = df1[['weightMultiplied', sColumn1]]
    y2 = df2[['weightMultiplied', sColumn2]]
    
    #extract predictors
    X1 = df1.drop([sColumn1], axis = 'columns')
    X2 = df2.drop([sColumn2], axis = 'columns')
    
    #get training set for illicit drug use outcome
    Xtrain1, XRemaining1, ytrain1, yRemaining1 = train_test_split(X1, y1, test_size = (testSize + validationSize), random_state = 0)
    #split remaning into test and validation 
    Xvalidate1, Xtest1, yvalidate1, ytest1 = train_test_split(XRemaining1, yRemaining1, test_size = testSize, random_state = 0)
    
    #remove weight vars from all but training set
    Xvalidate1 = Xvalidate1.drop(['weight', 'weightMultiplied'], axis = 'columns')
    yvalidate1 = yvalidate1.drop(['weightMultiplied'], axis = 'columns')
    Xtest1 = Xtest1.drop(['weight', 'weightMultiplied'], axis = 'columns')
    ytest1 = ytest1.drop(['weightMultiplied'], axis = 'columns')
    
    #check sizes 
    print("SIZES for 1\n\n")
    print("val X", Xvalidate1.shape)
    print("val y", yvalidate1.shape)
    print("test X", Xtest1.shape)
    print("test y", ytest1.shape)
    
    
    
    #### second outcome
    #get training set for alc and tobacco use outcome
    Xtrain2, XRemaining2, ytrain2, yRemaining2 = train_test_split(X2, y2, test_size = (testSize + validationSize), random_state = 0)
    #split remaning into test and validation 
    Xvalidate2, Xtest2, yvalidate2, ytest2 = train_test_split(XRemaining2, yRemaining2, test_size = testSize, random_state = 0)
    
    #remove weight vars from all but training set
    Xvalidate2 = Xvalidate2.drop(['weight', 'weightMultiplied'], axis = 'columns')
    yvalidate2 = yvalidate2.drop(['weightMultiplied'], axis = 'columns')
    Xtest2 = Xtest2.drop(['weight', 'weightMultiplied'], axis = 'columns')
    ytest2 = ytest2.drop(['weightMultiplied'], axis = 'columns')
    
    #check sizes 
    print("SIZES for 2\n\n")
    
    print("val X", Xvalidate2.shape)
    print("val y", yvalidate2.shape)
    print("test X", Xtest2.shape)
    print("test y", ytest2.shape)
    
    
    ###################################
    #Duplicate training by weights
    ###################################
    
    #to check duplication for 1
    print("mod 1 \n sum of weights in  X training set\n", Xtrain1['weightMultiplied'].sum())
    print("describe X train\n", Xtrain1['weightMultiplied'].describe())
    print("sum of weights in y training set\n", ytrain1['weightMultiplied'].sum())
    print("describe t train\n", ytrain1['weightMultiplied'].describe())
    
    #to check duplication for 2
    print("mod 2 \n sum of weights in  X training set\n", Xtrain2['weightMultiplied'].sum())
    print("describe X train\n", Xtrain2['weightMultiplied'].describe())
    print("sum of weights in y training set\n", ytrain2['weightMultiplied'].sum())
    print("describe t train\n", ytrain2['weightMultiplied'].describe())
    
    
    #multiply training set by weights
    
    XtrainDuplicate1 = Xtrain1.loc[Xtrain1.index.repeat(Xtrain1.weightMultiplied)]
    ytrainDuplicate1 = ytrain1.loc[ytrain1.index.repeat(ytrain1.weightMultiplied)]
    
    XtrainDuplicate2 = Xtrain2.loc[Xtrain2.index.repeat(Xtrain2.weightMultiplied)]
    ytrainDuplicate2 = ytrain2.loc[ytrain2.index.repeat(ytrain2.weightMultiplied)]
    
    
    
    #drop the weight variables from training data
    Xtrain1 = Xtrain1.drop(['weight', 'weightMultiplied'], axis = 'columns')
    ytrain1 = ytrain1.drop(['weightMultiplied'], axis = 'columns')
    XtrainDuplicate1 = XtrainDuplicate1.drop(['weight', 'weightMultiplied'], axis = 'columns')
    ytrainDuplicate1 = ytrainDuplicate1.drop(['weightMultiplied'], axis = 'columns')
    
    
    Xtrain2 = Xtrain2.drop(['weight', 'weightMultiplied'], axis = 'columns')
    ytrain2 = ytrain2.drop(['weightMultiplied'], axis = 'columns')
    XtrainDuplicate2 = XtrainDuplicate2.drop(['weight', 'weightMultiplied'], axis = 'columns')
    ytrainDuplicate2 = ytrainDuplicate2.drop(['weightMultiplied'], axis = 'columns')
    
    
    
    print("sizes for training data outcome 1\n\n")
    print("train X1:", Xtrain1.shape)
    print("train y1:", ytrain1.shape) 
    print("shape x 1 duplidated", XtrainDuplicate1.shape)
    print("shape y 1 duplidated", ytrainDuplicate1.shape)
    
    
    
    print("sizes for training data outcome 2\n\n")
    
    print("train X2:", Xtrain2.shape)
    print("train y2:", ytrain2.shape) 
    print("shape x 2 duplidated", XtrainDuplicate2.shape)
    print("shape y 2 duplidated", ytrainDuplicate2.shape)

#    #export the dfs for illicit drug use
#    Xtrain1.to_csv(r'   /Xtrain1', index = False)
#    ytrain1.to_csv(r'  /ytrain1', index = False)
#    XtrainDuplicate1.to_csv(r'  /XtrainDuplicate1', index = False)
#    ytrainDuplicate1.to_csv(r' /ytrainDuplicate1', index = False)
#    Xvalidate1.to_csv(r' /Xvalidate1', index = False)
#    yvalidate1.to_csv(r' /yvalidate1', index = False)
#    Xtest1.to_csv(r' /Xtest1', index = False)
#    ytest1.to_csv(r' /ytest1', index = False) 
#    
#    #for alcohol and tobacco use
#    Xtrain2.to_csv(r' /Xtrain2', index = False)
#    ytrain2.to_csv(r' /ytrain2', index = False)
#    XtrainDuplicate2.to_csv(r' /XtrainDuplicate2', index = False)
#    ytrainDuplicate2.to_csv(r' /ytrainDuplicate2', index = False)
#    Xvalidate2.to_csv(r' /Xvalidate2', index = False)
#    yvalidate2.to_csv(r' /yvalidate2', index = False)
#    Xtest2.to_csv(r' /Xtest2', index = False)
#    ytest2.to_csv(r' /ytest2', index = False) 
    return Xtrain1, ytrain1, XtrainDuplicate1, ytrainDuplicate1, Xvalidate1, yvalidate1, Xtest1, ytest1, Xtrain2, ytrain2, XtrainDuplicate2, ytrainDuplicate2, Xvalidate2, yvalidate2, Xtest2, ytest2



#%% load data
def NFoldCrossVal(X, y, sOutDir, numFolds = 5):
    # split into 5 fold
    cv  = sklearn.model_selection.KFold(n_splits = numFolds, random_state=0, shuffle= True)
    
    # store results
    pdConf = pd.DataFrame(columns = ['Fold', 'FNR', 'FPR', 'auc', 'optimal threshold'])
    for fold, index in enumerate(cv.split(X)):
        train_index = index[0]
        test_index = index[1]
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
    
        # split into train test
        Xtrain, Xtest, ytrain, ytest = X.values[train_index], X.values[test_index], y.values[train_index], y.values[test_index]
        
        # split into train, val
        Xtrain, Xval, ytrain, yval = sklearn.model_selection.train_test_split(Xtrain, ytrain, random_state = 0, test_size=0.33) 
        
        # train data
        model = fcNet(iInputSize = np.shape(Xtrain)[1])
        
        # compile model
        opt = Adam(lr=1e-3)
        model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', Recall, Precision])
    
        Xtrain = np.array(Xtrain)
        ytrain = np.reshape(np.array(ytrain, dtype=np.int8), (-1))
        history = model.fit(Xtrain, ytrain, epochs=150, batch_size=250,
                  class_weight = {0:1, 1:10}, validation_data = (Xval, yval))
        
        # get prediction son test set
        ypred = model.predict(Xtest)
        TN, FP, FN, TP  = Error.ConfMatrix(ytest, ypred > 0.5)
        auc, fpr, tpr, thresholds = Error.GenAUC(ypred, ytest)
        FNR, FPR = Error.GenFNRFPR(ypred>0.5, ytest)
        F1, recall, precision, ErrorRate  = Error.GenErrorRates(ytest, ypred > 0.5)
        
        pdConf = pdConf.append({'Fold':fold, 'FNR': FNR, 'FPR':FPR, 'auc':auc, 'f1':F1,
                                                'recall': recall, 'precision':precision,
                                                'Error Rate':ErrorRate}, ignore_index=True)
        Error.SaveTrainingHistory(os.path.join(sOutDir, "{}foldhistory".format(fold)), model, history)
    pdConf.to_csv(os.path.join(sOutDir, 'ErrorMetrics.csv'), index=False)   
    means = pdConf.mean()
    means.to_csv(os.path.join(sOutDir, 'means'))
    return means

def TrainTest(Xtrain,ytrain, Xval, yval, Xtest, ytest, sModelID):
    """ trains a model and outputs the testing values along with training history
    """
    model = fcNet(iInputSize = np.shape(Xtrain)[1])

    opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy', Recall, Precision])
    history = model.fit(Xtrain, ytrain, epochs=50, batch_size=500,
              class_weight = {0:1, 1:10}, validation_data = (Xval, yval))
    
    # get prediction son test set
    ypred = model.predict(Xtest)
    TN, FP, FN, TP  = Error.ConfMatrix(ytest, ypred > 0.5)
    auc, fpr, tpr, thresholds = Error.GenAUC(ypred, ytest)
    FNR, FPR = Error.GenFNRFPR(ypred>0.5, ytest)
    F1, recall, precision, ErrorRate  = Error.GenErrorRates(ytest, ypred > 0.5)
    pdConf = pd.DataFrame({'Model':sModelID, 'FNR': FNR, 'FPR':FPR, 'auc':auc, 'f1':F1,
                                            'recall': recall, 'precision':precision, 'Error Rate':ErrorRate,
                                            'sensitivity': TP/(TP+FN), 'specificity':TN/(TN+FP), 'accuracy':(TN+TP)/(TN+TP+FN+FP)}, index=[0])
    
    # save outputs
    dfOutputs = pd.DataFrame({'Pred':ypred.flatten(), 'Truth':ytest.flatten()})
    return pdConf, history, dfOutputs

def OneSplit(sOutDir, sModel1Data, sModel2Data, sColumns1, sColumns2):
    """ Splits and trains the data using the spltting method found in splitDuplicate
    """
    Xtrain1, ytrain1, XtrainDuplicate1, ytrainDuplicate1, Xvalidate1, yvalidate1, Xtest1, ytest1, Xtrain2, ytrain2, XtrainDuplicate2, ytrainDuplicate2, Xvalidate2, yvalidate2, Xtest2, ytest2 = SplitData(sModel1Data, sModel2Data,  sColumns1, sColumns2)
          
    pdConf1, history1, dfOutputs1 = TrainTest(XtrainDuplicate1.values,ytrainDuplicate1.values, Xvalidate1.values,
                                 yvalidate1.values, Xtest1.values, ytest1.values, sModel1Data)
                                  
    pdConf2, history2, dfOutputs2 = TrainTest(XtrainDuplicate2.values,ytrainDuplicate2.values, Xvalidate2.values,
                                 yvalidate2.values, Xtest2.values, ytest2.values, sModel2Data)
#    
#    pdConf1, history1, dfOutputs1 = TrainTest(Xtrain1.values, ytrain1.values, Xvalidate1.values,
#                                 yvalidate1.values, Xtest1.values, ytest1.values, sModel1Data)
#                                  
#    pdConf2, history2, dfOutputs2 = TrainTest(Xtrain2.values, ytrain2.values, Xvalidate2.values,
#                                 yvalidate2.values, Xtest2.values, ytest2.values, sModel2Data)
    
    
    os.makedirs(os.path.join(sOutDir, os.path.split(sModel1Data)[0].split('/')[-1]+'1'), exist_ok=True)
    os.makedirs(os.path.join(sOutDir, os.path.split(sModel2Data)[0].split('/')[-1]+'2'), exist_ok=True)
    Error.SaveTrainingHistory(os.path.join(sOutDir, os.path.split(sModel1Data)[0].split('/')[-1]+'1'), history1)
    Error.SaveTrainingHistory(os.path.join(sOutDir, os.path.split(sModel2Data)[0].split('/')[-1]+'2'), history2)
    pdConf1.to_csv(os.path.join(sOutDir, os.path.split(sModel1Data)[0].split('/')[-1]+'1', 'ErrorMetrics.csv'))    
    pdConf2.to_csv(os.path.join(sOutDir, os.path.split(sModel2Data)[0].split('/')[-1]+'2', 'ErrorMetrics.csv'))   
    # save AUC plot
    Error.PlotAUC(dfOutputs1['Pred'].values, dfOutputs1['Truth'].values, os.path.join(sOutDir, os.path.split(sModel1Data)[0].split('/')[-1]+'1', 'AUC.png'))
    Error.PlotAUC(dfOutputs2['Pred'].values, dfOutputs2['Truth'].values, os.path.join(sOutDir, os.path.split(sModel2Data)[0].split('/')[-1]+'2', 'AUC.png'))
    
    # save prediction data
    dfOutputs1.to_csv(os.path.join(sOutDir, os.path.split(sModel1Data)[0].split('/')[-1]+'1', 'Predictions.csv'), index=False)    
    dfOutputs2.to_csv(os.path.join(sOutDir, os.path.split(sModel2Data)[0].split('/')[-1]+'2', 'Predictions.csv'), index=False)
    return pdConf1, pdConf2



def ModelTrain(dfModelData, sXColumn, sOutDir):
    """ reads in a pandas dataframe. Removes one column. This column will be the Y data. Then trains a
    NN using this data. Outputs error metrics to sOUtDir
    """
    X = dfModelData.drop([sXColumn], axis='columns')
    y = pd.DataFrame(dfModelData[sXColumn])
    
    pdConf = NFoldCrossVal(X, y, sOutDir)
    return pdConf

#%%
#
#dfModel1 = pd.read_csv("../data/SelectedFeatures/Model1_135Feat_KBest_chi2.csv")
#ModelTrain(dfModel1, 'Ever used ilicit drugs', 'Model1/5FoldModel1NN_RFECV/')
#
#
#
##%% model 2, RFECV recursive feature elimination cross val
#dfModel2 = pd.read_csv("../data/SelectedFeatures/Model2WOStratum_RFECV118.csv")
#pdConfModel2 = ModelTrain(dfModel2, 'Freq Drug Use', 'Model2/5FoldModel1NN_RFECV/')

#%% training for 5 cross val
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1/RFECV.csv"), 
#           'Ever used ilicit drugs', 'Model1/Model1RFECV/')
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1NoDrugQs/RFECV.csv"),
#           'Ever used ilicit drugs', 'Model1/Model1NoDrugQsRFECV/')
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1NoDummies/RFECV.csv"),
#           'Ever used ilicit drugs', 'Model1/Model1NoDummiesRFECV/')
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1NoDrugQsNoDummies/RFECV.csv"),
#           'Ever used ilicit drugs', 'Model1/Model1NoDrugQsNoDummiesRFECV/')
#
#
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2/RFECV.csv"),
#           'Freq Drug Use', 'Model2/Model2RFECV/')
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2NoDummies/RFECV.csv"),
#           'Freq Drug Use', 'Model2/Model2NoDummies//')
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2NoDrugQsNoDummies/RFECV.csv"),
#           'Freq Drug Use', 'Model2/Model2NoDrugQsNoDummies/')
#ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2NoDrugQsNoDummies/RFECV.csv"),
#           'Freq Drug Use', 'Model2/5Model2NoDrugQsNoDummies/')

#%%
lModels1 = ["../data/SelectedFeatures/Model1/RFECV.csv",
           "../data/SelectedFeatures/Model1NoDrugQs/RFECV.csv",
           "../data/SelectedFeatures/Model1NoDummies/RFECV.csv",
           "../data/SelectedFeatures/Model1NoDrugQsNoDummies/RFECV.csv",
           '../data/dfManual']
lModels2= ["../data/SelectedFeatures/Model2/RFECV.csv",
           "../data/SelectedFeatures/Model2NoDrugQs/RFECV.csv",
           "../data/SelectedFeatures/Model2NoDummies/RFECV.csv",
           "../data/SelectedFeatures/Model2NoDrugQsNoDummies/RFECV.csv",
           '../data/dfManual']   
lyColumns1 = ['Ever used ilicit drugs',
                 'Ever used ilicit drugs',
                 'Ever used ilicit drugs',
                 'Ever used ilicit drugs',
                 'modOneOutcome' ]
lyColumns2 =['Freq Drug Use',
                 'Freq Drug Use',
                 'Freq Drug Use',
                 'Freq Drug Use',
                 'modTwoOutcome']
pdConf = pd.DataFrame(columns = ['Model', 'FNR', 'FPR','f1',
                                 'recall','precision', 'Error Rate', 'auc', 'sensitivity', 'specificity', 'accuracy'])
for i in range(0,5):
    pdConf1, pdConf2 = OneSplit('Results2_Duplicates', lModels1[i],lModels2[i],lyColumns1[i],lyColumns2[i])
#    Xtrain1, ytrain1, XtrainDuplicate1, ytrainDuplicate1, Xvalidate1, yvalidate1, Xtest1, ytest1, Xtrain2, ytrain2, XtrainDuplicate2, ytrainDuplicate2, Xvalidate2, yvalidate2, Xtest2, ytest2 = SplitData(lModels1[i],lModels2[i],lyColumns1[i],lyColumns2[i])

    pdConf = pdConf.append(pdConf1, ignore_index=True)
    pdConf = pdConf.append(pdConf2, ignore_index=True)
    
pdConf.to_csv('allErrors2Duplicates.csv')
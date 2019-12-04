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
    model.add(Dropout(0.5))
#    model.add(Dense(128, activation='relu')) 
    model.add(Dense(128, activation='relu')) 
    
#    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


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
        from keras.optimizers import Adam
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

def ModelTrain(dfModelData, sXColumn, sOutDir):
    """ reads in a pandas dataframe. Removes one column. This column will be the Y data. Then trains a
    NN using this data. OUtputs error metrics to sOUtDir
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

#%%
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1/RFECV.csv"), 
           'Ever used ilicit drugs', 'Model1/Model1RFECV/')
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1NoDrugQs/RFECV.csv"),
           'Ever used ilicit drugs', 'Model1/Model1NoDrugQsRFECV/')
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1NoDummies/RFECV.csv"),
           'Ever used ilicit drugs', 'Model1/Model1NoDummiesRFECV/')
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model1NoDrugQsNoDummies/RFECV.csv"),
           'Ever used ilicit drugs', 'Model1/Model1NoDrugQsNoDummiesRFECV/')


ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2/RFECV.csv"),
           'Freq Drug Use', 'Model2/Model2RFECV/')
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2NoDummies/RFECV.csv"),
           'Freq Drug Use', 'Model2/Model2NoDummies//')
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2NoDrugQsNoDummies/RFECV.csv"),
           'Freq Drug Use', 'Model2/Model2NoDrugQsNoDummies/')
ModelTrain(pd.read_csv("../data/SelectedFeatures/Model2NoDrugQsNoDummies/RFECV.csv"),
           'Freq Drug Use', 'Model2/5Model2NoDrugQsNoDummies/')



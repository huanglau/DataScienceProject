#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:59:39 2019

@author: vector
"""

"""
Created on Fri Jun 21 12:39:02 2019
Error metric functions
@author: lhuang
"""

import gc
import os
import sys
#import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import glob
#import numba 
import numpy as np 
#import skimage.io as io
#from . import IO as IO
#import src.ImageObject as image
#import src.DataGeneratorFunctions as Gen
#%% error metric functions

#@numba.jit
def Thresh(npImage, fThreshVal = 0.5):
    """ Thresholds a 2D multiple channel image
    """
    npThreshImage = np.zeros(np.shape(npImage))
    npThreshImage[npImage >= fThreshVal] = 1
    npThreshImage[npImage < fThreshVal] = 0
    return npThreshImage

#@numba.jit # jit compilse and increases the speed
def ConfMatrix(npTruthImg, npPredImg):
    """ uses parallel computing to generate confusion matrix 
    Found to be faster than sklearn's confusion_matrix for large images at least
    
    Can be a list of images.
    
    Returns  [tn, fp, fn, tp]
    """
    npPredImg = npPredImg.astype(np.bool)
    npTruthImg  = npTruthImg.astype(np.bool)
    npTP = np.logical_and(npPredImg, npTruthImg)
    npTN = np.logical_and(np.invert(npPredImg), np.invert(npTruthImg))
    npFN = np.logical_and(np.invert(npPredImg), npTruthImg)     
    npFP = np.logical_and(npPredImg, np.invert(npTruthImg))    
    #TODO: check if npTP is a numpy array
    return [np.sum(npTN), np.sum(npFP), np.sum(npFN), np.sum(npTP)]#confusion_matrix(npTruthImg.flatten(), npPredImg.flatten(), labels = lLabels).ravel()  


#%% error metrics for classifications
def GenAUC(npPred, npTruth):
    """
    generates auc, false pos rate, true pos rate and thresholds of each given a prediction and
    truth numpt array. Should work in any dimentional data. 
    Assumes binary classification and that a positive result is a 1
    """
    fpr, tpr, thresholds = metrics.roc_curve(npTruth, npPred, pos_label =1)
    return metrics.auc(fpr, tpr), fpr, tpr, thresholds

def OptimalThreshAUC(fpr, tpr, thresholds):
    """ returns the optimal threshold value in a binary classification
    when using an ROC caluclator
    
    Chose the optimal threshold by finding the point on the ROC curve that has the minimal 
    distance from (0,1). This was calculated by simple geometry. Use pythagras therom to find the 
    shortest distance from 1. The x-axis is fpr. Let the fpr for a given threshold be b.
    The y-axis is tpr. Let the tpr for a given threshold be a. The distance from a threshold
    to the point (0,1) is np.sqrt(fpr**2+(1-tpr)**2)
    """
    distance = np.sqrt((1-tpr)**2+fpr**2)
    index_min = np.argmin(distance)
    return thresholds[index_min]
    
def GenErrorRates(npPred, npTruth):
    """ gets a set of data, finds the F1,recall, precision, and error rate 
    Assumes prediction is already thresholded.
    
    Only works for boolean classifications
    
    REturns 
    F1,recall, precision, and error rate 
    """
    if np.shape(npPred) != np.shape(npTruth):
        raise ValueError('prediction and truth labels must be the same size')
    if np.sum((npTruth !=0) * (npTruth != 1.0)) > 0 or np.sum((npPred !=0) * (npPred != 1.0)) > 0 :
        raise ValueError('inputs must be 0s or 1s')
    npTN, npFP, npFN, npTP = ConfMatrix(npTruth, npPred)
    recall = npTP/(npTP+npFN)
    precision = npTP/(npTP+npFP)
    ErrorRate = (npFP+npFN)/(npTN+npFP+npFN+npTP)
    F1 = 2*precision*recall/(precision+recall)
    
    return F1, recall, precision, ErrorRate 

def GenFNRFPR(npPred, npTruth):
    """ gets a set of data, finds the false negative rate and false positive rate
    
    """
    if np.shape(npPred) != np.shape(npTruth):
        raise ValueError('prediction and truth labels must be the same size')
    if np.sum((npTruth !=0) * (npTruth != 1.0)) > 0 or np.sum((npPred !=0) * (npPred != 1.0)) > 0 :
        raise ValueError('inputs must be 0s or 1s')
    npTN, npFP, npFN, npTP = ConfMatrix(npTruth, npPred)
    FNR = npFN/(npFN+npTP)
    FPR = npFP/(npFP+npTN)
    return FNR, FPR
 
def CalcErrorRates(npResults, npTruthValues, pdConf, sPatID, sSlideID, lClasses):
    """ Calculates AUC, fpr, tpr, thresholds optimal thresholds, for npresults and npvalues
    returns an pdConf that has all the values added
    """
    auc, fpr, tpr, thresholds = GenAUC(npResults, npTruthValues)
    optThresh = OptimalThreshAUC(fpr, tpr, thresholds)
    F1, recall, precision, ErrorRate = GenErrorRates(Thresh(npResults[:,0]), npTruthValues)
    pdConf = pdConf.append({'PatID':sPatID, 'Slide ID':sSlideID,
                                'fpr': fpr, 'tpr':tpr, 'auc':auc, 'f1':F1,
                                'recall': recall, 'precision':precision,
                                'Error Rate':ErrorRate,
                                'optimal threshold':optThresh}, ignore_index=True)  

    
def SaveTrainingHistory(sOutDir, Model, history):
    """
    Saves training information to sOutDir. 
    Also does some error metrics
    
    Saves ROC, accuracy and loss plots. Save the training history( val, val_acc, acc and loss)
    to a pickle file. Saves lnpPredictions and truth labels to a .txt file.
    
    """
#    Model.save(os.path.join(sOutDir,'trainedModel.h5')) # file sizes too large to do every cross validation
    os.makedirs(sOutDir, exist_ok = True)
    FigAcc, FigLoss = TrainingHistoryPlots(history)
    FigAcc.savefig(os.path.join(sOutDir,'model_accuracy.png'), dpi = 300)
    FigLoss.savefig(os.path.join(sOutDir,'model_loss.png'), dpi = 300)
    plt.close(FigAcc)
    plt.close(FigLoss)

def TrainingHistoryPlots(history):
    """
    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    Create loss, validation loss plots and accuracy and validatio accuracy plots.
    """
    Fig1 = plt.figure()
    
    # make plots for things that aren't loss
    allKeys = [str(x) for x in history.history.keys() if 'loss' not in str(x)]
    for key in allKeys:
        plt.plot(np.arange(1,len(history.history[key])+1), history.history[key], figure = Fig1)
#        plt.plot(history.history[key], figure = Fig1)
    plt.title('model accuracy', figure=Fig1)
    plt.ylabel('accuracy', figure=Fig1)
    plt.xlabel('epoch', figure=Fig1)
    plt.legend(allKeys, loc='upper left')
    # summarize history for loss
    Fig2 = plt.figure()
    plt.plot(np.arange(1,len(history.history[key])+1), history.history['loss'], figure=Fig2)
    plt.plot(np.arange(1,len(history.history[key])+1), history.history['val_loss'], figure=Fig2)
    plt.title('model loss', figure=Fig2)
    plt.ylabel('loss', figure=Fig2)
    plt.xlabel('epoch', figure=Fig2)
    plt.legend(['train', 'val'], loc='upper left')
    return Fig1, Fig2

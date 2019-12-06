#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:41:29 2019

@author: vector
"""
import os
import sys
if os.getcwd not in sys.path:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.split(os.getcwd())[0])

import errorMetrics as Error
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


"""Logistic together, trees, neural network grouped together. 
Aucs 
"""


def GenAUCPlots(file, label):
    data = pd.read_csv(file)
    fpr, tpr, thresholds = metrics.roc_curve(data.Truth.values, data.Pred.values, pos_label =1)
    plt.plot(fpr, tpr, label = label)
def GenAUCPlotsB(ypred, ytruth, label):

    fpr, tpr, thresholds = metrics.roc_curve(ytruth, ypred, pos_label =1)
    plt.plot(fpr, tpr, label = label)

#GenAUCPlots(file, label)
#GenAUCPlots(file, label)
#GenAUCPlots(file, label)
#GenAUCPlots(file, label)
    
#%% plot trees model 1
ypred = np.load('treesPredsTruth/Preds_Forest_M1_DD_NoDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M1_DD_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Not Weighted')
ypred = np.load('treesPredsTruth/Preds_Forest_M1_DD_YesDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M1_DD_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Weighted')
ypred = np.load('treesPredsTruth/Preds_Forest_M1_MF_YesDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M1_MF_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Not Weighted')
ypred = np.load('treesPredsTruth/Preds_Forest_M1_MF_NoDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M1_MF_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_Dummy.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_Dummy.npy')
GenAUCPlotsB(ypred, ytruth, label='Dummy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Model1ForestAUCs.png')
plt.show()

#%% plot trees model 2
ypred = np.load('treesPredsTruth/Preds_Forest_M2_DD_NoDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M2_DD_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Not Weighted')
ypred = np.load('treesPredsTruth/Preds_Forest_M2_DD_YesDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M2_DD_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Weighted')
ypred = np.load('treesPredsTruth/Preds_Forest_M2_MF_YesDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M2_MF_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Not Weighted')
ypred = np.load('treesPredsTruth/Preds_Forest_M2_MF_NoDup.npy')
ytruth = np.load('treesPredsTruth/Truth_Forest_M2_MF_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_Dummy.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_Dummy.npy')
GenAUCPlotsB(ypred, ytruth, label='Dummy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Model2ForestAUCs.png')
plt.show()


#%% plot log reg model 1
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_DD_NoDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_DD_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Not Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_DD_YesDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_DD_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_MF_YesDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_MF_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Not Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_MF_NoDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_MF_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_Dummy.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_Dummy.npy')
GenAUCPlotsB(ypred, ytruth, label='Dummy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Model1LogRegAUCs.png')
plt.show()

#%% plot logreg model 2
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M2_DD_NoDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M2_DD_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Not Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M2_DD_YesDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M2_DD_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='RFE Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M2_MF_YesDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M2_MF_YesDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Not Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M2_MF_NoDup.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M2_MF_NoDup.npy')
GenAUCPlotsB(ypred, ytruth, label='Manual Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_Dummy.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_Dummy.npy')
GenAUCPlotsB(ypred, ytruth, label='Dummy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Model2LogRegAUCs.png')
plt.show()
#%% neural network model 1
GenAUCPlots('NN/Results2_NoDuplicates_Val/Model1NoDrugQs1/Predictions.csv', label='RFE Not Weighted')
GenAUCPlots('NN/Results2_Duplicates_Val/Model1NoDrugQs1/Predictions.csv', label='RFE Weighted')
GenAUCPlots('NN/Results2_Duplicates/data1/Predictions.csv', label='Manual Not Weighted')
GenAUCPlots('NN/Results2_Duplicates/data2/Predictions.csv', label='Manual Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_Dummy.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_Dummy.npy')
GenAUCPlotsB(ypred, ytruth, label='Dummy')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Model1NNAUCs.png')
plt.show()

#%% neural network model 2
GenAUCPlots('NN/Results2_NoDuplicates_Val/Model2NoDrugQs1/Predictions.csv', label='RFE Not Weighted')
GenAUCPlots('NN/Results2_Duplicates_Val/Model2NoDrugQs1/Predictions.csv', label='RFE Weighted')
GenAUCPlots('NN/Results2_Duplicates/data1/Predictions.csv', label='Manual Not Weighted')
GenAUCPlots('NN/Results2_Duplicates/data2/Predictions.csv', label='Manual Weighted')
ypred = np.load('logisticRegressionPredsAndTruth/Preds_LR_M1_Dummy.npy')
ytruth = np.load('logisticRegressionPredsAndTruth/Truth_LR_M1_Dummy.npy')
GenAUCPlotsB(ypred, ytruth, label='Dummy')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Model2NNAUCs.png')
plt.show()




#%% Testing AUCS Model1
ypred = np.load('testingPredTruth/Preds_LR_M1_TEST.npy')
ytruth = np.load('testingPredTruth/Truth_LR_M1_TEST.npy')
GenAUCPlotsB(ypred, ytruth, label='')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('Model1_LR_M1_Test_AUCs.png')
plt.show()

#%% Testing AUCS Model2
ypred = np.load('testingPredTruth/Preds_Forest_M2_TEST.npy')
ytruth = np.load('testingPredTruth/Truth_Forest_M2_TEST.npy')
GenAUCPlotsB(ypred, ytruth, label='')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('Model2_Forest_Test_AUCs.png')
plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:53:01 2019

@author: vector
"""

import numpy as np
from kmodes.kModesAnywhere.kmodes.kmodes import KModes
import pandas as pd
import os
import sys
import errorMetrics as Error

if os.getcwd not in sys.path:
    sys.path.append(os.getcwd())


"""
Extensions to thek-Means Algorithm for ClusteringLarge Data Sets with Categorical Values
https://link.springer.com/article/10.1023/A:1009769707641
K -means doesn't work for catagorical data. k-modes does

https://pypi.org/project/kmodes/
"""



def kmodesClusters(dfModel, yColumn, n_clusters = 2):
    """ runs kmode clustering on a pandas model. Uses the column yColumn as the truth values.
    Outputs the predictions.
    """
    
    X = dfModel.drop([yColumn], axis='columns') 
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(X)

    return clusters

#%%
def kModeClusters(sOutDir, n_clusters):
    """
    """
    pdConf = pd.DataFrame(columns = ['data', 'accuracy', 'f1', 'recall', 'precision' ])
#    dfModel =  pd.read_csv("data/SelectedFeatures/Model1/RFECV.csv")
    lModels = ["data/SelectedFeatures/Model1/RFECV.csv",
               "data/SelectedFeatures/Model1NoDrugQs/RFECV.csv",
               "data/SelectedFeatures/Model1NoDummies/RFECV.csv",
               "data/SelectedFeatures/Model1NoDrugQsNoDummies/RFECV.csv",
               "data/SelectedFeatures/Model2/RFECV.csv",
               "data/SelectedFeatures/Model2NoDrugQs/RFECV.csv",
               "data/SelectedFeatures/Model2NoDummies/RFECV.csv",
               "data/SelectedFeatures/Model2NoDrugQsNoDummies/RFECV.csv"]
    lyColumns = ['Ever used ilicit drugs',
                 'Ever used ilicit drugs',
                 'Ever used ilicit drugs',
                 'Ever used ilicit drugs',
                 'Freq Drug Use',
                 'Freq Drug Use',
                 'Freq Drug Use',
                 'Freq Drug Use']
    for index, model in enumerate(lModels):
        try:
            dfModel =  pd.read_csv(model)
        except FileNotFoundError:
            pdConf.to_csv(sOutDir)
            print("{} file not found!".format(model))
                
        clusters = kmodesClusters(dfModel,lyColumns[index], n_clusters)
        y = dfModel[lyColumns[index]].values

        # save predictions
        dfPredictions = pd.DataFrame({'predictions':clusters, 'truth':y.flatten()})
        dfPredictions.to_csv(os.path.join(os.path.split(model)[0], 'ClusteringPredictions.csv'), index=False)    

        # get error metrics
        TN, FP, FN, TP  = Error.ConfMatrix(y.flatten(), clusters > 0.5)
        F1, recall, precision, ErrorRate  = Error.GenErrorRates(y.flatten(), clusters > 0.5)
        pdConf = pdConf.append({'data':model, 'accuracy': (TN+TP)/(len(clusters)), 'f1':F1,
                                            'recall': recall, 'precision':precision}, ignore_index=True)
    pdConf.to_csv(sOutDir)

#kModeClusters('clusteringErrorMetrics.csv', 2)
kModeClusters('clusteringErrorMetrics1.csv', 1)
kModeClusters('clusteringErrorMetrics3.csv', 3)
kModeClusters('clusteringErrorMetrics4.csv', 4)
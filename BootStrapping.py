# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:24:04 2019

@author: lhuang
"""

import matplotlib.pyplot as plt 
import scipy.stats as ss
import pandas as pd 
import numpy as np
import seaborn as sns
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import t
import errorMetrics as Error

def threshold_loss(obs, pred, thresh):
    
    # Check if residual is within threshold
    resids = abs(obs-pred)>thresh
    
    # This turns booleans into ints.  1==True, 0==False
    # Most people used a loop.  That works too
    resids = resids.astype(int)
    return resids.mean(), resids.std(ddof=1)


#%%
    
def BootStrap():
    numits = 10000
    thresh = 0.5
    bootstrap_acc=np.zeros(numits)
    bootstrap_auc=np.zeros(numits)
    bootstrap_sen=np.zeros(numits)
    bootstrap_spe=np.zeros(numits)
    
    ypred = np.load('testingPredTruth/Preds_LR_M1_TEST.npy')
    ytruth = np.load('testingPredTruth/Truth_LR_M1_TEST.npy')
    n_abs_loss = len(ypred)*.5
    
    for i in range(numits):
        # Sample incicies from
        ix = np.random.randint(low=0, high = int(n_abs_loss), size = int(n_abs_loss))
        # Passing randomly sampled indicies is the same as randomly sampling
        ypredboot = ypred[ix]
        ytruthboot = ytruth[ix]
        TN, FP, FN, TP  = Error.ConfMatrix(ytruthboot, ypredboot > 0.5)
        auc, fpr, tpr, thresholds = Error.GenAUC(ypredboot, ytruthboot)
        FNR, FPR = Error.GenFNRFPR(ypredboot>0.5, ytruthboot)
        F1, recall, precision, ErrorRate  = Error.GenErrorRates(ytruthboot, ypredboot > 0.5)
        
        bootstrap_acc[i]= (TP+TN)/(TN + FP+ FN + TP)
        bootstrap_auc[i]= auc
        bootstrap_sen[i]= (TP)/(TN + FN)
        bootstrap_spe[i]= (TN)/(TN + FP)
    
    
    #Interval is quantiles
    #interval_estimate = np.quantile(bootstrap_auc*(np.isnan(bootstrap_auc)==False), [0.025, 0.85, 0.975])
    
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(bootstrap_acc, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(bootstrap_acc, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

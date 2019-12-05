
######################################################
#Imports
######################################################

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, make_scorer, confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt



######################################################
#Reading in and formatting data
######################################################


#starting with the first outcome
#load training and validation data. Non-duplicated. Manual Feature extraction.

Xtrain1 = pd.read_csv('./DataSplit/Xtrain1')
ytrain1 = pd.read_csv('./DataSplit/ytrain1')

Xvalidate1 = pd.read_csv('./DataSplit/Xvalidate1')
yvalidate1 = pd.read_csv('./DataSplit/yvalidate1')


#check that the data are correct 
print("X train shape", Xtrain1.shape)
print("y train shape", ytrain1.shape)
print("X validate shape", Xvalidate1.shape)
print("y validate shape", yvalidate1.shape)

#change the y's to numpy arrays of type integer

yvalidate1 = yvalidate1['modOneOutcome'].values
# print("type of valid", type(yvalidate1))
yvalidate1 = yvalidate1.astype('int64')
#print("these validation outcomes of type integer", yvalidate1.dtype)

ytrain1 = ytrain1['modOneOutcome'].values
ytrain1 = ytrain1.astype('int64')


######################################################
#Functions to get metrics
######################################################



# Computing metrics function from Lab 7
def compute_performance(yhat, y):
    # First, get tp, tn, fp, fn
    tp = sum(np.logical_and(yhat == 1, y == 1))
    tn = sum(np.logical_and(yhat == 0, y == 0))
    fp = sum(np.logical_and(yhat == 1, y == 0))
    fn = sum(np.logical_and(yhat == 0, y == 1))

    print(f"tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
    
    # Accuracy
    trainacc = (tp + tn) / (tp + tn + fp + fn) 
    
    # Precision
    # "Of the ones I labeled +, how many are actually +?"
    precision = tp / (tp + fp)
    
    # Recall
    # "Of all the + in the data, how many do I correctly label?"
    recall = tp / (tp + fn)    
    
    # Sensitivity
    # "Of all the + in the data, how many do I correctly label?"
    sensitivity = recall
    
    # Specificity
    # "Of all the - in the data, how many do I correctly label?"
    specificity = tn / (fp + tn)
    
    # Print results
   
    print("Acc:",trainacc,"Sens:",sensitivity,"Spec:",specificity)
    


######################################################
#Model Building
######################################################


#pipeline for LR model - parameters that give best performance
logisticPipeline = Pipeline([
    ('logistic regression', LogisticRegression(solver = 'lbfgs',
                                               penalty = 'l2',
                                               C = 10,
                                               max_iter = 10000))
    ])




###run for different combinations of things


#fit model - manual feature selection; no duplication; obj 1 outcome
lrM1 = logisticPipeline.fit(Xtrain1, ytrain1)

#make predictions on validation set
lrM1_validPred = lrM1.predict(Xvalidate1) #.reshape(-1,1)
lrM1_validPred = lrM1_validPred.astype('int64') #convert to type integer

#call above function to get performance metrics
print("\nPerformance metrics for logistic regression for Objective 1 outcome, non-duplicated data\n")
compute_performance(lrM1_validPred, yvalidate1)

#call sklearn functions to get ROC curve (currently commented out) 
yvalM1_pred_prob = lrM1.predict_proba(Xvalidate1)[:,1]
fpr_M1, tpr_M1, _ = roc_curve(yvalidate1, yvalM1_pred_prob)

#incomment roc curve plot
# sns.lineplot(fpr_M1,tpr_M1)
# plt.show()

#AUC metrics
auc_M1 = auc(fpr_M1, tpr_M1)
print("AUC\n", auc_M1)



######################################################
#Reading in and formatting data for objective 1 duplicated manual feature selection
######################################################


#starting with the first outcome
#load training and validation data. Non-duplicated. Manual Feature extraction.

XtrainDuplicate1 = pd.read_csv('./DataSplit/XtrainDuplicate1')
ytrainDuplicate1 = pd.read_csv('./DataSplit/ytrainDuplicate1')

ytrainDuplicate1 = ytrainDuplicate1['modOneOutcome'].values
ytrainDuplicate1 = ytrainDuplicate1.astype('int64')

#fit model - manual feature selection; no duplication; obj 1 outcome
lrMDup1 = logisticPipeline.fit(XtrainDuplicate1, ytrainDuplicate1)

#make predictions on validation set
lrMDup1_validPred = lrMDup1.predict(Xvalidate1) #.reshape(-1,1)
lrMDup1_validPred = lrMDup1_validPred.astype('int64') #convert to type integer

#call above function to get performance metrics
print("\nPerformance metrics for logistic regression for Objective 1 outcome, duplicated data\n")
compute_performance(lrMDup1_validPred, yvalidate1)

#call sklearn functions to get ROC curve (currently commented out) 
yvalMDup1_pred_prob = lrMDup1.predict_proba(Xvalidate1)[:,1]
fpr_MDup1, tpr_MDup1, _ = roc_curve(yvalidate1, yvalMDup1_pred_prob)

#incomment roc curve plot
# sns.lineplot(fpr_M1,tpr_M1)
# plt.show()

#AUC metrics
auc_MDup1 = auc(fpr_MDup1, tpr_MDup1)
print("AUC\n", auc_MDup1)





######################################################
#Reading in and formatting data for objective 2
######################################################


#starting with the first outcome
#load training and validation data. Non-duplicated. Manual Feature extraction.

Xtrain2 = pd.read_csv('./DataSplit/Xtrain2')
ytrain2 = pd.read_csv('./DataSplit/ytrain2')

Xvalidate2 = pd.read_csv('./DataSplit/Xvalidate2')
yvalidate2 = pd.read_csv('./DataSplit/yvalidate2')


#check that the data are correct 
print("X train shape", Xtrain2.shape)
print("y train shape", ytrain2.shape)
print("X validate shape", Xvalidate2.shape)
print("y validate shape", yvalidate2.shape)

#change the y's to numpy arrays of type integer

yvalidate2 = yvalidate2['modTwoOutcome'].values
# print("type of valid", type(yvalidate1))
yvalidate2 = yvalidate2.astype('int64')
#print("these validation outcomes of type integer", yvalidate1.dtype)

ytrain2 = ytrain2['modTwoOutcome'].values
ytrain2 = ytrain2.astype('int64')



#fit model - manual feature selection; no duplication; obj 1 outcome
lrM2 = logisticPipeline.fit(Xtrain2, ytrain2)

#make predictions on validation set
lrM2_validPred = lrM2.predict(Xvalidate2) #.reshape(-1,1)
lrM2_validPred = lrM2_validPred.astype('int64') #convert to type integer

#call above function to get performance metrics
print("\nPerformance metrics for logistic regression for Objective 2 outcome, non-duplicated data\n")
compute_performance(lrM2_validPred, yvalidate2)

#call sklearn functions to get ROC curve (currently commented out) 
yvalM2_pred_prob = lrM2.predict_proba(Xvalidate2)[:,1]
fpr_M2, tpr_M2, _ = roc_curve(yvalidate2, yvalM2_pred_prob)

#incomment roc curve plot
# sns.lineplot(fpr_M1,tpr_M1)
# plt.show()

#AUC metrics
auc_M2 = auc(fpr_M2, tpr_M2)
print("AUC\n", auc_M2)


######################################################
#Reading in and formatting data for objective 2 duplicated manual feature selection
######################################################


#starting with the first outcome
#load training and validation data. Non-duplicated. Manual Feature extraction.

XtrainDuplicate2 = pd.read_csv('./DataSplit/XtrainDuplicate2')
ytrainDuplicate2 = pd.read_csv('./DataSplit/ytrainDuplicate2')

ytrainDuplicate2 = ytrainDuplicate2['modTwoOutcome'].values
ytrainDuplicate2 = ytrainDuplicate2.astype('int64')


#fit model - manual feature selection; no duplication; obj 1 outcome
lrMDup2 = logisticPipeline.fit(XtrainDuplicate2, ytrainDuplicate2)

#make predictions on validation set
lrMDup2_validPred = lrMDup2.predict(Xvalidate2) #.reshape(-1,1)
lrMDup2_validPred = lrMDup2_validPred.astype('int64') #convert to type integer

#call above function to get performance metrics
print("\nPerformance metrics for logistic regression for Objective 2 outcome, duplicated data\n")
compute_performance(lrMDup2_validPred, yvalidate2)

#call sklearn functions to get ROC curve (currently commented out) 
yvalMDup2_pred_prob = lrMDup2.predict_proba(Xvalidate2)[:,1]
fpr_MDup2, tpr_MDup2, _ = roc_curve(yvalidate2, yvalMDup2_pred_prob)

#incomment roc curve plot
# sns.lineplot(fpr_M1,tpr_M1)
# plt.show()

#AUC metrics
auc_MDup2 = auc(fpr_MDup2, tpr_MDup2)
print("AUC\n", auc_MDup2)









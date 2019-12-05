

######################################################
#Imports
######################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import recall_score, make_scorer, confusion_matrix, roc_curve, auc



######################################################
#Read in Data 
######################################################


#first outcome. manual feature selection. non-duplicated. 
Xtrain1 = pd.read_csv('./DataSplit/Xtrain1')
ytrain1 = pd.read_csv('./DataSplit/ytrain1')
ytrain1 = ytrain1['modOneOutcome'].values
ytrain1 = ytrain1.astype('int64')

Xvalidate1 = pd.read_csv('./DataSplit/Xvalidate1')
yvalidate1 = pd.read_csv('./DataSplit/yvalidate1')
yvalidate1 = yvalidate1['modOneOutcome'].values
yvalidate1 = yvalidate1.astype('int64')

#second outcome. manual feature selection. non-duplicated. 
Xtrain2 = pd.read_csv('./DataSplit/Xtrain2')
ytrain2 = pd.read_csv('./DataSplit/ytrain2')
ytrain2 = ytrain2['modTwoOutcome'].values
ytrain2 = ytrain2.astype('int64')

Xvalidate2 = pd.read_csv('./DataSplit/Xvalidate2')
yvalidate2 = pd.read_csv('./DataSplit/yvalidate2')
yvalidate2 = yvalidate2['modTwoOutcome'].values
yvalidate2 = yvalidate2.astype('int64')


#first outcome. manual feature selection. duplicated.
XtrainDuplicate1 = pd.read_csv('./DataSplit/XtrainDuplicate1')
ytrainDuplicate1 = pd.read_csv('./DataSplit/ytrainDuplicate1')
ytrainDuplicate1 = ytrainDuplicate1['modOneOutcome'].values
ytrainDuplicate1 = ytrainDuplicate1.astype('int64')


#second outcome. manual feature selection. duplicated
XtrainDuplicate2 = pd.read_csv('./DataSplit/XtrainDuplicate2')
ytrainDuplicate2 = pd.read_csv('./DataSplit/ytrainDuplicate2')
ytrainDuplicate2 = ytrainDuplicate2['modTwoOutcome'].values
ytrainDuplicate2 = ytrainDuplicate2.astype('int64')


######################################################
#Function to get performance 
######################################################
# Computing metrics function from Lab 7
def compute_performance(yhat, y, modtype):

    print("\n Model Type:", modtype, "\n")
    
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
#Set Up Models
######################################################

#use these two for model 1 outcome
rf1 = RandomForestClassifier(n_estimators = 800, max_features = 5)
rf2 = RandomForestClassifier(n_estimators = 800, max_features = None)

#use these two for model 2 outcome
rf3 = RandomForestClassifier(n_estimators = 800, max_depth = 2)
rf4 = RandomForestClassifier(n_estimators = 800, max_depth = 10, max_features = 10)


######################################################
#Fit and Predict on All Data Types
######################################################

print("OUTCOME 1 NON DUPLICATED MANUAL FEATURE SELECTION\n")

'''
#fit models
mod1_M1 = rf1.fit(Xtrain1, ytrain1)
mod2_M1 = rf2.fit(Xtrain1, ytrain1)

#make predictions
ypred_rf1_M1 = rf1.predict(Xvalidate1)
ypred_rf2_M1 = rf2.predict(Xvalidate1)


#first model metrics

#call performance function
compute_performance(ypred_rf1_M1, yvalidate1, mod1_M1)
#call sklearn functions to get auc
mod1_pred_prob_M1 = mod1_M1.predict_proba(Xvalidate1)[:,1]
fpr_mod1_M1, tpr_mod1_M1, _ = roc_curve(yvalidate1, mod1_pred_prob_M1)
auc_mod1_M1 = auc(fpr_mod1_M1, tpr_mod1_M1)
print("AUC\n", auc_mod1_M1)

#second model metrics

compute_performance(ypred_rf2_M1, yvalidate1, mod2_M1)

mod2_pred_prob_M1 = mod2_M1.predict_proba(Xvalidate1)[:,1]
fpr_mod2_M1, tpr_mod2_M1, _ = roc_curve(yvalidate1, mod2_pred_prob_M1)
auc_mod2_M1 = auc(fpr_mod2_M1, tpr_mod2_M1)
print("AUC\n", auc_mod2_M1)

'''


print("\nOUTCOME 2 NON DUPLICATED MANUAL FEATURE SELECTION\n")

#fit models
# mod3_M2 = rf3.fit(Xtrain2, ytrain2)
mod4_M2 = rf4.fit(Xtrain2, ytrain2)

#make predictions
# ypred_rf3_M2 = rf3.predict(Xvalidate2)
ypred_rf4_M2 = rf4.predict(Xvalidate2)

'''
#call performance function
compute_performance(ypred_rf3_M2, yvalidate2, mod3_M2)
#call sklearn functions to get auc
mod3_pred_prob_M2 = mod3_M2.predict_proba(Xvalidate2)[:,1]
fpr_mod3_M2, tpr_mod3_M2, _ = roc_curve(yvalidate2, mod3_pred_prob_M2)
auc_mod3_M2 = auc(fpr_mod3_M2, tpr_mod3_M2)
print("AUC\n", auc_mod3_M2)
'''

#call performance function
compute_performance(ypred_rf4_M2, yvalidate2, mod4_M2)
#call sklearn functions to get auc
mod4_pred_prob_M2 = mod4_M2.predict_proba(Xvalidate2)[:,1]
fpr_mod4_M2, tpr_mod4_M2, _ = roc_curve(yvalidate2, mod4_pred_prob_M2)
auc_mod4_M2 = auc(fpr_mod4_M2, tpr_mod4_M2)
print("AUC\n", auc_mod4_M2)



print("\nOUTCOME 1 DUPLICATED MANUAL FEATURE SELECTION\n")

#fit models
mod1_MD1 = rf1.fit(XtrainDuplicate1, ytrainDuplicate1)
mod2_MD1 = rf2.fit(XtrainDuplicate1, ytrainDuplicate1)

#make predictions
ypred_rf1_MD1 = mod1_MD1.predict(Xvalidate1)
ypred_rf2_MD1 = mod2_MD1.predict(Xvalidate1)

#call performance function
compute_performance(ypred_rf1_MD1, yvalidate1, mod1_MD1)
#call sklearn functions to get auc
mod1_pred_prob_MD1 = mod1_MD1.predict_proba(Xvalidate1)[:,1]
fpr_mod1_MD1, tpr_mod1_MD1, _ = roc_curve(yvalidate1, mod1_pred_prob_MD1)
auc_mod1_MD1 = auc(fpr_mod1_MD1, tpr_mod1_MD1)
print("AUC\n", auc_mod1_MD1)


compute_performance(ypred_rf2_MD1, yvalidate1, mod2_MD1)
#call sklearn functions to get auc
mod2_pred_prob_MD1 = mod2_MD1.predict_proba(Xvalidate1)[:,1]
fpr_mod2_MD1, tpr_mod2_MD1, _ = roc_curve(yvalidate1, mod2_pred_prob_MD1)
auc_mod2_MD1 = auc(fpr_mod2_MD1, tpr_mod2_MD1)
print("AUC\n", auc_mod2_MD1)

print("\nOUTCOME 2 DUPLICATED MANUAL FEATURE SELECTION\n")

#fit models
mod3_MD2 = rf3.fit(XtrainDuplicate2, ytrainDuplicate2)
mod4_MD2 = rf4.fit(XtrainDuplicate2, ytrainDuplicate2)

#make predictions
ypred_rf3_MD2 = mod3_MD2.predict(Xvalidate2)
ypred_rf4_MD2 = mod4_MD2.predict(Xvalidate2)

#call performance function
compute_performance(ypred_rf3_MD2, yvalidate2, mod3_MD2)
#call sklearn functions to get auc
mod3_pred_prob_MD2 = mod3_MD2.predict_proba(Xvalidate2)[:,1]
fpr_mod3_MD2, tpr_mod3_MD2, _ = roc_curve(yvalidate2, mod3_pred_prob_MD2)
auc_mod3_MD2 = auc(fpr_mod3_MD2, tpr_mod3_MD2)
print("AUC\n", auc_mod3_MD2)


compute_performance(ypred_rf4_MD2, yvalidate2, mod4_MD2)
#call sklearn functions to get auc
mod4_pred_prob_MD2 = mod4_MD2.predict_proba(Xvalidate2)[:,1]
fpr_mod4_MD2, tpr_mod4_MD2, _ = roc_curve(yvalidate2, mod4_pred_prob_MD2)
auc_mod4_MD2 = auc(fpr_mod4_MD2, tpr_mod4_MD2)
print("AUC\n", auc_mod4_MD2)


#can plot a fitted tree
# tree.plot_tree(clf.fit(iris.data, iris.target))



import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats


#import data - change filename in future
df1 = pd.read_csv('dfManual1')
df2 = pd.read_csv('dfManual2')

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

df1['weightMultiplied'] = df1['weight'] * 100
df2['weightMultiplied'] = df2['weight'] * 100

#check 
# print("multiplied weights", df['weightMultiplied'].describe())
# print("Shape of data", df.shape)
# print("sum of all weights  adn other stuff", df['weight'].describe())


# SEPARATE OUTCOMES FROM PREDICTORS

#extract outcomes - names may differ
y1 = df1[['weightMultiplied', 'modOneOutcome']]
y2 = df2[['weightMultiplied', 'modTwoOutcome']]

#extract predictors
X1 = df1.drop(['modOneOutcome'], axis = 'columns')
X2 = df2.drop(['modTwoOutcome'], axis = 'columns')

#get training set for illicit drug use outcome
Xtrain1, XRemaining1, ytrain1, yRemaining1 = train_test_split(X1, y1, test_size = (testSize + validationSize), random_state = 0)
#split remaning into test and validation 
Xvalidate1, Xtest1, yvalidate1, ytest1 = train_test_split(XRemaining1, yRemaining1, test_size = testSize, random_state = 0)


#get training set for alc and tobacco use outcome
Xtrain2, XRemaining2, ytrain2, yRemaining2 = train_test_split(X2, y2, test_size = (testSize + validationSize), random_state = 0)
#split remaning into test and validation 
Xvalidate2, Xtest2, yvalidate2, ytest2 = train_test_split(XRemaining2, yRemaining2, test_size = testSize, random_state = 0)

#check sizes 
#print("SIZES")
#print("train:", Xtrain.shape) 
#print("val", Xvalidate.shape)
#print("test", Xtest.shape)


###################################
#Duplicate training by weights
###################################

#to check duplication
#print("sum of weights in  X training set\n", Xtrain['weightMultiplied'].sum())
#print("describe X train\n", Xtrain['weightMultiplied'].describe())
#print("sum of weights in y training set\n", ytrain['weightMultiplied'].sum())
#print("describe t train\n", ytrain['weightMultiplied'].describe())


#multiply training set by weights

XtrainDuplicate1 = Xtrain1.loc[Xtrain1.index.repeat(Xtrain1.weightMultiplied)]
XtrainDuplicate2 = Xtrain2.loc[Xtrain2.index.repeat(Xtrain2.weightMultiplied)]
#print("shape x duplidated", XtrainDuplicate.shape)

ytrainDuplicate1 = ytrain1.loc[ytrain1.index.repeat(ytrain1.weightMultiplied)]
ytrainDuplicate2 = ytrain2.loc[ytrain2.index.repeat(ytrain2.weightMultiplied)]
#print("shape y duplidated", ytrainDuplicate.shape)


#export the dfs for illicit drug use
Xtrain1.to_csv(r' ', index = False)
ytrain1.to_csv(r' ', index = False)
XtrainDuplicate1.to_csv(r' ', index = False)
ytrainDuplicate1.to_csv(r' ', index = False)
Xvalidate1.to_csv(r' ', index = False)
yvalidate1.to_csv(r' ', index = False)
Xtest1.to_csv(r' ', index = False)
ytest1.to_csv(r' ', index = False) 

#for alcohol and tobacco use
Xtrain2.to_csv(r' ', index = False)
ytrain2.to_csv(r' ', index = False)
XtrainDuplicate2.to_csv(r' ', index = False)
ytrainDuplicate2.to_csv(r' ', index = False)
Xvalidate2.to_csv(r' ', index = False)
yvalidate2.to_csv(r' ', index = False)
Xtest2.to_csv(r' ', index = False)
ytest2.to_csv(r' ', index = False) 








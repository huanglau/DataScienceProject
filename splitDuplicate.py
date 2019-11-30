
import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats


#import data - change filename in future
df = pd.read_csv('dfManual')

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

df['weightMultiplied'] = df['weight'] * 100


#check 
# print("multiplied weights", df['weightMultiplied'].describe())
# print("Shape of data", df.shape)
# print("sum of all weights  adn other stuff", df['weight'].describe())


# SEPARATE OUTCOMES FROM PREDICTORS

#extract outcomes - names may differ
y = df[['weightMultiplied', 'modOneOutcome', 'modTwoOutcome']]

#extract predictors
X = df.drop(['modOneOutcome', 'modTwoOutcome'], axis = 'columns')

#get training set
Xtrain, XRemaining, ytrain, yRemaining = train_test_split(X, y, test_size = (testSize + validationSize), random_state = 0)

#split remaning into test and validation 
Xvalidate, Xtest, yvalidate, ytest = train_test_split(XRemaining, yRemaining, test_size = testSize, random_state = 0)

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

XtrainDuplicate = Xtrain.loc[Xtrain.index.repeat(Xtrain.weightMultiplied)]
#print("shape x duplidated", XtrainDuplicate.shape)

ytrainDuplicate = ytrain.loc[ytrain.index.repeat(ytrain.weightMultiplied)]
#print("shape y duplidated", ytrainDuplicate.shape)


#export the dfs
Xtrain.to_csv(r' ', index = False)
ytrain.to_csv(r' ', index = False)
XtrainDuplicate.to_csv(r' ', index = False)
ytrainDuplicate.to_csv(r' ', index = False)
Xvalidate.to_csv(r' ', index = False)
yvalidate.to_csv(r' ', index = False)
Xtest.to_csv(r' ', index = False)
ytest.to_csv(r' ', index = False) 






#next steps

#train models on
# ytrain - no duplicated
# ytrainDuplicate - duplicated by factor of 100 * weight (may change)

#select model on validation set (Xvalidate, yvalidate)

#test best model on testing set (Xtest, ytest) 






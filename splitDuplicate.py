
import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats


#import data - change filename in future
df1 = pd.read_csv('dfManual1')
df2 = pd.read_csv('dfManual2')

print("shape of first df", df1.shape)
print("shape of second df", df2.shape) 

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
y1 = df1[['weightMultiplied', 'modOneOutcome']]
y2 = df2[['weightMultiplied', 'modTwoOutcome']]

#extract predictors
X1 = df1.drop(['modOneOutcome'], axis = 'columns')
X2 = df2.drop(['modTwoOutcome'], axis = 'columns')

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



#export the dfs for illicit drug use
Xtrain1.to_csv(r'   /Xtrain1', index = False)
ytrain1.to_csv(r'  /ytrain1', index = False)
XtrainDuplicate1.to_csv(r'  /XtrainDuplicate1', index = False)
ytrainDuplicate1.to_csv(r' /ytrainDuplicate1', index = False)
Xvalidate1.to_csv(r' /Xvalidate1', index = False)
yvalidate1.to_csv(r' /yvalidate1', index = False)
Xtest1.to_csv(r' /Xtest1', index = False)
ytest1.to_csv(r' /ytest1', index = False) 

#for alcohol and tobacco use
Xtrain2.to_csv(r' /Xtrain2', index = False)
ytrain2.to_csv(r' /ytrain2', index = False)
XtrainDuplicate2.to_csv(r' /XtrainDuplicate2', index = False)
ytrainDuplicate2.to_csv(r' /ytrainDuplicate2', index = False)
Xvalidate2.to_csv(r' /Xvalidate2', index = False)
yvalidate2.to_csv(r' /yvalidate2', index = False)
Xtest2.to_csv(r' /Xtest2', index = False)
ytest2.to_csv(r' /ytest2', index = False) 








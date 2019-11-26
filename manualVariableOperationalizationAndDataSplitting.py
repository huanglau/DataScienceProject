
import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats

#import data
df = pd.read_csv('yrbs1517')

print("Shape of data", df.shape)


#operationalize first outcome

#A yes response to any of cocaine, inhalants, heroin, methamphetamines,
#hallucinogens, or ecstasy is coded as a 1 for yes illicit substance use;
#otherwise, the outcome is coded as 0.

def modOneOutcome(df):
    if df['qn49'] == 1 or df['qn50'] == 1 or df['qn51'] == 1 or df['qn52'] == 1 or df['qn53'] == 1 or df['qn57'] == 1:
        return 1
    return 0

# add first outcome to the model
df['modOneOutcome'] = df.apply(lambda df: modOneOutcome(df), axis = 1)

# get summaries

#print("counts\n", df['modOneOutcome'].value_counts())

print(3607 / 30389 * 100, "% have mod 1 outcome")



#operationalization of second outcome

#frequency of alcohol and/or tobacco product use in the past 30 days
#'q32', or 'q35' or 'q37' or 'q38' or 'q42' or 'q48'

#look at individual counts
#print(df['q32'].value_counts())
#print(df['q35'].value_counts())
#print(df['q37'].value_counts())
#print(df['q38'].value_counts())
#print(df['q42'].value_counts())
# print(df['q48'].value_counts())

#want to combine such that the highest frequency is taken across all categories

#note we initially considered including q48 (marijuana)
#excluding because different response options and because we debated including in the first place 

#note this function turns Nan's into 0s
def modTwoOutcome(df):
    highest = 0
    if (df['q32'] > highest):
        highest = df['q32']
    if (df['q35'] > highest):
        highest = df['q35']
    if (df['q37'] > highest):
        highest = df['q37']
    if (df['q38'] > highest):
        highest = df['q38']
    if (df['q42'] > highest):
        highest = df['q42']
    return highest


df['modTwoOutcome'] = df.apply(lambda df: modTwoOutcome(df), axis = 1)

print("Mod 2 value counts", df['modTwoOutcome'].value_counts())


#look at outcomes by age

# df2 = df.groupby(by = ['age'])

#print("model one outcomes by age\n", df2['modOneOutcome'].value_counts()) 

#print("model two outcomes by age\n", df2['modTwoOutcome'].value_counts())

#print(" ")

###################################
#Manual Predictor Operationalization
###################################

#used to check availability by year
#df15 = df[df['year'] == 2015]
#df17 = df[df['year'] == 2017]

#used to check missingness
# df['q8'].isnull().sum()

#check coding
# df['q15'].value_counts() 



#### safety practices

#make missingness level explicit - might want to  do this for a lot of touchy questions, up to some level of missingness
#there was no prefer not to answer option on the questions
#note that we will also need to check and remove people that have high missingness across questions - that indicates not going through  the survey rather than not wanting to respond

#q8 #15% NA - seat belt - may be one where missingness is informative due to social pressures

#q9 #riding in car with intoxicated driver

#q10 #drinking and driving you

#q11 #text or email while driving

#combined q8 through q9  into composite variable with category for not answered 


#function to create a composite variable for safety behaviours
# relies on questions 8,9,10,11
# Final values:
# 0 - Missing (i.e. no  answer provided after remove high  missingness rows) 
# 1 - never
# 2 - Rarely/non habitual levels
# 3 - frequent
# 4 - always/a lot


def safetyPracticesComposite(df):

    #reverse code question 8
    if (df['q8'] == 1 or df['q8'] == 2):
        df['q8'] == 3
    if (df['q8'] == 3 or df['q8'] == 4):
        df['q8'] == 2
    if (df['q8'] == 5):
        df['q8'] == 1
    

    #set up variable for final score
    highestSafety = 0

    #check if any have the highest risk category
    if (df['q8'] == 3 or df['q9'] == 5 or df['q9'] == 4 or df['q10'] == 6 or df['q10'] == 7 or df['q10'] == 8 or df['q11'] == 6 or df['q11'] == 7 or df['q11'] == 8):
        highestSafety = 3
    #then second highest
    elif (df['q8'] == 2 or df['q9'] == 3 or df['q9'] == 2 or df['q10'] == 4 or df['q10'] == 5 or df['q11'] == 5 or df['q11'] == 4):
        highestSafety = 2
    #then lowest
    elif (df['q8'] == 1 or df['q9'] == 1 or df['q10'] == 2 or df['q10'] == 3 or df['q11'] == 2 or df['q11'] == 3):
        highestSafety = 1
    #then not applicable for those that have option 
    elif (df['q11'] == 1 or df['q10'] == 1):
        highestSafety = 0 

    return highestSafety


#apply to df 
df['safetyComposite'] = df.apply(lambda df: safetyPracticesComposite(df), axis = 1)

print("safety composite\n", df['modTwoOutcome'].value_counts())


#### violence
#  when there were two questions 1) in general 2) specific to school, 1 was chosen. 

# q12 #carry weapon - 72% are no.
#make code for no answer
#otehrwise use as is 

#### perceived or actioned threat to personal safety 
#q15 #flet unsife so did not go to school
#q16 #threatened by someone physically
#q17 #physical fight
#NaNs coded as 0

def perceivedThreatComposite(df):

  #set up variable for final score
    highestThreat = 0
    #very frequent get score 4
    if (df['q15'] == 5 or df['q16'] == 8 or df['q16'] == 7 or df['q17'] == 8 or df['q17'] == 7):
        highestThreat = 4
    
    elif (df['q15'] == 4 or df['q16'] == 6 or df['q16'] == 5 or df['q16'] == 4 or df['q17'] == 6 or df['q17'] == 5 or df['q17'] == 4):
        highestThreat = 3

     #lowest level threat
    elif (df['q15'] == 3 or df['q15'] == 2 or df['q16'] == 2 or df['q16'] == 3 or df['q17'] == 2 or df['q17'] == 3):
        highestThreat = 2
        
       #no threat
    elif (df['q15'] == 1 or df['q16'] == 1 or df['q17'] == 1):
        highestThreat = 1
        
    return highestThreat

df['perceivedThreatComposite'] = df.apply(lambda df: perceivedThreatComposite(df), axis = 1)

print("perceived threat composite\n", df['perceivedThreatComposite'].value_counts())




#### sexual assault

#q19 #forced sexual intercourse 
#q21 # date sexual assault
#q22 ##date violence

#create composite binary outcome of the above, with NA as a tgurd missingness category
# any assault can have negative impact, so lump all non zero positive responses 

def sexualAssaultComposite(df):

    highestSexualAssault = 0

    #at least once of one  type
    if (df['q19'] == 1 or df['q21'] == 3 or df['q21'] == 4 or df['q21'] == 5 or df['q21'] == 6 or df['q22'] == 3 or df['q22'] == 4 or df['q22'] == 5 or df['q22'] == 6):
        highestSexualAssault = 2
        
    #no instances
    elif (df['q19'] == 2 or df['q21'] == 2 or df['q22'] == 2):
         highestSexualAssault = 1

    #no answer or not applicale
    else:
        highestSexualAssault = 0

    return highestSexualAssault

df['sexualAssaultComposite'] = df.apply(lambda df: sexualAssaultComposite(df), axis = 1)

print("sexual assault composite\n", df['sexualAssaultComposite'].value_counts())




#### bullying

#q23
#q24

#create composite
def bullyingComposite(df):
    
    highestBullying = 0
    
    if (df['q23'] == 1 or df['q24'] == 1):
        highestBullying = 2
    elif (df['q23'] == 2 or df['q24'] == 2):
        highestBullying = 1

    return highestBullying

df['bullyingComposite'] = df.apply(lambda df: bullyingComposite(df), axis = 1)

print("bullying compisite\n", df['bullyingComposite'].value_counts())



#### depressive symptoms or suicidal ideation or attempts
# note q28 no extra info so remove
#q25
#q26
#q27

#create composite wiht level for NA

def deprSuicComposite(df):
    
    highestDeprSuic = 0
    
    if (df['q25'] == 1 or df['q26'] == 1 or df['q27'] == 1):
        highestDeprSuic = 2
    elif (df['q25'] == 2 or df['q26'] == 2 or df['q27'] == 2):
        highestDeprSuic = 1
    else:
        highestDeprSuic = 0

    return highestDeprSuic

df['deprSuicComposite'] = df.apply(lambda df: deprSuicComposite(df), axis = 1)
    
    
print("depr suic comp\n", df['deprSuicComposite'].value_counts())



#### sexual intercourse
    
# q62 #number of sexual partners, see if can logically fill in some wiht q 59

## TRY LOGICAL IMPUTATION     then use as is


#### weight
#use as is 
#these two will need an alternative missingness strategy 

#q68 #perceived weight

# q69 #trying to lose? 16.5% missingness - might want to use since tobacco suppresses appetite 


#### nutrition (some serve as  proxy for food security and SES)  
# use as is 

#qnfr2 #fruit or fruit juice at least twice per day

#qnveg2 #vegetables at least twice per day


#### exercise

#q79 #physical activity
#q82 #number of days in PE

#create composite
def physicalActivityComposite(df):

    physicalActivityDays = 0 #NaN becomes this

    if (df['q79'] == 8):
        physicalActivityDays = 8

    elif (df['q79'] == 7):
        physicalActivityDays = 7

    elif (df['q79'] == 6 or df['q82'] == 6):
        physicalActivityDays = 6

    elif (df['q79'] == 5 or df['q82'] == 5):
        physicalActivityDays = 5

    elif (df['q79'] == 4 or df['q82'] == 4):
        physicalActivityDays = 4

    elif (df['q79'] == 3 or df['q82'] == 3):
        physicalActivityDays = 3

    elif (df['q79'] == 2 or df['q82'] == 2):
        physicalActivityDays = 2

    elif (df['q79'] == 1 or df['q82'] == 1):
        physicalActivityDays = 1 #0 days
        
    return physicalActivityDays


df['physicalActivityComposite'] = df.apply(lambda df: physicalActivityComposite(df), axis = 1)




#### sedentary time
# use as is
#q80 #hours tv
#q81 #hours video games




# health

#q86 #dentist check up - this one doesn't make sense for logical imputation
#q87 #asthma dx
#q88 #hours of sleep
#q89 #grades in school - maybe remove bc  13% missing and a school could figure this out



#### Demographics

#selected variables to cover main 'categories' of information
#checked missingness - the only one over  10% was sexual contact questions -
# - removed as q66 captures sexual contact and q67 captures sexuality (misses potential interaction of sexuality not matching sexual contacts)
#when multiple options for one category, the most detailed option  was selected

#demographic vars
# demographicPredictors = df[ ['age', 'sex', 'race7', 'q67', 'q66']]


predictors = df[ ['age', 'sex', 'race7', 'q67', 'q66', 'safetyComposite', 'safetyComposite', 'perceivedThreatComposite', 'sexualAssaultComposite', 'bullyingComposite', 'deprSuicComposite', 'physicalActivityComposite',
         'q12', 'q62', 'q68', 'q69', 'qnfr2', 'qnveg2', 'q80', 'q81', 'q86', 'q87', 'q88', 'q89']]

print("shape predicotrs\n", predictors.shape)
        


#not creating substance use at this point since outcomes.
#may look at q31 - q58 if above do not work well 


###################################
#data splitting
###################################
#test set
testSize = int(((2 * 0.5) / 0.015)**2) #4444

#validation set

#to check the probability of correct model selection, with different validation test sizes
# loc is mean = -0.01 for 1% model difference 
# scale is std deviation = depends on n; using 0.5 for bound on loss standard deviation in formula
# x is the value to go up to = 0 for getting the negative quadrant

stats.norm.cdf(x = 0, loc = -0.01, scale = 0.0076696)
#tried this for several SD values; I think n = 8500 seems reasonable 

validationSize = 8500




#extract outcomes

y = df[['modOneOutcome', 'modTwoOutcome']]
#y1 = df.modOneOutcome
#y2 = df.modTwoOutcome


#extract predictors

#need to have df more cleaned before this step
X = df.drop(['modOneOutcome', 'modTwoOutcome'], axis = 'columns')

#get training set
Xtrain, XRemaining, ytrain, yRemaining = train_test_split(X, y, test_size = (testSize + validationSize), random_state = 0)

#split remaning into test and validation 
Xvalidate, Xtest, yvalidate, ytest = train_test_split(XRemaining, yRemaining, test_size = testSize, random_state = 0)

#check sizes 
print("SIZES")
print("train:", Xtrain.shape) #17445

print("val", Xvalidate.shape)

print("test", Xtest.shape)




###################################
#Weighting stuff
###################################

#multiply training set by weights

#assess on validation set if multiplication improves  model performance

#use the better trained model on testing set 






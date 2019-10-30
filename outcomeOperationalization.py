
import numpy  as np
import pandas as pd

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

print("counts\n", df['modOneOutcome'].value_counts())

print(3607 / 30389 * 100, "% have mod 1 outcome")



#operationalization of second outcome

#frequency of alcohol and/or tobacco product use in the past 30 days
#'q32', or 'q35' or 'q37' or 'q38' or 'q42' or 'q48'

#look at individual counts
print(df['q32'].value_counts())
print(df['q35'].value_counts())
print(df['q37'].value_counts())
print(df['q38'].value_counts())
print(df['q42'].value_counts())
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

print(df['modTwoOutcome'].value_counts())


#look at outcomes by age

df2 = df.groupby(by = ['age'])

print("model one outcomes by age\n", df2['modOneOutcome'].value_counts()) 

print("model two outcomes by age\n", df2['modTwoOutcome'].value_counts())

print(" ") 



import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats


#read in data 2015 and 2017
df = pd.read_csv('nationalYRBS20172015')

print("Shape of data", df.shape)

df15 = df[df['year'] == 2015]
df17 = df[df['year'] == 2017]



for j in range(8, 90):
    if ((df17['q' + str(j)].isnull().sum() / df17.shape[0]) == 1):
       print("question not in 2017", j)
    else:
        print("checked", j)


for i in range(8, 90):
    if ((df15['q' + str(i)].isnull().sum() / df15.shape[0]) == 1):
        print("question not in 2015", i, df.columns[i])






        
#question not in 2015 14
#question not in 2015 20
#question not in 2015 31
#question not in 2015 36
#question not in 2015 37
#question not in 2015 39
#question not in 2015 44
#question not in 2015 56
#question not in 2015 84

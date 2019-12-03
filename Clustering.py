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

if os.getcwd not in sys.path:
    sys.path.append(os.getcwd())


"""
Extensions to thek-Means Algorithm for ClusteringLarge Data Sets with Categorical Values

K -means doesn't work for catagorical data. k-modes does

https://pypi.org/project/kmodes/
"""
#%%
# load data for model1
dfModel1 =  pd.read_csv("../data/SelectedFeatures/Model1WOStratum_RFECV135.csv")
#ModelTrain(dfModel1, 'Ever used ilicit drugs', 'Model1/5FoldModel1NN_RFECV/')
X = dfModel1.drop(['Ever used ilicit drugs'], axis='columns')
#y = pd.read_csv("../data/SelectedFeatures/Model1_10Feat_KBest_chi2.csv")
#y = pd.DataFrame(y['Ever used ilicit drugs'])

km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)
#X = dfModel1.drop(['Ever used ilicit drugs'], axis='columns')
#y = pd.DataFrame(dfModel1['Ever used ilicit drugs'])
clusters = km.fit_predict(X)

# Print the cluster centroids
print(km.cluster_centroids_)

#%%
# load data for model1
dfModel2 = pd.read_csv("../data/SelectedFeatures/Model2WOStratum_RFECV118.csv")
km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)
X = dfModel2.drop(['Freq Drug Use'], axis='columns')
y = pd.DataFrame(dfModel2['Freq Drug Use'])
clusters = km.fit_predict(X)

# Print the cluster centroids
print(km.cluster_centroids_)
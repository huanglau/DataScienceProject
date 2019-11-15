#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:54:20 2019

@author: vector
"""

import keras
import numpy as np
import keras.backend as K

#%% other accuracy models
def Precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    Implimented in a previous version of keras
    """
#    y_pred = K.cast(y_pred > 0.5, y_true.dtype)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def Recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    

    
    In binary it's the sensitivity TP/(TP+FN)
    """
#    y_pred = K.cast(y_pred > 0.5, y_true.dtype) # need this for siamese networks
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
#%% build model

def fcNet(iInputSize = 100, pretrained_weights = None, iNumFreeze = 0):
    """ 
    """
   

    model = Sequential()
    model.add(Dense(256, input_dim=iInputSize, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


#%%
model = fcNet(iInputSize = np.shape(Xtrain.values)[1], pretrained_weights = None, iNumFreeze = 0)
#model.add(Dense(12, input_dim=np.shape(Xtrain.values)[1], activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', Recall, Precision])

# fit the keras model on the dataset
X = np.array(Xtrain.values)
y = np.reshape(np.array(ytrain.values, dtype=np.int8), (-1))
model.fit(X, y, epochs=150, batch_size=2**10,
              class_weight = {0:1, 1:100})
# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:01:59 2019
These functions are used in setp.py
These helpers are associated with saving the model, training, testing and saving 
accuracies, plots and roc curves
This trains, vals and tests a parallel CNN whos architecture is defined in ParallelNet.py
This intakes a directory which has a structure as follows. This does cross validation
where each patient a directory and left out. 
Function takes the program setup.py, HelperFunctions.py, ParalllelNet.py
and DataGeneratorFunctions.py and saves it to an output directory that also contains all
of the saved accuracies, models, etc.
@author: lhuang
"""
import matplotlib.pyplot as plt
#import pickle
import keras
import os
import numpy as np
import sklearn
import random
import sklearn.model_selection
from keras.utils import to_categorical
#from keras.callbacks import ModelCheckpoint
#import DataGeneratorFunctions #as DataGenFun
#import IO #as IO
#import ErrorMetrics #as Error
#%%

def flatten_model(model_nested):
    # https://stackoverflow.com/questions/54648296/how-to-flatten-a-nested-model-keras-functional-api
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    model_flat = keras.models.Sequential(layers_flat)
    return model_flat

def SplitDirs(npFiles, np2FoldIndexes, fTestSize, iFold,  bShuffle = True, iSeed = 0):
    """ splits a numpy array depending on a numpy array. Removes the ith index, then splits the rest
    of the list into training, validation, and testing sets
    
    Arguments:
        npFiles: list of file pairs. This list will be split
        np2FoldIndexes: list of indexes to be split in the folds, Shape 2xN
        iFold: ith fold
        fTestSize: size of the testing batch. 
        bShuffle: wheather or not the training and validation list will be shuffled
        iSeed: random seed for the shuffle
        
    returns:
        np1TrainDir, np1ValDir, np1TestDir: npFiles but split into 3 groups
            depending on n3FoldIndexes.
            From np2FoldIndexes the iFold group of indexes is selected. The remaining
            indexes in np2FoldIndexes is shuffled or not, then split into training and validation.
    
    example:
        npFiles = [g1, g2, g3, g4, g5, g6]
        np2FoldIndexes = [[0,1],[2,3],[4,5]]
        iFold = 0
        fTestSize = 0.25, shufffle = True
        Returns: npTestDir = [g1, g2],  npValDir = [np4], npTrainDir = [g6, g3, g5]
    """
    np1TestDir = npFiles[np2FoldIndexes[iFold]]
    # remove all directories but ith one
    lUniqueDirNoI = npFiles[FlattenAndRemoveIndex(np2FoldIndexes, iFold)]
    # randomly split the remaining files (train and val) by splitting an index
    TrainIndex, ValIndex = sklearn.model_selection.train_test_split(range(0, len(lUniqueDirNoI)), test_size = fTestSize , shuffle = bShuffle, random_state = iSeed)
    np1TrainDir = lUniqueDirNoI[TrainIndex]
    np1ValDir = lUniqueDirNoI[ValIndex]
    # error check to ensure all patients have been accounted for
    if (len(np1TrainDir) + len(np1ValDir) + len(np1TestDir)) != len(npFiles) :
        raise ValueError('Num train, val, and test patients not equal to number of unique patients. Patient data was lost in the data generation')
    return np1TrainDir, np1ValDir, np1TestDir 
                                  
def CreateNFolds(npGroupList, iNFolds, bShuffle = True, seed = 1):
    """ Creates an 2 d numpy array of indexes with N arrays of arrays.
    The values of the list are NOT returned. This uses random.random package
    as np.random package is not thread safe
    Arguments
        npGroupList: Input list that will be split into N arrays.
        iNFolds: Number of splits
        shuffle: Indicates if the groups will be shuffled or not
        seed: if shuffle = True this indicates what random seed, if any, will be used to
            initiate the shuffle.
            
    Return
        A 2D numpy array containing the indexes. For example if there are 10 groups 
        and iNFolds = 2, then the returned array would be [[1,2,3,4,5], [6,7,8,9,10]]
        if shuffling was False or [[10,5,9,3,6], [4,8,2,7,1]] if shuffling was true
    """
    # TODO: make error metrics for this
    ilIndexes = np.arange(len(npGroupList))
    if seed is not None:
        random.seed(seed)
    if bShuffle:
        random.shuffle(ilIndexes)
    return np.array_split(ilIndexes, iNFolds)

def FoldNSplit(npFiles, bShufflePatientOrder, iPatientOrderSeed, fTestSize, bShuffleSplitOrder,  iNumFoldsTotal, iFold):
    """ breaks npFiles into iNumFoldsTotal. Then splits into train, val and test sets
    """
    np2FoldIndexes = CreateNFolds(npFiles, iNumFoldsTotal, bShuffle = bShufflePatientOrder, seed = iPatientOrderSeed)
    # generate list of training, validatio and testing directories
    np1TrainDir, np1ValDir, np1TestDir = SplitDirs(npFiles, np2FoldIndexes, fTestSize, iFold, bShuffleSplitOrder)
    return np1TrainDir, np1ValDir, np1TestDir

def Classifications(TruthLabels, Predictions):
    """ Calculates percent accuracy given output is not hot encoded. 
    The truth label and prediction is only one number
    I.e. 
    TruthLabels = [0,1,1,0]
    Predictions = [1,1,1,0]
    for 4 items. Two are correct and two are incorrect
    """
    Predictions = [x for sublist in Predictions for x in sublist]#np.array(predictions).reshape(-1)
    Predictions = np.array(Predictions).flatten()
    TruthLabels = [x for sublist in TruthLabels for x in sublist]
    percentAcc = compute_accuracy(TruthLabels, Predictions)
    return percentAcc, TruthLabels, Predictions


def FlattenAndRemoveIndex(ListOfLists, NIndex):
    """ Flattens a list of lists after removing the Nth Index. Can only remove one index
    """
    ListOfLists =  ListOfLists[:NIndex] +  ListOfLists[NIndex+1:]
    return [item for sublist in ListOfLists for item in sublist]
    

def compute_accuracy(y_true, y_pred, Thresh = 0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred > Thresh
    return np.mean(pred == y_true)

    return SectionDict
import pandas as pd
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import csv

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import tree, linear_model

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from random import shuffle

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scipy import interpolate
import math


def Classifier(allSamples,numattr):
    #print("старт Log_reg_c\n")

    shuffle(allSamples)

    nArray = np.matrix(allSamples, dtype=float)

    X = (nArray[:,:numattr-1])#отделили векторы обьектов от меток
    y = np.array(nArray[:,numattr-1])#отделяем последний столбец меток от обьектов
 
    kf = KFold(n_splits=10)#тестом будет произвольно выбранная 1/10 часть данных
    clf = linear_model.LogisticRegression()

    
    meanTP, meanFP, mean_precision = 0,0,0
    i=0

    for train, test in kf.split(X, y):
        i+=1
        
        predicted = clf.fit(X[train], np.ravel(y[train])).predict(X[test])
        #кажддому обьекту метка класса 0 или 1

        TP, FP, N, P, precision = 0,0,0,0,0        
        
        tmp0 = np.reshape(y[test],len(y[test]))

        t = np.array(tmp0, dtype=int)
        p = np.array(predicted, dtype=int)
        e = np.array(np.ones(len(y[test])), dtype=int)
        '''
        print(t)
        print(p)
        print(e)
        '''
        P = np.sum(y[test])  
        N = len(y[test]) - P

        tmp1 = np.bitwise_and(t,p)
        #print(tmp1)
        #tmp2 = np.logical_and(t, p)
        TP = np.sum(tmp1)
        #print(TP)

        tmp3 = np.subtract(e,t)
        #print(tmp3)
        tmp4 = np.bitwise_and(tmp3, p)
        #print(tmp4)
        FP = np.sum(tmp4)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        #print(FP)

        TP /= P
        FP /= N             
        precision = TP/(TP+FP)
        '''
        print(TP)
        print(FP)
        '''
        meanTP += TP
        meanFP += FP 
        mean_precision += precision

    meanTP /= 10 #делим на число сплитов(усредняем)
    meanFP /= 10
    mean_precision /= 10

    #print("Log_reg_c выполнился\n")
    #print('(',meanFP,',',meanTP,')')
    if math.isnan(meanFP) or math.isnan(meanTP):
        return 0,0,0
    else:
        return meanFP, meanTP, mean_precision
    


#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from Smote import SMOTE_c
from RandomUnderSampling import UnderSampling

from Classifier import NaiveBayes_c
from Classifier import LogReg_c
from Classifier import RandomForest_c

import matplotlib.pyplot as plt
import csv
import random
from random import randint
import math
import operator
import csv
from sklearn import metrics
import pandas as pd
import numpy as np


def main():
    #print("Forest found\n")
    dataFile1=r'datasets/ForestCover.txt'
    classColumnNumber = 55
    minorClassValue='1'
    numattr=55


    #print("MPA found\n")
    dataFile2 = r'datasets/Mammography.txt'
    classColumnNumberMammo = 7
    minorClassValueMammo = '1'
    numattrMammo = 7



    #print("Phoneme found\n")
    dataFile3 = r'datasets/phoneme.txt'
    classColumnNumberPhoneme = 6
    minorClassValuePhoneme = '1'
    numattrPhoneme = 6




    #Тут надо выбрать данные на которых проводить анализ

    performClassifying(dataFile3,classColumnNumberPhoneme, numattrPhoneme,minorClassValuePhoneme, 'Phoneme')
    
    performClassifying(dataFile2,classColumnNumberMammo, numattrMammo,minorClassValueMammo, 'Mammography')
    
    #очень долго
    #performClassifying(dataFile1,classColumnNumber, numattr,minorClassValue, 'ForestCover')

#====================================================================================================
#выполянем сравнительный анализ разных подходов для конкретных данных

def performClassifying(dataFile,classColumnNumber,numattrs,minorClassValue, data_name):
    
    print('We perfom experiment on '+data_name+' dataset\n')

    dataSet, MinorityData, MajorityData = getSeparatedSamples(dataFile, classColumnNumber, minorClassValue)
    
    #Cледующая строка нужна только если используем Smote
    KNN_matrix = SMOTE_c.compute_nearest_neighbors(dataSet, MinorityData, 5)
    
    #_________Выберем лучший уровень Smote из 5 вариантов для двух классификаторов

    #Plot_Smote_ROC_parameters_LR(dataSet, MinorityData, MajorityData, numattrs, minorClassValue, data_name,KNN_matrix,'LogReg')
    
    #Plot_Smote_ROC_parameters_RF(dataSet, MinorityData, MajorityData, numattrs, minorClassValue, data_name,KNN_matrix,'Random Forest')
    
    #________Сравним методы решения проблемы несбалансированности
    
    Plot_Roc_Undersampling_LR(dataSet, MinorityData, MajorityData, numattrs, minorClassValue, data_name,KNN_matrix,'LogReg')
    Plot_Smote_ROC_LR(dataSet, MinorityData, MajorityData, numattrs, minorClassValue, data_name,KNN_matrix,'LogReg')
    Plot_Roc_Naive_Nayes(dataSet, numattrs, data_name)
    #Plot_ROC_RandomOverSampling_LR(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,'LogReg')
    plt.savefig(data_name+' ROC LogReg compare.png')
    plt.show()
    
    
    Plot_Roc_Undersampling_RF(dataSet, MinorityData, MajorityData, numattrs, minorClassValue, data_name,KNN_matrix,'Random Forest')
    Plot_Smote_ROC_RF(dataSet, MinorityData, MajorityData, numattrs, minorClassValue, data_name,KNN_matrix,'Random Forest')
    Plot_Roc_Naive_Nayes(dataSet, numattrs, data_name)
    #Plot_ROC_RandomOverSampling_RF(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,'Random Forest')
    plt.savefig(data_name+' ROC Random Forest compare.png')
    plt.show()
    
#==============================================================================================================
#функции работы с данными

def unite_data(part1, part2):
    
    both = []

    for eachRow in part1:
        both.append(eachRow)
    for eachRow in part2:
        both.append(eachRow)
    
    #print("обьединил датасеты\n")
    return both


def getSeparatedSamples(filename,number_of_columns,minorityClassValue):
    minoritySamples = []
    majoritySamples = []
    dataset=[]

    #print("читаю даннные и разделяю на мин и макс классы\n")

    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if (row[number_of_columns-1] == minorityClassValue):
                minoritySamples.append(row)
            else:
                majoritySamples.append(row)
            dataset.append(row)
    csvfile.close
    return dataset, minoritySamples, majoritySamples

'''
def clean_points(fprs, tprs):
    err = 1

    while err > 0:
        for j in range(len(fprs)):
            if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
                fprs.pop(j)
                tprs.pop(j)
                err += 1
                break
        if err > 0:
            continue
        err = 0

    return fprs, tprs               
'''
#====================================================================================================

#строим roc_curve при разных параметрах SMOTE чтоб выбрать луший параметр классификаторs - Logistic Regression & Random Forest   

def Plot_Smote_ROC_parameters_RF(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,KNN_matrix,classificator_name):
    #Для замены классификатора изменить две строки в данном коде и последний аргумент при вызове
    
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    Smote_levels = [100, 200, 300, 400, 500]

    colors = ['r','g','c','b','y']
    markers = ['o', 'v' ,'+', '*','s']
    i = -1
    
    #fp, tp, precision = LogReg_c.Classifier(dataSet, numattrs)
    fp, tp = RandomForest_c.Classifier(dataSet, numattrs)
    if fp == 0 and tp == 0:
        pass
    else:
        plt.plot(fp, tp, 'kx')#классификация на исходных данных


    for SN in Smote_levels:
    #for SN in [100]:
        fprs = []
        tprs = []
        precisions = []
        i += 1

        print('\n Smote number level ', SN)

        Final_minority_Data =[] 
        Synthetic_minority_Data = []

        Synthetic_minority_Data = SMOTE_c.SMOTE(len(MinorityData), SN, MinorityData, numattrs, minorClassValue, KNN_matrix, 5)
        Final_minority_Data = unite_data(Synthetic_minority_Data, MinorityData)
        f_min_len = len(Final_minority_Data)

        for N in Undersampling_levels:
            print('undersampling ', N)

            underSampled_majority_Data = []
            Undersampled_Smoted_Dataset = []

            underSampled_majority_Data = UnderSampling.underSampling(f_min_len, MajorityData, N)
            
            Undersampled_Smoted_Dataset = unite_data(Final_minority_Data, underSampled_majority_Data)
            print('total number of samples ',len(Undersampled_Smoted_Dataset))     

            #fp, tp, precision = LogReg_c.Classifier(Undersampled_Smoted_Dataset, numattrs)
            fp, tp = RandomForest_c.Classifier(Undersampled_Smoted_Dataset, numattrs)

            if fp == 0 and tp == 0:
                pass
            else:
                fprs.append(fp)
                tprs.append(tp)
            
                plt.scatter(fp, tp, s=10, c=colors[i],marker=markers[i])
        
        fprs.append(1)
        tprs.append(1)
        
        for j in range(len(fprs)):
            if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
                fprs.pop(j)
                tprs.pop(j)
        
        #fprs.sort()
        roc_auc = 0
        try:
            roc_auc = metrics.auc(fprs, tprs)
        except Exception:
            pass

        plt.plot(fprs, tprs, color = colors[i],label=str(SN)+'-Smote, AUC = '+"%.4f" % roc_auc,lw=1)
    
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC curves with '+classificator_name)
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)

    plt.savefig(data_name+' Smote_parameters ROC with '+classificator_name+'.png')
    #plt.savefig('ROC curves for '+classificator_type+' classificator'+'.pdf')
    plt.show()


def Plot_Smote_ROC_parameters_LR(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,KNN_matrix,classificator_name):
    #Для замены классификатора изменить две строки в данном коде и последний аргумент при вызове
    
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    Smote_levels = [100, 200, 300, 400, 500]

    colors = ['r','g','c','b','y']
    markers = ['o', 'v' ,'+', '*','s']
    i = -1
    
    fp, tp, precision = LogReg_c.Classifier(dataSet, numattrs)
    #fp, tp = RandomForest_c.Classifier(dataSet, numattrs)
    if fp == 0 and tp == 0:
        pass
    else:
        plt.plot(fp, tp, 'kx')#классификация на исходных данных


    for SN in Smote_levels:
    #for SN in [100]:
        fprs = []
        tprs = []
        precisions = []
        i += 1

        print('\n Smote number level ', SN)

        Final_minority_Data =[] 
        Synthetic_minority_Data = []

        Synthetic_minority_Data = SMOTE_c.SMOTE(len(MinorityData), SN, MinorityData, numattrs, minorClassValue, KNN_matrix, 5)
        Final_minority_Data = unite_data(Synthetic_minority_Data, MinorityData)
        f_min_len = len(Final_minority_Data)

        for N in Undersampling_levels:
            print('undersampling ', N)

            underSampled_majority_Data = []
            Undersampled_Smoted_Dataset = []

            underSampled_majority_Data = UnderSampling.underSampling(f_min_len, MajorityData, N)
            
            Undersampled_Smoted_Dataset = unite_data(Final_minority_Data, underSampled_majority_Data)
            print('total number of samples ',len(Undersampled_Smoted_Dataset))     

            fp, tp, precision = LogReg_c.Classifier(Undersampled_Smoted_Dataset, numattrs)
            #fp, tp = RandomForest_c.Classifier(Undersampled_Smoted_Dataset, numattrs)

            if fp == 0 and tp == 0:
                pass
            else:
                fprs.append(fp)
                tprs.append(tp)
            
                plt.scatter(fp, tp, s=10, c=colors[i],marker=markers[i])
    
        fprs.append(1)
        tprs.append(1)
        
        #print('points',fprs, '\n', tprs)

        #fprs, tprs = clean_points(fprs, tprs)        
        #print('points\n',fprs,'\n',tprs)
        #fprs.sort()
        #print('points',fprs, '\n', tprs)
        for j in range(len(fprs)):
            if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
                fprs.pop(j)
                tprs.pop(j)
        

        roc_auc = 0
        try:
            roc_auc = metrics.auc(fprs, tprs)
        except Exception:
            pass

        plt.plot(fprs, tprs, color = colors[i],label=str(SN)+'-Smote, AUC = '+"%.4f" % roc_auc,lw=1)
    
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC curves with '+classificator_name)
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)

    plt.savefig(data_name+' Smote_parameters ROC with '+classificator_name+'.png')
    #plt.savefig('ROC curves for '+classificator_type+' classificator'+'.pdf')
    plt.show()




    ''' Uncomment this and replace plotting line to plot the precision-recall curve with LogReg

        plt.plot(tprs, precisions, color = colors[i],label=str(N)+'-Smote, AUC = '+"%.4f" % roc_auc,lw=1)
        plt.savefig(data_name+' Precision-Recall curve with LogReg.png')
        plt.show()
    '''



def Plot_Roc_Naive_Nayes(dataSet, numattrs, data_name):

    fprs = []
    tprs = []

    for i in range(50):
        fp, tp = NaiveBayes_c.Classifier(dataSet, numattrs, 1/(i+1), i/(i+1) )   
        if fp == 0 and tp == 0:
            pass
        else:
            tprs.append(tp)
            fprs.append(fp)
            plt.scatter(fp, tp, s=10, c='r',marker='*')

    tprs.append(1)
    fprs.append(1)
    
    '''
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    #print(fprs,'\n',tprs)

    plt.plot(fprs, tprs, color = 'r',label='Naive Bayes',lw=1)

    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)

    #plt.savefig(data_name+' ROC 1.png')        
        

def Plot_Roc_Undersampling_LR(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,KNN_matrix,classificator_name):
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    
    
    fprs = []
    tprs = []
    
    for N in Undersampling_levels:
        print('undersampling ', N)

        underSampled_majority_Data = []
        Undersampled_Smoted_Dataset = []

        underSampled_majority_Data = UnderSampling.underSampling(len(MinorityData), MajorityData, N)
            
        Undersampled_Dataset = unite_data(MinorityData, underSampled_majority_Data)

        print('total number of samples ',len(Undersampled_Dataset))     

        fp, tp, precision = LogReg_c.Classifier(Undersampled_Dataset, numattrs)
        #fp, tp = RandomForest_c.Classifier(Undersampled_Dataset, numattrs)
        
        #ДЛЯ КАЖДОГО СЛУЧАЯ ВЫБРАТЬ СВОЙ КЛАССИФИКАТОР ЗАМЕНОЙ ЭТОЙ СТРОКИ И ЛЕГЕНДЫ ДЛЯ ЭТОЙ КРИВОЙ
        if fp == 0 and tp == 0:
            pass
        else:
            fprs.append(fp)
            tprs.append(tp)
            plt.scatter(fp, tp, s=10, c='g',marker='v')
    
    fprs.append(1)
    tprs.append(1)
    '''    
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    fprs.sort()
    roc_auc = 0
    try:
        roc_auc = metrics.auc(fprs, tprs)
    except Exception:
        pass

    plt.plot(fprs, tprs, color = 'g',label='Under-'+classificator_name+', AUC = '+"%.4f" % roc_auc,lw=1)
    
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)
    
    #plt.savefig(data_name+' ROC 2.png')


def Plot_Roc_Undersampling_RF(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,KNN_matrix,classificator_name):
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    

    fprs = []
    tprs = []
    
    for N in Undersampling_levels:
        print('undersampling ', N)

        underSampled_majority_Data = []
        Undersampled_Smoted_Dataset = []

        underSampled_majority_Data = UnderSampling.underSampling(len(MinorityData), MajorityData, N)
            
        Undersampled_Dataset = unite_data(MinorityData, underSampled_majority_Data)

        print('total number of samples ',len(Undersampled_Dataset))     

        #fp, tp, precision = LogReg_c.Classifier(Undersampled_Dataset, numattrs)
        fp, tp = RandomForest_c.Classifier(Undersampled_Dataset, numattrs)
        
        #ДЛЯ КАЖДОГО СЛУЧАЯ ВЫБРАТЬ СВОЙ КЛАССИФИКАТОР ЗАМЕНОЙ ЭТОЙ СТРОКИ И ЛЕГЕНДЫ ДЛЯ ЭТОЙ КРИВОЙ
        if fp == 0 and tp == 0:
            pass
        else:
            fprs.append(fp)
            tprs.append(tp)
            plt.scatter(fp, tp, s=10, c='g',marker='v')
    
    fprs.append(1)
    tprs.append(1)
    '''   
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    fprs.sort()
    roc_auc = 0
    try:
        roc_auc = metrics.auc(fprs, tprs)
    except Exception:
        pass

    plt.plot(fprs, tprs, color = 'g',label='Under-'+classificator_name+', AUC = '+"%.4f" % roc_auc,lw=1)
   
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)
 
    #plt.savefig(data_name+' ROC 2.png')


def Plot_Smote_ROC_LR(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,KNN_matrix, classificator_name):
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    Smote_level = 300 
    #ЛУЧШИЙ ПАРАМЕТР ДЛЯ ЭТОГО КЛАССИФИКАТОРА

    fprs = []
    tprs = []

    Synthetic_minority_Data = SMOTE_c.SMOTE(len(MinorityData), Smote_level, MinorityData, numattrs, minorClassValue, KNN_matrix, 5)
    Final_minority_Data = unite_data(Synthetic_minority_Data, MinorityData)
    final_min_len = len(Final_minority_Data)

    for N in Undersampling_levels:
        print('undersampling ', N)

        underSampled_majority_Data = UnderSampling.underSampling(final_min_len, MajorityData, N)
            
        Undersampled_Smoted_Dataset = unite_data(Final_minority_Data, underSampled_majority_Data)
        print('total number of samples ',len(Undersampled_Smoted_Dataset))     

        fp, tp, precision = LogReg_c.Classifier(Undersampled_Smoted_Dataset, numattrs)
        #fp, tp = RandomForest_c.Classifier(Undersampled_Dataset, numattrs)
        if fp == 0 and tp == 0:
            pass
        else:
            fprs.append(fp)
            tprs.append(tp)
            plt.scatter(fp, tp, s=10, c='b', marker='o')
    
    fprs.append(1)
    tprs.append(1)
    '''
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    fprs.sort()
    roc_auc = 0
    try:
        roc_auc = metrics.auc(fprs, tprs)
    except Exception:
        pass

    plt.plot(fprs, tprs, color = 'b',label=str(Smote_level)+'-Smote-'+classificator_name+', AUC = '+"%.4f" % roc_auc,lw=1)
  
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)
  
    #plt.savefig(data_name+' ROC 3.png')




def Plot_Smote_ROC_RF(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,KNN_matrix, classificator_name):
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    Smote_level = 500 
    #ЛУЧШИЙ ПАРАМЕТР ДЛЯ ЭТОГО КЛАССИФИКАТОРА

    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC with LogReg')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)

    fprs = []
    tprs = []

    Synthetic_minority_Data = SMOTE_c.SMOTE(len(MinorityData), Smote_level, MinorityData, numattrs, minorClassValue, KNN_matrix, 5)
    Final_minority_Data = unite_data(Synthetic_minority_Data, MinorityData)
    final_min_len = len(Final_minority_Data)

    for N in Undersampling_levels:
        print('undersampling ', N)

        underSampled_majority_Data = UnderSampling.underSampling(final_min_len, MajorityData, N)
            
        Undersampled_Smoted_Dataset = unite_data(Final_minority_Data, underSampled_majority_Data)
        print('total number of samples ',len(Undersampled_Smoted_Dataset))     

        #fp, tp, precision = LogReg_c.Classifier(Undersampled_Smoted_Dataset, numattrs)
        fp, tp = RandomForest_c.Classifier(Undersampled_Smoted_Dataset, numattrs)
        if fp == 0 and tp == 0:
            pass
        else:
            fprs.append(fp)
            tprs.append(tp)
            plt.scatter(fp, tp, s=10, c='b', marker='o')
    
    fprs.append(1)
    tprs.append(1)
    '''
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    fprs.sort()

    roc_auc = 0
    try:
        roc_auc = metrics.auc(fprs, tprs)
    except Exception:
        pass

    plt.plot(fprs, tprs, color = 'b',label=str(Smote_level)+'-Smote-'+classificator_name+', AUC = '+"%.4f" % roc_auc,lw=1)
   
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)
 
    #plt.savefig(data_name+' ROC 3.png')

#=================================================================
def Plot_ROC_RandomOverSampling_LR(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,classificator_name):
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    Oversampling_level = 300 
    #ЛУЧШИЙ ПАРАМЕТР ДЛЯ ЭТОГО КЛАССИФИКАТОРА на этих данных
    Oversampled_minority_Data = []
    
    print('minority ', len(MinorityData))

    N = int(len(MinorityData)*Oversampling_level/100)

    print('to choose from min ',N )
    
    random_Minority_indeces = np.random.choice(range(len(MinorityData)), N, replace=True)
    #берем N произвольных разных образцов меньшего класса

    print('выбрал индексы мин класса')
    
    num = len(random_Minority_indeces)
    
    print ('chosen ', num)

    for index in random_Minority_indeces:
        Oversampled_minority_Data.append(MinorityData[index])

    Final_minority_Data = unite_data(Oversampled_minority_Data, MinorityData)
    final_min_len = len(Final_minority_Data)

    print('total min ', final_min_len)

    fprs = []
    tprs = []
    
    for N in Undersampling_levels:
        print('undersampling ', N)

        underSampled_majority_Data = UnderSampling.underSampling(final_min_len, MajorityData, N)
            
        UnderOversampled_Dataset = unite_data(Final_minority_Data, underSampled_majority_Data)
        print('total number of samples ',len(UnderOversampled_Dataset))     

        fp, tp, precision = LogReg_c.Classifier(UnderOversampled_Dataset, numattrs)
        #fp, tp = RandomForest_c.Classifier(Undersampled_Dataset, numattrs)
        if fp == 0 and tp == 0:
            pass
        else:
            fprs.append(fp)
            tprs.append(tp)
            plt.scatter(fp, tp, s=10, c='y', marker='s')
    
    fprs.append(1)
    tprs.append(1)
    '''
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    fprs.sort()
    roc_auc = 0
    try:
        roc_auc = metrics.auc(fprs, tprs)
    except Exception:
        pass

    plt.plot(fprs, tprs, color = 'y',label=str(Oversampling_level)+'-Over-'+classificator_name+', AUC = '+"%.4f" % roc_auc,lw=1)
  
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)

    #plt.savefig(data_name+' Over ROC with '+classificator_name+'.png')
    

def Plot_ROC_RandomOverSampling_RF(dataSet,MinorityData,MajorityData,numattrs,minorClassValue, data_name,classificator_name):
    Undersampling_levels = [10, 15,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # 18 точек в каждой roc
    Oversampling_level = 500 
    #ЛУЧШИЙ ПАРАМЕТР ДЛЯ ЭТОГО КЛАССИФИКАТОРА на этих данных
    Oversampled_minority_Data = []
    
    print('minority ', len(MinorityData))

    N = int(len(MinorityData)*Oversampling_level/100)

    print('to choose from min ',N )
    
    random_Minority_indeces = np.random.choice(range(len(MinorityData)), N, replace=True)
    #берем N произвольных разных образцов меньшего класса

    print('выбрал индексы мин класса')
    
    num = len(random_Minority_indeces)
    
    print ('chosen ', num)

    for index in random_Minority_indeces:
        Oversampled_minority_Data.append(MinorityData[index])

    Final_minority_Data = unite_data(Oversampled_minority_Data, MinorityData)
    final_min_len = len(Final_minority_Data)

    print('total min ', final_min_len)

    fprs = []
    tprs = []
    
    for N in Undersampling_levels:
        print('undersampling ', N)

        underSampled_majority_Data = UnderSampling.underSampling(final_min_len, MajorityData, N)
            
        UnderOversampled_Dataset = unite_data(Final_minority_Data, underSampled_majority_Data)
        print('total number of samples ',len(UnderOversampled_Dataset))     

        fp, tp = RandomForest_c.Classifier(UnderOversampled_Dataset, numattrs)
        if fp == 0 and tp == 0:
            pass
        else:
            fprs.append(fp)
            tprs.append(tp)
            plt.scatter(fp, tp, s=10, c='y', marker='s')
    
    fprs.append(1)
    tprs.append(1)
    '''
    for j in range(len(fprs)):
        if (j <= len(fprs)-2) and (fprs[j] > fprs[j+1]):
            fprs.pop(j)
            tprs.pop(j)
        
    print('points\n',fprs,'\n',tprs)
    '''
    fprs.sort()
    roc_auc = 0
    try:
        roc_auc = metrics.auc(fprs, tprs)
    except Exception:
        pass

    plt.plot(fprs, tprs, color = 'y',label=str(Oversampling_level)+'-Over-'+classificator_name+', AUC = '+"%.4f" % roc_auc,lw=1)
  
    plt.rcParams['font.size'] = 12
    plt.title(data_name+' ROC')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(False)
    plt.legend(loc=0)

    #plt.savefig(data_name+' Over ROC with '+classificator_name+'.png')
    

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import csv


def underSampling(num_of_minority, MajorityData, relation):
    #print("старт андерсэмплинга\n")

    underSampledData = []
    random_Majority_indeces = []

    N = int(num_of_minority/(relation / 100))

    random_Majority_indeces = np.random.choice(range(len(MajorityData)), N, replace=True)
    #берем N произвольных разных образцов большего класса

    #print('выбрал индексы макс класса')

    for index in random_Majority_indeces:
        underSampledData.append(MajorityData[index])

    print("Undersampling number ", len(underSampledData))
        
    return underSampledData#тут N выбранных образцов макс класса





'''
    with open(r"../output/Synthetic_Data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(underSampledData)
        writer.writerows(minorityClassData)

'''
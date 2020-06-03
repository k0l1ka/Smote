import csv
import random
import math
import operator
import csv


def populate(N, minorSamples, knnarray, numattrs,MinorClassName, k):
    synthetic_positive_dataset = []

    for i in range(len(minorSamples)):

        for _ in range(N):
            nn = random.randint(0, k-1)
            newUnit = []

            for attr in range(0, numattrs - 1):
                diff = float(knnarray[i][nn][attr]) - float(minorSamples[i][attr])
                gap = random.uniform(0, 1)
                newUnit.append(float(minorSamples[i][attr]) + gap * diff)
        
            newUnit.append(MinorClassName)
            synthetic_positive_dataset.append(newUnit)
            
    print('Created synthetic samples ',len(synthetic_positive_dataset),'\n')       
    
    return synthetic_positive_dataset




#создыем исскуственные образцы
def SMOTE(min_class_volume, N, minorSamples, numattrs, minorClassName, knnArray, k):

    #print('старт Smote','\n')
    print('Minority number = ',len(minorSamples),'\n')

    if (N < 100):
        print("Number of sample to be generated should be more than 100%")
        raise ValueError

    N = int(N / 100)
    
    #print('конец Smote')

    return populate(N, minorSamples, knnArray, numattrs, minorClassName, k)#тут синтетические образцы мин класса

#=============================================================================================


def euclideanDistance(a, b, length):
    distance = 0
    for x in range(length):
        distance += pow((float(a[x]) - float(b[x])), 2)
    
    #print("in smote: distance")

    return math.sqrt(distance)


def get_KNeighbours(dataSet, eachMinorsample, k):
    distances = []
    #count = 0
    length = len(eachMinorsample) - 1

    for x in range(len(dataSet)):
        dist = euclideanDistance(eachMinorsample, dataSet[x], length)
        #count += 1
        #print("расстояние номер ", count, "\n")
        distances.append((dataSet[x], dist))
    
    distances.sort(key=operator.itemgetter(1))
    
    neighbours = []

    for x in range(k):
        neighbours.append(distances[x + 1][0])

    #print("get_KNeighbours works")

    return neighbours




#для каждого образца мин класса найдем по 5 ближайших соседей
def compute_nearest_neighbors(dataSet, minorSamples, k):
    knnArray = []
    
    print('Look for k nearest_neighbors for each minority sample\n') 

    for eachMinorsample in minorSamples:
        knnArray.append(get_KNeighbours(dataSet, eachMinorsample, k))

    #print('нашел соcедей')    
    return knnArray

    



'''
print("in smote: создаем исскуственные образцы и пишем их в output/Synthetic_Data.csv")

    with open(r"output/Synthetic_Data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(synthetic_dataset)
'''



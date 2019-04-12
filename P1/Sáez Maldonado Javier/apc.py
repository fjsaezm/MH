import scipy
import sys
import math
import sys
import numpy as np
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from prepareData import *
from statistics import *
from algorithms import *
from numpy.random import uniform
import time

n1 = 'Datasets/colposcopy.arff'
n2 = 'Datasets/ionosphere.arff'
n3 = 'Datasets/texture.arff'
"d1 is the data, m1 is the metadata"
d1,c1 = load_arff(n1)
d2,c2 = load_arff(n2)
d3,c3 = load_arff(n3)
d1 = normalizeData(d1)
d2 = normalizeData(d2)
d3 = normalizeData(d3)

"Results. Rows: partition-i. Cols: T-clas,T-red,Agr,T"

"1-NN"
resCol1NN = [[0 for x in range(4)] for y in range (5)]
resIon1NN = [[0 for x in range(4)] for y in range (5)]
resText1NN = [[0 for x in range(4)] for y in range (5)]

"GreedyRelief"

resColGR = [[0 for x in range(4)] for y in range (5)]
resIonGR = [[0 for x in range(4)] for y in range (5)]
resTextGR = [[0 for x in range(4)] for y in range (5)]

"Local search"

resColLS = [[0 for x in range(4)] for y in range (5)]
resIonLS = [[0 for x in range(4)] for y in range (5)]
resTextLS = [[0 for x in range(4)] for y in range (5)]

"Indicates partition"
i = 0

skf = StratifiedKFold(n_splits = 5)

print("File: ", n1)
#---------------------------
#---------------------------
#------- COLPOSCOPY --------
#---------------------------
#---------------------------
for trainIndex , testIndex in skf.split(d1,c1):
    onennTime = 0
    greedyTime = 0
    localSTime = 0
    n = testIndex.shape[0]
    onennRight = 0
    greedyRight = 0
    lsRight = 0
    "Time for weights calc"
    startTime = time.time()
    greedyWeights = greedyRelief(d1[trainIndex],c1[trainIndex])
    greedyTime += time.time() - startTime
    ini_uniform_weights = np.random.uniform(0.0,1.0,d1.shape[1])
    "Time for waeights calc"
    startTime = time.time()
    lsWeights = localSearch(ini_uniform_weights,d1[trainIndex],c1[trainIndex])
    localSTime += time.time() - startTime

    for element in testIndex:
        startTime = time.time()
        if one_nn(d1[trainIndex],c1[trainIndex],d1[element]) == c1[element]:
            onennRight +=1
        onennTime += time.time()-startTime

        startTime = time.time()
        if weighted_onenn(greedyWeights,d1[trainIndex],c1[trainIndex],d1[element]) == c1[element]:
            greedyRight +=1
        greedyTime += time.time() -startTime

        startTime = time.time()
        if weighted_onenn(lsWeights,d1[trainIndex],c1[trainIndex],d1[element]) == c1[element]:
            lsRight += 1
        localSTime += time.time() - startTime

    "Save results"
    resCol1NN[i][0] = onennRight/n
    resCol1NN[i][1] = 0 #its always 0 in this case
    resCol1NN[i][2] = 0.5*(onennRight/n)
    resCol1NN[i][3] = onennTime
    resColGR[i][0]  = greedyRight/n
    resColGR[i][1]  = tasa_red(greedyWeights)
    resColGR[i][2]  = f(greedyWeights,d1,c1)
    resColGR[i][3]  = greedyTime
    resColLS[i][0]  = lsRight/n
    resColLS[i][1]  = tasa_red(lsWeights)
    resColLS[i][2]  = f(lsWeights,d1,c1)
    resColLS[i][3]  = localSTime
    i+=1
    print("Partition ",i)

i = 0

#---------------------------
#---------------------------
#------- IONOSPHERE --------
#---------------------------
#---------------------------

print("File :", n2)
for trainIndex , testIndex in skf.split(d2,c2):
    onennTime = 0
    greedyTime = 0
    localSTime = 0
    n = testIndex.shape[0]
    onennRight = 0
    greedyRight = 0
    lsRight = 0
    "Time for weights calc"
    startTime = time.time()
    greedyWeights = greedyRelief(d2[trainIndex],c2[trainIndex])
    greedyTime += time.time() - startTime
    ini_uniform_weights = np.random.uniform(0.0,1.0,d2.shape[1])
    "Time for weights calc"
    startTime = time.time()
    lsWeights = localSearch(ini_uniform_weights,d2[trainIndex],c2[trainIndex])
    localSTime += time.time() - startTime

    for element in testIndex:
        startTime = time.time()
        if one_nn(d2[trainIndex],c2[trainIndex],d2[element]) == c2[element]:
            onennRight +=1
        onennTime += time.time()-startTime
        startTime = time.time()
        if weighted_onenn(greedyWeights,d2[trainIndex],c2[trainIndex],d2[element]) == c2[element]:
            greedyRight +=1
        greedyTime += time.time() -startTime
        startTime = time.time()
        if weighted_onenn(lsWeights,d2[trainIndex],c2[trainIndex],d2[element]) == c2[element]:
            lsRight += 1
        localSTime += time.time() - startTime

    resIon1NN[i][0] = onennRight/n
    resIon1NN[i][1] = 0 #its always 0 in this case
    resIon1NN[i][2] = 0.5*(onennRight/n)
    resIon1NN[i][3] = onennTime
    resIonGR[i][0]  = greedyRight/n
    resIonGR[i][1]  = tasa_red(greedyWeights)
    resIonGR[i][2]  = f(greedyWeights,d2,c2)
    resIonGR[i][3]  = greedyTime
    resIonLS[i][0]  = lsRight/n
    resIonLS[i][1]  = tasa_red(lsWeights)
    resIonLS[i][2]  = f(lsWeights,d2,c2)
    resIonLS[i][3]  = localSTime
    i+=1
    print("Partition ",i)
    

i = 0


#---------------------------
#---------------------------
#--------- TEXTURE ---------
#---------------------------
#---------------------------

print("File:  ", n3)
for trainIndex , testIndex in skf.split(d3,c3):
    onennTime = 0
    greedyTime = 0
    localSTime = 0
    n = testIndex.shape[0]
    onennRight = 0
    greedyRight = 0
    lsRight = 0
    "Time for weights calc"
    startTime = time.time()
    greedyWeights = greedyRelief(d3[trainIndex],c3[trainIndex])
    greedyTime += time.time() - startTime
    ini_uniform_weights = np.random.uniform(0.0,1.0,d3.shape[1])
    "Time for waeights calc"
    startTime = time.time()
    lsWeights = localSearch(ini_uniform_weights,d3[trainIndex],c3[trainIndex])
    localSTime += time.time() - startTime

    for element in testIndex:
        startTime = time.time()
        if one_nn(d3[trainIndex],c3[trainIndex],d3[element]) == c3[element]:
            onennRight +=1
        onennTime += time.time()-startTime
        startTime = time.time()
        if weighted_onenn(greedyWeights,d3[trainIndex],c3[trainIndex],d3[element]) == c3[element]:
            greedyRight +=1
        greedyTime += time.time() -startTime
        startTime = time.time()
        if weighted_onenn(lsWeights,d3[trainIndex],c3[trainIndex],d3[element]) == c3[element]:
            lsRight += 1
        localSTime += time.time() - startTime


    resText1NN[i][0] = onennRight/n
    resText1NN[i][1] = 0 #its always 0 in this case
    resText1NN[i][2] = 0.5*(onennRight/n)
    resText1NN[i][3] = onennTime
    resTextGR[i][0]  = greedyRight/n
    resTextGR[i][1]  = tasa_red(greedyWeights)
    resTextGR[i][2]  = f(greedyWeights,d3,c3)
    resTextGR[i][3]  = greedyTime
    resTextLS[i][0]  = lsRight/n
    resTextLS[i][1]  = tasa_red(lsWeights)
    resTextLS[i][2]  = f(lsWeights,d3,c3)
    resTextLS[i][3]  = localSTime
            
    i +=1
    print("Partition ",i)

print("Finished algorithms\n")
print("\nResults for 1NN")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resCol1NN[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIon1NN[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resText1NN[i][j],end=" - ")
    print("\n",end="")


print("\nResults for Greedy Relief")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColGR[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonGR[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextGR[i][j],end=" - ")
    print("\n",end="")


print("\nResults for Local Search")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColLS[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonLS[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextLS[i][j],end=" - ")
    print("\n",end="")


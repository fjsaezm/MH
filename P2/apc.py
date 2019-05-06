import scipy
import sys
import math
import numpy as np
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from prepareData import *
from statistics import *
from algorithms import *
from genetics import *
from memetics import *
from declarations import *
from numpy.random import uniform
import time

"Indicates partition"
i = 0

skf = StratifiedKFold(n_splits = 5)

print("File: ", n1)
for trainIndex , testIndex in skf.split(d1,c1):
    onennTime  = greedyTime = localSTime = 0
    AGGBLXTime = AGGACTime  = AGEBLXTime = AGEACTime = 0
    AM1Time    = AM2Time    = AM3Time    = 0
    n = testIndex.shape[0]
    #onenn
    start = time.time()
    t_clas_onenn = newone_nn(d1, c1,trainIndex,testIndex)
    onennTime = time.time() - start
    #Greedy
    start = time.time()
    t_clas_greedy, t_red_greedy = newgreedyRelief(d1,c1,trainIndex,testIndex)
    greedyTime = time.time() - start
    #Local search
    start = time.time()
    t_clas_local, t_red_local = newLocalSearch(d1,c1,trainIndex,testIndex)
    localSTime = time.time() - start
    #AGGBLX
    start = time.time()
    t_clas_aggblx,t_red_aggblx = AGG(d1,c1,trainIndex,testIndex,BLX)
    AGGBLXTime = time.time()-start

    #AGGAC
    start = time.time()
    t_clas_agg,t_red_aggac = AGG(d1,c1,trainIndex,testIndex,arithmeticCross)
    AGGACTime = time.time()-start

    #AGEBLX
    start = time.time()
    t_clas_ageblx,t_red_ageblx = AGE(d1,c1,trainIndex,testIndex,BLX)
    AGEBLXTime = time.time()-start

    #AGEAC
    start = time.time()
    t_clas_ageac,t_red_ageac = AGE(d1,c1,trainIndex,testIndex,artithmeticCross)
    AGGBLXTime = time.time()-start

    #AM1
    start = time.time()
    t_clas_am1,t_red_am1 = AM(d1,c1,trainIndex,testIndex,BLX,am1)
    AM1Time = time.time()-start

    #AM2
    start = time.time()
    t_clas_am2,t_red_am2 = AM(d1,c1,trainIndex,testIndex,BLX,am2)
    AM2Time = time.time()-start

    #AM3
    start = time.time()
    t_clas_am3,t_red_am3 = AM(d1,c1,trainIndex,testIndex,BLX,am3)
    AM3Time = time.time()-start
    
    # "Save results"
    resCol1NN[i][0] = t_clas_onenn
    resCol1NN[i][1] = 0 #its always 0 in this case
    resCol1NN[i][2] = 0.5*t_clas_onenn
    resCol1NN[i][3] = onennTime
    resColGR[i][0]  = t_clas_greedy
    resColGR[i][1]  = t_red_greedy
    resColGR[i][2]  = 0.5*t_clas_greedy + 0.5*t_red_greedy
    resColGR[i][3]  = greedyTime
    resColLS[i][0]  = t_clas_local
    resColLS[i][1]  = t_red_local
    resColLS[i][2]  = 0.5*t_clas_local + 0.5*t_red_local
    resColLS[i][3]  = localSTime
    resColAGGBLX[i][0]
    i+=1
    print("Partition ",i)

    
i = 0


print("File: ", n2)
for trainIndex , testIndex in skf.split(d2,c2):
    onennTime = 0
    greedyTime = 0
    localSTime = 0
    n = testIndex.shape[0]
    start = time.time()
    t_clas_onenn = newone_nn(d2, c2,trainIndex,testIndex)
    onennTime = time.time() - start
    start = time.time()
    t_clas_greedy, t_red_greedy = newgreedyRelief(d2,c2,trainIndex,testIndex)
    greedyTime = time.time() - start
    start = time.time()
    t_clas_local, t_red_local = newLocalSearch(d2,c2,trainIndex,testIndex)
    localSTime = time.time() - start
    # "Save results"
    resIon1NN[i][0] = t_clas_onenn
    resIon1NN[i][1] = 0 #its always 0 in this case
    resIon1NN[i][2] = 0.5*t_clas_onenn
    resIon1NN[i][3] = onennTime
    resIonGR[i][0]  = t_clas_greedy
    resIonGR[i][1]  = t_red_greedy
    resIonGR[i][2]  = 0.5*t_clas_greedy + 0.5*t_red_greedy
    resIonGR[i][3]  = greedyTime
    resIonLS[i][0]  = t_clas_local
    resIonLS[i][1]  = t_red_local
    resIonLS[i][2]  = 0.5*t_clas_local + 0.5*t_red_local
    resIonLS[i][3]  = localSTime
    i+=1
    print("Partition ",i)

i = 0

print("File: ", n3)
for trainIndex , testIndex in skf.split(d3,c3):
    onennTime = 0
    greedyTime = 0
    localSTime = 0
    n = testIndex.shape[0]
    start = time.time()
    t_clas_onenn = newone_nn(d3, c3,trainIndex,testIndex)
    onennTime = time.time() - start
    start = time.time()
    t_clas_greedy, t_red_greedy = newgreedyRelief(d3,c3,trainIndex,testIndex)
    greedyTime = time.time() - start
    start = time.time()
    t_clas_local, t_red_local = newLocalSearch(d3,c3,trainIndex,testIndex)
    localSTime = time.time() - start
    # "Save results"
    resText1NN[i][0] = t_clas_onenn
    resText1NN[i][1] = 0 #its always 0 in this case
    resText1NN[i][2] = 0.5*t_clas_onenn
    resText1NN[i][3] = onennTime
    resTextGR[i][0]  = t_clas_greedy
    resTextGR[i][1]  = t_red_greedy
    resTextGR[i][2]  = 0.5*t_clas_greedy + 0.5*t_red_greedy
    resTextGR[i][3]  = greedyTime
    resTextLS[i][0]  = t_clas_local
    resTextLS[i][1]  = t_red_local
    resTextLS[i][2]  = 0.5*t_clas_local + 0.5*t_red_local
    resTextLS[i][3]  = localSTime
    i+=1
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


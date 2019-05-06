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
    n = testIndex.shape[0]
    #onenn
    print("onenn")
    start = time.time()
    resCol1NN[i][0] = newone_nn(d1, c1,trainIndex,testIndex)
    resCol1NN[i][1] = 0
    resCol1NN[i][2] = 0.5*resCol1NN[i][0]
    resCol1NN[i][3] = time.time() - start
    #Greedy
    print("greedy")
    start = time.time()
    resColGR[i][0],resColGR[i][1]  = newgreedyRelief(d1,c1,trainIndex,testIndex)
    resColGR[i][2] = 0.5*(resColGR[i][0]+resColGR[i][1])
    resColGR[i][3] = time.time() - start
    #Local search
    print("LS")
    start = time.time()
    resColLS[i][0], resColLS[i][1] = newLocalSearch(d1,c1,trainIndex,testIndex)
    resColLS[i][2] = 0.5*(resColLS[i][0] + resColLS[i][1])
    resColLS[i][3] = time.time() - start
    #AGGBLX
    print("AGGBLX")
    start = time.time()
    resColAGGBLX[i][0],resColAGGBLX[i][1] = AGG(d1,c1,trainIndex,testIndex,BLX)
    resColAGGBLX[i][2] = 0.5*(resColAGGBLX[i][0] + resColAGGBLX[i][1])
    resColAGGBLX[i][3] = time.time()-start

    #AGGAC
    print("AGGAC")
    start = time.time()
    resColAGGAC[i][0],resColAGGAC[i][1] = AGG(d1,c1,trainIndex,testIndex,arithmeticCross)
    resColAGGAC[i][2] = 0.5*(resColAGGAC[i][0] + resColAGGAC[i][1])
    resColAGGAC[i][3] = time.time()-start

    #AGEBLX
    print("AGEBLX")
    start = time.time()
    resColAGEBLX[i][0],resColAGEBLX[i][1] = AGE(d1,c1,trainIndex,testIndex,BLX)
    resColAGEBLX[i][2] = 0.5*(resColAGEBLX[i][0] + resColAGEBLX[i][1])
    resColAGEBLX[i][3] = time.time()-start

    #AGEAC
    print("AGEAC")
    start = time.time()
    resColAGEAC[i][0],resColAGEAC[i][1] = AGE(d1,c1,trainIndex,testIndex,arithmeticCross)
    resColAGEAC[i][2] = 0.5*(resColAGEAC[i][0] + resColAGEAC[i][1])
    resColAGEAC[i][3] = time.time()-start

    #AM1
    print("AM1")
    start = time.time()
    resColAM1[i][0],resColAM1[i][1] = AM(d1,c1,trainIndex,testIndex,BLX,am1)
    resColAM1[i][2] = 0.5*(resColAM1[i][0] + resColAM1[i][1])
    resColAM1[i][3] = time.time()-start

    #AM2
    print("AM2")
    start = time.time()
    resColAM2[i][0],resColAM2[i][1] = AM(d1,c1,trainIndex,testIndex,BLX,am2)
    resColAM2[i][2] = 0.5*(resColAM2[i][0] + resColAM2[i][1])
    resColAM2[i][3] = time.time()-start
    #AM3
    print("AM3")
    start = time.time()
    resColAM3[i][0],resColAM3[i][1] = AM(d1,c1,trainIndex,testIndex,BLX,am3)
    resColAM3[i][2] = 0.5*(resColAM3[i][0] + resColAM3[i][1])
    resColAM3[i][3] = time.time()-start
    
    i+=1
    print("Partition ",i)

    
i = 0


print("File: ", n2)
for trainIndex , testIndex in skf.split(d2,c2):
    n = testIndex.shape[0]
    #onenn
    start = time.time()
    resIon1NN[i][0] = newone_nn(d2, c2,trainIndex,testIndex)
    resIon1NN[i][1] = 0
    resIon1NN[i][2] = 0.5*resIon1NN[i][0]
    resIon1NN[i][3] = time.time() - start
    #Greedy
    start = time.time()
    resIonGR[i][0],resIonGR[i][1]  = newgreedyRelief(d2,c2,trainIndex,testIndex)
    resIonGR[i][2] = 0.5*(resIonGR[i][0]+resIonGR[i][1])
    resIonGR[i][3] = time.time() - start
    #Local search
    start = time.time()
    resIonLS[i][0], resIonLS[i][1] = newLocalSearch(d2,c2,trainIndex,testIndex)
    resIonLS[i][2] = 0.5*(resIonLS[i][0] + resIonLS[i][1])
    resIonLS[i][3] = time.time() - start
    #AGGBLX
    start = time.time()
    resIonAGGBLX[i][0],resIonAGGBLX[i][1] = AGG(d2,c2,trainIndex,testIndex,BLX)
    resIonAGGBLX[i][2] = 0.5*(resIonAGGBLX[i][0] + resIonAGGBLX[i][1])
    resIonAGGBLX[i][3] = time.time()-start

    #AGGAC
    start = time.time()
    resIonAGGAC[i][0],resIonAGGAC[i][1] = AGG(d2,c2,trainIndex,testIndex,arithmeticCross)
    resIonAGGAC[i][2] = 0.5*(resIonAGGAC[i][0] + resIonAGGAC[i][1])
    resIonAGGAC[i][3] = time.time()-start

    #AGEBLX
    start = time.time()
    resIonAGEBLX[i][0],resIonAGEBLX[i][1] = AGE(d2,c2,trainIndex,testIndex,BLX)
    resIonAGEBLX[i][2] = 0.5*(resIonAGEBLX[i][0] + resIonAGEBLX[i][1])
    resIonAGEBLX[i][3] = time.time()-start

    #AGEAC
    start = time.time()
    resIonAGEAC[i][0],resIonAGEAC[i][1] = AGE(d2,c2,trainIndex,testIndex,arithmeticCross)
    resIonAGEAC[i][2] = 0.5*(resIonAGEAC[i][0] + resIonAGEAC[i][1])
    resIonAGEAC[i][3] = time.time()-start

    #AM1
    start = time.time()
    resIonAM1[i][0],resIonAM1[i][1] = AM(d2,c2,trainIndex,testIndex,BLX,am1)
    resIonAM1[i][2] = 0.5*(resIonAM1[i][0] + resIonAM1[i][1])
    resIonAM1[i][3] = time.time()-start

    #AM2
    start = time.time()
    resIonAM2[i][0],resIonAM2[i][1] = AM(d2,c2,trainIndex,testIndex,BLX,am2)
    resIonAM2[i][2] = 0.5*(resIonAM2[i][0] + resIonAM2[i][1])
    resIonAM2[i][3] = time.time()-start
    #AM3
    start = time.time()
    resIonAM3[i][0],resIonAM3[i][1] = AM(d2,c2,trainIndex,testIndex,BLX,am3)
    resIonAM3[i][2] = 0.5*(resIonAM3[i][0] + resIonAM3[i][1])
    resIonAM3[i][3] = time.time()-start
    
    i+=1
    print("Partition ",i)

i = 0

print("File: ", n3)
for trainIndex , testIndex in skf.split(d3,c3):
    n = testIndex.shape[0]
    #onenn
    start = time.time()
    resText1NN[i][0] = newone_nn(d3, c3,trainIndex,testIndex)
    resText1NN[i][1] = 0
    resText1NN[i][2] = 0.5*resText1NN[i][0]
    resText1NN[i][3] = time.time() - start
    #Greedy
    start = time.time()
    resTextGR[i][0],resTextGR[i][1]  = newgreedyRelief(d3,c3,trainIndex,testIndex)
    resTextGR[i][2] = 0.5*(resTextGR[i][0]+resTextGR[i][1])
    resTextGR[i][3] = time.time() - start
    #Local search
    start = time.time()
    resTextLS[i][0], resTextLS[i][1] = newLocalSearch(d3,c3,trainIndex,testIndex)
    resTextLS[i][2] = 0.5*(resTextLS[i][0] + resTextLS[i][1])
    resTextLS[i][3] = time.time() - start
    #AGGBLX
    start = time.time()
    resTextAGGBLX[i][0],resTextAGGBLX[i][1] = AGG(d3,c3,trainIndex,testIndex,BLX)
    resTextAGGBLX[i][2] = 0.5*(resTextAGGBLX[i][0] + resTextAGGBLX[i][1])
    resTextAGGBLX[i][3] = time.time()-start

    #AGGAC
    start = time.time()
    resTextAGGAC[i][0],resTextAGGAC[i][1] = AGG(d3,c3,trainIndex,testIndex,arithmeticCross)
    resTextAGGAC[i][2] = 0.5*(resTextAGGAC[i][0] + resTextAGGAC[i][1])
    resTextAGGAC[i][3] = time.time()-start

    #AGEBLX
    start = time.time()
    resTextAGEBLX[i][0],resTextAGEBLX[i][1] = AGE(d3,c3,trainIndex,testIndex,BLX)
    resTextAGEBLX[i][2] = 0.5*(resTextAGEBLX[i][0] + resTextAGEBLX[i][1])
    resTextAGEBLX[i][3] = time.time()-start

    #AGEAC
    start = time.time()
    resTextAGEAC[i][0],resTextAGEAC[i][1] = AGE(d3,c3,trainIndex,testIndex,arithmeticCross)
    resTextAGEAC[i][2] = 0.5*(resTextAGEAC[i][0] + resTextAGEAC[i][1])
    resTextAGEAC[i][3] = time.time()-start

    #AM1
    start = time.time()
    resTextAM1[i][0],resTextAM1[i][1] = AM(d3,c3,trainIndex,testIndex,BLX,am1)
    resTextAM1[i][2] = 0.5*(resTextAM1[i][0] + resTextAM1[i][1])
    resTextAM1[i][3] = time.time()-start

    #AM2
    start = time.time()
    resTextAM2[i][0],resTextAM2[i][1] = AM(d3,c3,trainIndex,testIndex,BLX,am2)
    resTextAM2[i][2] = 0.5*(resTextAM2[i][0] + resTextAM2[i][1])
    resTextAM2[i][3] = time.time()-start
    #AM3
    start = time.time()
    resTextAM3[i][0],resTextAM3[i][1] = AM(d3,c3,trainIndex,testIndex,BLX,am3)
    resTextAM3[i][2] = 0.5*(resTextAM3[i][0] + resTextAM3[i][1])
    resTextAM3[i][3] = time.time()-start
    
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

print("\nResults for AGG-BLX")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAGGBLX[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAGGBLX[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAGGBLX[i][j],end=" - ")
    print("\n",end="")

print("\nResults for AGG-AC")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAGGAC[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAGGAC[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAGGAC[i][j],end=" - ")
    print("\n",end="")

print("\nResults for AGE-BLX")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAGEBLX[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAGEBLX[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAGEBLX[i][j],end=" - ")
    print("\n",end="")

print("\nResults for AGE-AC")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAGEAC[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAGEAC[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAGEAC[i][j],end=" - ")
    print("\n",end="")

print("\nResults for AM(10,1.0)")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAM1[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAM1[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAM1[i][j],end=" - ")
    print("\n",end="")

print("\nResults for AM(10,0.1)")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAM2[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAM2[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAM2[i][j],end=" - ")
    print("\n",end="")

print("\nResults for AM(10,0.1mej)")
print("Partition  \t Colposcopy \t \t Ionosphere \t \t Texture \t")
print("   \t T_clas - T_red - Agr - T\tT_clas - T_red - Agr - T \tT_clas - T_red - Agr - T ")
for i in range(5):
    print(i, end="\t ")
    for j in range(4):
        print("%.2f" % resColAM3[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resIonAM3[i][j], end=" - ")
    print("\t",end="")
    for j in range(4):
        print("%.2f" % resTextAM3[i][j],end=" - ")
    print("\n",end="")

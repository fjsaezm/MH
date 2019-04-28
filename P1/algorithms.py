

import numpy as np
import operator
from scipy.spatial.distance import pdist,squareform
from sklearn.model_selection import LeaveOneOut
from numpy.random import normal
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
import time

max_iterations = 15000
alpha = 0.5

def newone_nn(data,classes,trainIndex,testIndex):

    classifier = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform')

    classifier.fit(data[trainIndex],classes[trainIndex])
    
    predictions = classifier.predict(data[testIndex])
    #print("Accuracy is", accuracy_score(classes[testIndex],predictions)*100, "% for knn")

    return accuracy_score(classes[testIndex],predictions)*100

def findFriend(data,index,classes):
    sol = []
    val_sol = 100000.0

    # loo
    aux = np.delete(data,index,0)
    class_aux = np.delete(classes,index,0)

    val_i = sum(data[index])
    for j in range(len(aux)):
        total = sum(aux[j])
        dist = (val_i - total)**2

        if dist < val_sol and class_aux[j] == classes[index]:
            sol = j
            val_sol = dist

    return data[sol]

def findEnemy(data,index,classes):
    sol = []
    val_sol = 100000.0
    aux = np.copy(data)

    val_i = sum(data[index])
    for j in range(len(aux)):
        total = sum(aux[j])
        dist = (val_i - total)**2

        if dist < val_sol and classes[j] != classes[index]:
            sol = j
            val_sol = dist

    return data[sol]
    

def newgreedyRelief(data,classes,trainIndex,testIndex):

    w = np.zeros(len(data[trainIndex][0]))
    for k in range(len(data[trainIndex])):
        friend = np.asarray(findFriend(data[trainIndex],k,classes[trainIndex]))
        enemy  = np.asarray(findEnemy(data[trainIndex],k,classes[trainIndex]))
       
        w = w + (data[trainIndex][k] - enemy) - (data[trainIndex][k] - friend)

    maxVal = max(w)
    for k in range(len(w)):
        if w[k] < 0.0 : w[k] = 0.0
        else: w[k]/maxVal

    trainW = np.copy(data[trainIndex])
    testW  = np.copy(data[testIndex])
    

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainW,classes[trainIndex])

    predictions = classifier.predict(testW)
    #print("Accuracy is", accuracy_score(classes[testIndex],predictions)*100, "% for greedy")

    return accuracy_score(classes[testIndex],predictions)*100, tasa_red(w)
    

def tasa_red(w):
    return ((w[w < 0.2].shape[0])/w.shape[0])

def mov(w,j):
    w[j] = np.clip(w[j] + np.random.normal(0.0,0.3,None), 0, 1)
    return w


def newLocalSearch(data,classes,testIndex,trainIndex):
    w = np.random.uniform(0.0,1.0,data.shape[1])

    fmax = - 10000.0

    trainD = np.copy(data[trainIndex])
    trainC = np.copy(classes[trainIndex])

    testD  = np.copy(data[testIndex])
    testC  = np.copy(classes[testIndex])

    nneighs = 0
    it      = 0
    n = len(w)

    while nneighs < 20*n and it < max_iterations:

        for k in range (n):
            
            w_orig = np.copy(w)
            w = mov(w,k)
            nneighs += 1

            trainAux = trainD * w
            testAux  = testD  * w

            classifier = KNeighborsClassifier(n_neighbors = 1)
            classifier.fit(trainAux,trainC)

            predictions = classifier.predict(testAux)
            tclas = accuracy_score(classes[testIndex],predictions)*100
            func  = 0.5*tclas + 0.5*tasa_red(w)

            if func > fmax:
                fmax = func
                it   = 0
                break
            else:
                w = w_orig
                it += 1

            if nneighs == 20*n:
                break

    trainD *= w
    testD  *= w

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    tclas       = accuracy_score(testC,predictions)*100
    tred        = tasa_red(w)

    #print("Accuracy is:", tclas, "% for LS")

    return tclas,tred
        


                
                

            

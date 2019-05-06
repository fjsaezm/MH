
import numpy as np
import operator
from scipy.spatial.distance import pdist,squareform
from sklearn.model_selection import LeaveOneOut
from numpy.random import normal
from sklearn.metrics import accuracy_score
from random import randint
from sklearn.utils import shuffle

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

    return accuracy_score(classes[testIndex],predictions)*100, tasa_red(w)
    

def tasa_clas(w,data,classes):

    dataw = data*w
    loo = LeaveOneOut()
    nright = 0
    for train,test in loo.split(dataw):
        classifier = KNeighborsClassifier(n_neighbors = 1)
        classifier.fit(dataw[train],classes[train])

        pred = classifier.predict(dataw[test])
        if pred == classes[test]: nright +=1

    return (nright/len(data))*100

def tasa_red(w):
    return ((w[w < 0.2].shape[0])/w.shape[0])

def mov(w,j):
    w[j] = np.clip(w[j] + np.random.normal(0.0,0.3,None), 0, 1)
    return w

def newLocalSearch(data,classes,testIndex,trainIndex):
    w = np.random.uniform(0.0,1.0,data.shape[1])

    trainD = np.copy(data[trainIndex])
    trainC = np.copy(classes[trainIndex])

    testD  = np.copy(data[testIndex])
    testC  = np.copy(classes[testIndex])
    fmax   = (tasa_clas(w,trainD,trainC)+tasa_red(w))*0.5

    nneighs = 0
    it      = 0
    n = len(w)

    while it < max_iterations:
        nneighs = 0
        w_original = w

        while nneighs < 20*len(w):
            k = np.random.choice(range(n))
            w_orig = w[k]
            w = mov(w,k)
            tred = tasa_red(w)
            if((len(w) - np.count_nonzero(w)) == len(w)):
                w[k] = w_orig

            trainAux = trainD * w
            tclas = tasa_clas(w,trainAux,trainC)
            func  = 0.5*(tclas+tred)
            it +=1

            if func > fmax:
                fmax = func
                break
            else:
                w = w_original
                nneighs +=1
            if it >= 15000: break

        if nneighs >= 20*len(w): break

    trainD*= w
    testD *= w

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    tclas       = accuracy_score(testC,predictions)*100
    tred        = tasa_red(w)


    return tclas,tred


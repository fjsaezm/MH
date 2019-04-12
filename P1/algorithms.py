

import numpy as np
import operator
from scipy.spatial.distance import pdist,squareform
from sklearn.model_selection import LeaveOneOut
from numpy.random import normal


max_iterations = 20000
alpha = 0.5

def distance(e1, e2):
    "Euclidean distance between e1 and e2"
    return np.sqrt(np.sum((e1-e2)**2, axis=1))

def one_nn(data, classes, example):
    "Nearest neighbour"
    return classes[np.argmin(distance(data, example))]          
    
def greedyRelief(data,classes):

    w = np.zeros(shape=(data.shape[1],))

    "Calculate all the distances previously"
    distances = squareform(pdist(data))
    np.fill_diagonal(distances, np.infty)

    for i in range(0, data.shape[0]):
        en_indices = classes != classes[i]
        fr_indices = classes == classes[i]

        enemies = data[en_indices]
        friends = data[fr_indices]

        closest_friend = np.argmin(distances[fr_indices, i])
        closest_enemy = np.argmin(distances[en_indices, i])

        w = w + np.abs(data[i]-enemies[closest_enemy]) - \
            np.abs(data[i]-friends[closest_friend])

    w[w < 0.2] = 0
    w = w/np.max(w)
    return w


def weighted_onenn(weights,data,classes,example):
    dist = weights*(data - example)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return classes[np.argmin(dist)]

def tasa_red(w):
    return ((w[w < 0.2].shape[0])/w.shape[0])

def tasa_clas(w,data,classes):
    n = data.shape[0]
    rights = 0
    loo = LeaveOneOut()
    for train_index,test_index in loo.split(data):
        if weighted_onenn(w, data[train_index], classes, data[test_index]) == classes[test_index]:
            rights += 1
            
    return rights/n

def f(w,data,classes):     
    return alpha*tasa_clas(w,data,classes) +(1-alpha)*tasa_red(w)

def mov(w,sigma,j):
    w[j] = np.clip(w[j] + np.random.normal(scale=sigma), 0, 1)
    return w

def localSearch(ini_weight,data,classes):
    n = data.shape[1]
    weights = ini_weight
    bestF = f(ini_weight,data,classes)
    "Count if mutations is less than n*20"
    notMuted = 0
    for i in range(0,max_iterations):
        for j in range(0,n):
            w = np.copy(weights)
            "Mutation"
            w = mov(w,0.3,j)
            newF = f(w,data,classes)
            if(newF > bestF):
                bestF = newF
                weights = w
                notMuted = 0
            else:
                notMuted += 1

            if notMuted == n*20:
                return weights

    return weights
                
                

            

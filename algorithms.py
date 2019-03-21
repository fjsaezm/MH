

import numpy as np


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def distance (e1,e2) :
    "Euclidean distance between 2 elements"
    
    return np.sqrt(np.sum(np.power(e1-e2, 2)))
   # l = len(e1)
   # sum = 0.0
   # for i in range ( 0,l ):
   #     sum += pow(e1[i]-e2[i],2)
   # return math.sqrt(sum)

def distance_w (e1,e2):
    "Euclidean distance between 2 elements. However, the elements which weight is less than 0.2, dont count"
    l = len(e1)
    sum = 0.0
    for i in range (0,l):
        if (e1[i] >= 0.2 and e2[i] >= 0.2):
            sum += pow(e1[i]-e2[i],2)
    return math.sqrt(sum)

def nearestNeighbour (element , neighbours, classes ) :
    "Searches for the nearest neighbour in a list"
    cmin = classes[0]
    dmin = distance(element,neighbours[0])
    for i in range (len(neighbours)):
        d = distance(element,neighbours[i])
        if(d < dmin):
            cmin = classes[i]
            dmin = d
            
    return cmin

def onenn(data,classes,trainParts,testParts):
    percentages = []
    for i in range(0,len(trainParts)):
        guess = []
        for element in data[testParts[i]]:
             guess.append(nearestNeighbour(element,data[trainParts[i]],classes[trainParts[i]]))
        rights = 0
        for j in range (0,len(guess)):
            if(guess[i] == classes[testParts[i][j]]):
                rights += 1

        per = float((rights/len(guess)))*100
        percentages.append(per)

    avg = mean(percentages)

    print("Avg : ", avg)
                     
    

def localSearch():
    "Uses distance_w(e1,e2)"
    w = 0.0
    

def greedyRelief():
    w = 0.0
    #For each element at trainngData 
    #    find nearest enemy and friend
    #    Then w = w+d(element,enemy)
    #    Then w = w-d(element,friend)
    #Then, if w_i < 0 => w_i = 0
    #Then, normalize w"

import scipy
import math
import numpy as np
from scipy.io import arff



def loadDataSet( str ) :
    "Returns a record array with the data"
    return scipy.io.arff.loadarff(str)
            

def distance (e1,e2) :
    "Euclidean distance between 2 elements"
    l = len(e1)
    sum = 0.0
    for i in range ( 0,l-1 ):
        sum += pow(e1[i]-e2[i],2)
    return math.sqrt(sum)

def distance_w (e1,e2):
    "Euclidean distance between 2 elements. However, the elements which weight is less than 0.2, dont count"
    l = len(e1)
    sum = 0.0
    for i in range (0,l-1):
        if (e1[i] >= 0.2 and e2[i] >= 0.2):
            sum += pow(e1[i]-e2[i],2)
    return math.sqrt(sum)
        

def nearestNeighbour (element , neighbours ) :
    "Searches for the nearest neighbour in a list"
    cmin = neighbours[0][-1]
    print("Cmin %s" % (cmin))
    dmin = distance(element,neighbours[0])
    print("Distance %s" % (dmin))
    num = len(neighbours)
    for i in range (1,num):
        d = distance(element,neighbours[i])
        if(d < dmin):
            cmin = neighbours[i][-1]
            dmin = d
    return cmin


def tasa_clase () :
    "Its the percentage of elements which I've classified well"
    "TODO:"
    return 100


def tasa_red (element) :
    "Its the percentage of elements which weight is less than 0.2"
    num = 0
    tot = len(element)
    for i in range (tot-1):
        if(float(element[i]) < float(0.2)):
            num += 1
               
    ret = (float(num)/float(tot))*100
    print("Tasa red: %s" % (ret))
    return ret 
        


def main ( ) :
    n1 = 'Datasets/ionosphere.arff'
    d1,m1 = loadDataSet(n1)
    normalizeData(d1)
    #nearestNeighbour(d1[1],d1)
    #print(tasa_red(d1[1]))
	

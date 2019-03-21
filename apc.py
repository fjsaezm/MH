import scipy
import sys
import math
from prepareData import *
from statistics import *
from algorithms import *
alpha = 0.5


def main ( ) :
    n1 = 'Datasets/ionosphere.arff'
    "d1 is the data, m1 is the metadata"
    d1,m1 = loadDataSet(n1)
    d1 = normalizeData(d1)
    train,test = splitData(d1,m1)
    #print("Train : ", len(d1[test[0]]))

    rights = onenn(d1,m1,train,test)
	

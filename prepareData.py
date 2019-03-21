import scipy
import sys
import math
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def loadDataSet( str ) :
    "Returns a record array with the data"
    data_struct = loadarff(str)[0]
    data_labels = data_struct['class']
    data = rfn.drop_fields(data_struct, 'class').view(
        np.float64).reshape(data_struct.shape+(-1,))

    return data, data_labels

def splitData(data,labels):
    train = []
    test = []
    skf = StratifiedKFold(n_splits = 5)
    for train_index , test_index in skf.split(data,labels):
        train.append(train_index)
        test.append(test_index)

    return train,test
        


def normalizeData (data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)

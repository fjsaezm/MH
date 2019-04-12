import scipy
import sys
import math
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


def load_arff(filename):
    data_struct = loadarff(filename)[0]
    # FIXME: field may not be named 'class'
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
    return MinMaxScaler().fit_transform(data)

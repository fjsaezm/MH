from prepareData import *
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

n1 = 'Datasets/colposcopy.arff'
n2 = 'Datasets/ionosphere.arff'
n3 = 'Datasets/texture.arff'
"d1 is the data, m1 is the metadata"
d1,c1 = load_arff(n1)
d2,c2 = load_arff(n2)
d3,c3 = load_arff(n3)
d1 = normalizeData(d1)
d2 = normalizeData(d2)
d3 = normalizeData(d3)

"Results. Rows: partition-i. Cols: T-clas,T-red,Agr,T"

"1-NN"
resCol1NN = [[0 for x in range(4)] for y in range (5)]
resIon1NN = [[0 for x in range(4)] for y in range (5)]
resText1NN = [[0 for x in range(4)] for y in range (5)]

"GreedyRelief"

resColGR = [[0 for x in range(4)] for y in range (5)]
resIonGR = [[0 for x in range(4)] for y in range (5)]
resTextGR = [[0 for x in range(4)] for y in range (5)]

"Local search"

resColLS = [[0 for x in range(4)] for y in range (5)]
resIonLS = [[0 for x in range(4)] for y in range (5)]
resTextLS = [[0 for x in range(4)] for y in range (5)]

"AGG-BLX"

resColAGGBLX = [[0 for x in range(4)] for y in range (5)]
resIonAGGBLX = [[0 for x in range(4)] for y in range (5)]
resTextAGGBLX = [[0 for x in range(4)] for y in range (5)]

"AGE-BLX"

resColAGEBLX = [[0 for x in range(4)] for y in range (5)]
resIonAGEBLX = [[0 for x in range(4)] for y in range (5)]
resTextAGEBLX = [[0 for x in range(4)] for y in range (5)]

"AGG-AC"

resColAGGAC = [[0 for x in range(4)] for y in range (5)]
resIonAGGAC = [[0 for x in range(4)] for y in range (5)]
resTextAGGAC = [[0 for x in range(4)] for y in range (5)]

"AGE-AC"

resColAGEAC = [[0 for x in range(4)] for y in range (5)]
resIonAGEAC = [[0 for x in range(4)] for y in range (5)]
resTextAGEAC = [[0 for x in range(4)] for y in range (5)]

"AM(10,1.0)"

resColAM1 = [[0 for x in range(4)] for y in range (5)]
resIonAM1 = [[0 for x in range(4)] for y in range (5)]
resTextAM1 = [[0 for x in range(4)] for y in range (5)]

"AM(10,0.1)"

resColAM2 = [[0 for x in range(4)] for y in range (5)]
resIonAM2 = [[0 for x in range(4)] for y in range (5)]
resTextAM2 = [[0 for x in range(4)] for y in range (5)]

"AM(10,0.1*mej)"

resColAM3 = [[0 for x in range(4)] for y in range (5)]
resIonAM3 = [[0 for x in range(4)] for y in range (5)]
resTextAM3 = [[0 for x in range(4)] for y in range (5)]

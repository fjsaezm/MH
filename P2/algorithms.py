
import numpy as np
import operator
from scipy.spatial.distance import pdist,squareform
from sklearn.model_selection import LeaveOneOut
from numpy.random import normal
from sklearn.metrics import accuracy_score
from random import randint

from sklearn.neighbors import KNeighborsClassifier
import time


BLX_operator   = 1
ARI_operator   = 2
max_iterations = 15000
sizeAGG = 30
alpha = 0.5
pcross= 0.7
pmut  = 0.001
np.random.seed(2019)

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
    fmax   = tasa_clas(w,trainD,trainC)

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


#Alpha is 0.3
def BLX(c1,c2):
    cmax  = max([max(c1),max(c2)])
    cmin  = min([min(c1),min(c2)])
    l     = cmax - cmin
    #interval is [a,b]
    a     = cmin - l*0.3
    b     = cmax + l*0.3

    H1 = np.random.uniform(a,b,len(c1))
    H2 = np.random.uniform(a,b,len(c2))

    return H1,H2

def arithmeticCross(c1,c2):
    return (numpy.array(c1)+numpy.array(c2))/2

#def eval

def selection(population,fitness):
    i1 = randint(0,len(population)-1)
    i2 = randint(0,len(population)-1)
    if fitness[i1] > fitness[i2]:
        return population[i1]
    else:
        return population[i2]
    
    
def goGenetic(data,classes,trainIndex,testIndex,cross_operator):
    total_genes = sizeAGG*data.shape[1]
    population = []
    fitness    = []
    #Generation of population
    for i in range(sizeAGG):
        population.append(np.random.uniform(0.0,1.0,len(data[0])))
        fitness.append(tasa_clas(population[i],data[trainIndex],classes[trainIndex]))

    generation = 1
    #first sizeAGG iterations
    it   = sizeAGG
    #Number of chromosomes to be crossed
    to_be_crossed = int(pcross* (len(population)/2))

    #stop criterion: max_iteration evaluations
    while it < max_iterations:

        new_population = []
        bestParentIndex = np.argmax(fitness)

        #Selection operator
        for i in range (len(population)):
            new_population.append(selection(population,fitness))

        
        #Cross operator
        for k in range (to_be_crossed*2):
            if cross_operator == BLX_operator:
                h1,h2 = BLX(new_population[k],new_population[k+1])
                new_population[k] = h1
                new_population[k+1] = h2
                k +=1
            else:
                new_population[i] = arithmeticCross(new_population[i],new_population[2*sizeAGG - i])


        #Number of chromosomes to mutate. Minimum is 1
        to_mutate = max(int(pmut*len(population)),1)
        muted_population = new_population
        #We use possibilities to avoid repetition of value
        possibilities = []
        for k in range (len(population)):
            possibilities.append(k)  
        #Mutation operator
        for k in range (to_mutate):
            indexPossib = np.random.randint(0,len(possibilities))
            indexChromosome = possibilities[indexPossib]
            indexGen        = np.random.randint(0,len(population[0]))
            muted_population[indexChromosome] = mov(muted_population[indexChromosome],indexGen)
            possibilities.pop(indexPossib)
            
    
        new_population = np.copy(muted_population)
        new_fitness    = []

       
        for w in new_population:
            new_fitness.append(tasa_clas(w,data[trainIndex],classes[trainIndex]))
            it += 1
      
        print(it)

        #Elitism
        newBestIndex = np.argmax(new_fitness)
        if new_fitness[newBestIndex] < fitness[bestParentIndex]:
            #mistake is here in delete
            new_population = np.delete(new_population,newBestIndex,axis = 0)
            new_fitness    = np.delete(new_fitness,newBestIndex)
            new_population = np.vstack((new_population,population[bestParentIndex]))

            new_fitness    = np.append(new_fitness,fitness[bestParentIndex])
                                  
        population = np.copy(new_population)
        fitness    = np.copy(new_fitness)
        generation += 1


    finalW = fitness.index(max(fitness))
    trainD = np.copy(data[trainIndex])
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)*100
    t_red_ret   = tasa_red(finalW)
    

    #need to return the tasa_class or the number of generations!
    return t_clas_ret, t_red_ret

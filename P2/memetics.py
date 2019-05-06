from algorithms import *
from genetics import *


sizeAM = 10
iters_softBL_AM = 10

def low_localSearch(data,classes,trainIndex,chromosome):
    bestf = chromosome.fitness
    n = len(chromosome.w)
    it = 0

    while(it < 2*n):
        k = np.random.choice(range(n))
        mutChrom = chromosome
        mutChrom.w = mov(mutChrom.w,k)
        mutChrom.fitness = tasa_clas(mutChrom.w,data[trainIndex],classes[trainIndex])
        it += 1
        if mutChrom.fitness > bestf:
            chromosome = mutChrom
            bestf = mutChrom.fitness

    return it,chromosome



# General AM
def AM(data,classes,trainIndex,testIndex,cross_operator,typeMemetic):
    population = []

    #Generation of population
    for i in range (sizeAM):
        population.append(Chromosome(data[trainIndex],classes[trainIndex]))

    generation = 1
    #first iterations
    it = sizeAM
    #number of chromosomes to be crossed
    to_be_crossed = int(pcross*(len(population)/2))
     #Number of chromosomes to mutate. Minimum is 1
    to_mutate = max(int(pmut*len(population)),1)

    #stop criterion: max_iteration evaluations
    while it < 130:

        s = time.time()
        new_population = []
        
        bpIndex = 0
        for i in range (len(population)):
            if population[i].fitness > population[bpIndex].fitness :
                bpIndex = i      
        
        #Selection operator
        for i in range (len(population)):
            new_population.append(selection(population))

        #Cross operator
        for k in range (to_be_crossed*2):
            i1 = randint(0,len(new_population)-1)
            i2 = randint(0,len(new_population)-1)
            h1,h2 = cross_operator(new_population[i1],new_population[i2],data[trainIndex],classes[trainIndex])
            new_population[i1] = h1
            new_population[i2] = h2
            k += 1
            it +=2
 
        muted_population = new_population
        #We use possibilities to avoid repetition of value
        possibilities = []
        for k in range (len(population)):
            possibilities.append(k)
            
        #Mutation operator
        for k in range (to_mutate):
            indexPossib = np.random.randint(0,len(possibilities)-1)
            indexChromosome = possibilities[indexPossib]
            indexGen        = np.random.randint(0,len(population[0].w)-1)
            new_C = Chromosome(data[trainIndex],classes[trainIndex],mov(muted_population[indexChromosome].w,indexGen))
            muted_population[indexChromosome] = new_C
            possibilities.pop(indexPossib)
            it +=1
            
    
        new_population = np.copy(muted_population)


        #Find new best
        currentBestIndex = 0
        for i in range (len(new_population)):
            if new_population[i].fitness > new_population[bpIndex].fitness :
                currentBestIndex = i
        #Elitism       
        if new_population[currentBestIndex].fitness < population[bpIndex].fitness:
            newWorst = 0
            for i in range (len(new_population)):
                if new_population[i].fitness < new_population[newWorst].fitness:
                    newWorst = i
            new_population = np.delete(new_population,newWorst)
            new_population = np.append(new_population, population[bpIndex])
           
                                  
        population = np.copy(new_population)
        
        # Check if we have to apply soft local search
        if (generation % iters_softBL_AM) == 0:
            s,population = typeMemetic(population,data,classes,trainIndex)
            it +=s
        
        generation += 1


    finalWIndex = 0
    for i in range (len(population)):
        if population[i].fitness > population[finalWIndex].fitness:
            finalWIndex = i
    finalW = population[finalWIndex].w
    trainD = np.copy(data[trainIndex])*finalW
    trainC = np.copy(classes[trainIndex])
    testD  = np.copy(data[testIndex])*finalW
    testC  = np.copy(classes[testIndex])

    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(trainD,trainC)

    predictions = classifier.predict(testD)
    t_clas_ret  = accuracy_score(testC,predictions)*100
    t_red_ret   = tasa_red(finalW)
    

    #need to return the tasa_class or the number of generations!
    return t_clas_ret, t_red_ret

#Function for AM-(10,1.0)
def am1(population,data,classes,trainIndex):
    it = 0
    new_population = []
    for c in population:
        s,newC = low_localSearch(data,classes,trainIndex,c)
        it += s
        new_population.append(newC)

    return it,new_population

#Function for AM-(10,0.1)
#Since in our case its only 1 gen to modify, will select a random one
def am2(population,data,classes,trainIndex):
    k = np.random.choice(range(sizeAM-1))
    it,population[k] = low_localSearch(data,classes,trainIndex,population[k])
    return it, population

#Function for AM-(10,0.1mej)
def am3(population,data,classes,trainIndex):
    best = 0
    for i in range(len(population)):
        if population[i].fitness > population[best].fitness:
            best = i
    it,population[best] = low_localSearch(data,classes,trainIndex,population[best])

    return it,population

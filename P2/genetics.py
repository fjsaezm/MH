from algorithms import *


sizeAGG = 30
pcross= 0.7
pmut  = 0.001
gen_max_it = 200
np.random.seed(2019)


class Chromosome:
    def __init__(self,data,classes,weights = []):
        if len(weights) == 0:
            self.w = np.random.uniform(0.0,1.0,len(data[0]))
        else:
            self.w = np.copy(weights)
            
        self.fitness = 0.5*(tasa_clas(self.w,data,classes) + tasa_red(self.w))

        
#Alpha is 0.3
def BLX(c1,c2,data,classes):
    cmax  = max([max(c1.w),max(c2.w)])
    cmin  = min([min(c1.w),min(c2.w)])
    l     = cmax - cmin
    #interval is [a,b]
    a     = cmin - l*0.3
    b     = cmax + l*0.3

    H1 = np.random.uniform(a,b,len(c1.w))
    H2 = np.random.uniform(a,b,len(c2.w))

    return Chromosome(data,classes,H1),Chromosome(data,classes,H2)

def arithmeticCross(c1,c2,data,classes):
    return Chromosome(data,classes,(np.array(c1.w)+np.array(c2.w))/2),Chromosome(data,classes,(np.array(c1.w)+np.array(c2.w))/2)
    

# Returns index to best of selection
def selection(population):
    i1 = randint(0,len(population)-1)
    i2 = randint(0,len(population)-1)
    if population[i1].fitness > population[i2].fitness:
        return population[i1]
    else:
        return population[i2]



def AGG(data,classes,trainIndex,testIndex,cross_operator):
    population = []

    #Generation of population
    for i in range(sizeAGG):
        population.append(Chromosome(data[trainIndex],classes[trainIndex]))

    generation = 1
    #first sizeAGG iterations
    it   = sizeAGG
    #Number of chromosomes to be crossed
    to_be_crossed = int(pcross* (len(population)/2))
    #Number of chromosomes to mutate. Minimum is 1
    to_mutate = max(int(pmut*len(population)),1)

    #stop criterion: max_iteration evaluations
    while it < gen_max_it:

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


#Genetic Stationary algorithm
def AGE(data,classes,trainIndex,testIndex,cross_operator) :
    population = []

    #Generation of population
    for i in range(sizeAGG):
        population.append(Chromosome(data[trainIndex],classes[trainIndex]))

    generation = 1
    #first sizeAGG iterations
    it   = sizeAGG
    #Number of chromosomes to mutate. Minimum is 1
    to_mutate = max(int(pmut*len(population)),1)

    #stop criterion: max_iteration evaluations
    while it < gen_max_it:

        s = time.time()
        new_population = []

        #Best of population
        bpIndex = 0
        for i in range (len(population)):
            if population[i].fitness > population[bpIndex].fitness :
                bpIndex = i
                
        #The 2 parents of the AGE
        new_parents = [selection(population),selection(population)]

        #Cross operator
        new_parents[0],new_parents[1] = cross_operator(new_parents[0],new_parents[1],data[trainIndex],classes[trainIndex])
        it +=2
        possibilities  = [0,1]
        #Mutation operator
        for k in range (to_mutate):
            
            indexPossib = np.random.randint(0,len(possibilities)-1)
            indexChromosome = possibilities[indexPossib]
            indexGen        = np.random.randint(0,len(population[0].w)-1)
            new_C = Chromosome(data[trainIndex],classes[trainIndex],mov(new_parents[indexChromosome].w,indexGen))
            new_parents[indexPossib] = new_C
            possibilities.pop(indexPossib)
            it +=1
    

        #Replacement
        #Process: append both, sort, delete the 2 worst.
        population = np.append(population,new_parents[0])
        population = np.append(population,new_parents[1])
        population = sorted(population, key = lambda x : x.fitness)
        population = np.delete(population,len(population)-1)
        population = np.delete(population,len(population)-1)
        population = shuffle(population,random_state = 0)
            
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

import Reporter
import numpy as np
import random
import Reporter
from numba import jit

# Class name represents student number.
class TSPSolver:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.nLocation = 0
        self.lambdaa = 100
        self.mu = 30
        self.k = 3
        self.maxnIter = 10
        self.nIter = 0
        self.meanObj = 0
        self.pi = 0
        self.prevBest = 0
        self.bestSolution = None

    """ Given a tour (or candidate solutions), return its distance 
    between itself and all tours in the population. Distance is defined as the product of Hamming distance and 
     difference in objective value """
    def distance(self, tour1, population, fit_pop = None):
        dist = np.count_nonzero(tour1 != population, axis = 1)
        if fit_pop is None:
            return dist
        else:
            fit_tour = self.objectiveVal(tour1)
            delta = np.abs(fit_pop - fit_tour)
            return dist*delta

    """ Initialize the parameters that depend on nLocation """
    def initParam(self):
        self.nLocation = self.distanceM.shape[1]
        self.pi = 1 / np.log(self.nLocation ^ 3)

    def objectiveValOG(self, x):
        distanceMat = self.distanceM
        valTotal =  self.objectiveValOG2(distanceMat, x)
        return valTotal

    """ Objective value that is suitable for any vector x with variable length """
    @staticmethod
    @jit(nopython = True)
    def objectiveValOG2(distanceMat, x):
        n = len(x)
        valTotal = 0
        for i in range(n - 1):
            temp = distanceMat[x[i], x[i + 1]]
            valTotal += temp
        valTotal += distanceMat[x[n - 1], x[0]]
        return valTotal

    """ Compute the MODIFIED objective function
    (based on fitness sharing) at the vector of route """
    def objectiveVal(self, x, population = None, fitPop = None):
        fval = self.objectiveValOG(x)
        if population is None:
            return fval
        alpha = 1
        onePlusBeta = 0
        if fitPop is None:
            ds = self.distance(x, population)
            sigma = self.nLocation * 0.12
        else:
            ds = self.distance(x, population, fitPop)
            sigma = self.nLocation * fval * 0.12
        near = ds[ds <= sigma]
        np.seterr(divide='ignore', invalid='ignore')
        onePlusBeta = np.sum(1-np.power(near/sigma, alpha))
        if ds[ds <= sigma].size == 0: onePlusBeta = 1
        modObjVal = fval * onePlusBeta**np.sign(fval)
        return modObjVal

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        self.distanceM = distanceMatrix

        # Initialize the population
        self.initParam()
        stopCount = 0
        np.seterr(divide='ignore', invalid='ignore')
        maxCount = np.rint(30**2/np.sqrt(self.nLocation))
        population = self.initialize(4)
        # population = self.initializeShuffle(self.nLocation)

        while self.nIter <= self.maxnIter:
            selected = self.selection(population, self.k)
            offspring = self.crossover(selected)
            self.mutation(offspring)
            joinedPopulation = np.vstack((offspring, population))
            population = self.shareElimination(joinedPopulation, self.lambdaa)

            # Progress
            meanObjective = self.meanObj
            bestSolution = min(population, key = lambda x: self.objectiveVal(x))
            bestObjective = self.objectiveVal(bestSolution)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution) # Uncomment for Reporter.
            if timeLeft < 0 or stopCount == maxCount:
                break
            elif bestObjective == self.prevBest:
                stopCount += 1
            else:
                self.maxnIter = max(100, round(self.reporter.allowedTime/((self.reporter.allowedTime - timeLeft)/self.nIter)))
                stopCount = 0
                self.prevBest = bestObjective
        return 0

    """ Initialize the population by using nearest neighbors (greedy permutation method):
         https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8473531 """
    def initialize(self, neighborSize):
        cityInd = 0
        n = 0
        population = np.empty([self.nLocation * neighborSize, self.nLocation], dtype = int)
        popVals = np.empty(self.nLocation * neighborSize)

        while cityInd < self.nLocation:
            individual = np.array(range(2))
            individual[0] = cityInd
            neighbors = self.getNN(cityInd, neighborSize)
            for neighbor in neighbors:
                fitness = self.distanceM[cityInd, neighbor]
                individual[1] = neighbor
                (restCities, tourVal) = self.NN(individual, fitness)
                popVals[n] = tourVal
                population[n, :] = restCities
                n += 1
            cityInd += 1

        perm = np.argsort(popVals)
        population = population[perm[0:self.lambdaa], :]
        # ind = np.random.choice(len(perm), min(2 * self.lambdaa, self.nLocation * neighborSize), replace = False)
        # population = population[perm[ind]]
        self.bestSolution = min(population, key = lambda x: self.objectiveVal(x))
        return population

    """ Perform k-tournament selection (without replacement) and fitness sharing to select pairs of parents """
    def selection(self, population, k):
        selected = np.empty([self.mu, self.nLocation], dtype = int)
        np.random.shuffle(population)
        j = 0
        for i in range(self.mu):
            candidates = population[j:j+k, :]
            champion = min(candidates, key = lambda x, population = population: self.objectiveVal(x, population))
            selected[i,:] = champion[:] # should use deep copy? idk
            j += k
        self.nIter = self.nIter + 1
        return selected

    """ Perform best-order crossover (BOX) """
    def crossover(self, selected):
        nSelected = selected.shape[0]
        offsprings = np.empty([nSelected, self.nLocation], dtype = int)
        def getChild(self, par1, par2):
            prev = 0
            child = np.zeros(par1.shape, dtype=int)
            nCutPoints = np.random.randint(low=2, high=self.nLocation, size=1)
            lenCut = np.random.randint(low=1, high=(self.nLocation / (nCutPoints + 1)) + 1, size=nCutPoints - 1)
            startCut = np.random.randint(low=1, high=self.nLocation - sum(lenCut), size=1)
            qs = np.concatenate((startCut, lenCut))
            qs = np.cumsum(qs)
            qs = np.append(qs, self.nLocation)
            scenarios = np.random.randint(-3, 0, size = nCutPoints + 1)

            posPar1 = np.argsort(par1)
            posPar2 = np.argsort(par2)
            posBest = np.argsort(self.bestSolution)
            for i in range(nCutPoints[0] + 1):
                if scenarios[i] == -1:
                    child[prev:qs[i]] = par1[prev:qs[i]]
                elif scenarios[i] == -2:
                    ind = np.arange(self.nLocation)[(posPar1 < qs[i]) & (posPar1 >= prev)]
                    indPar2 = posPar2[ind]
                    indPar2 = np.sort(indPar2)
                    child[prev:qs[i]] = par2[indPar2]
                else:
                    ind = np.arange(self.nLocation)[(posPar1 < qs[i]) & (posPar1 >= prev)]
                    indBest = posBest[ind]
                    indBest = np.sort(indBest)
                    child[prev:qs[i]] = self.bestSolution[indBest]
                prev = qs[i]
            return child
        for ii in range(0, self.mu, 2):
            offsprings[ii] = getChild(self, selected[ii], selected[ii + 1])
            offsprings[ii + 1] = getChild(self, selected[ii + 1], selected[ii])
            # offsprings[ii] = self.localOpt(offsprings[ii])
            # offsprings[ii + 1] = self.localOpt(offsprings[ii + 1])
        return offsprings

    """ Perform mutation with inversion & scramble mutation """
    def mutation(self, offspring, alpha = None):
        stage = self.nIter/self.maxnIter
        def mutateIndividual(child, low, high):
            if child.size == 0: return child
            if self.nIter <= self.maxnIter * 0.1:
                child[:, low:high + 1] = np.fliplr(child[:, low:high + 1])
            else: # scramble mutation
                child[:, low:high + 1] = np.random.permutation(child[:, low:high + 1].T).T
            return child
        if alpha == None:
            alpha = 1 - 0.9*stage
        indMutate = np.where(np.random.rand(offspring.shape[0]) <= alpha)

        interval = round((1 - 0.8*stage)*self.nLocation/2)
        position = random.sample(range(self.nLocation - interval), 1)[0]
        offspring[indMutate] = mutateIndividual(offspring[indMutate], position, position + interval)
        return offspring

    """ Split the joined population into two parts with the size ratio pi/(1-pi).
            The first part contains only unique values of individuals that won RR tournament.
            The second part uses fitness sharing mechanism to select the rest """
    def shareElimination(self, joinedPopulation, keep):
        # pi = 1/np.log(self.nLocation^3)
        nJoined = joinedPopulation.shape[0]
        nFirstRound = int(round(self.pi * keep))
        finalVals = np.apply_along_axis(self.objectiveVal, 1, joinedPopulation)
        perm = np.argsort(finalVals)

        # Round Robin here
        ind = np.arange(nJoined)
        resultRR = self.rrTournament(k = 10, tourInd = ind, popVals = finalVals)
        permRR = np.argsort(resultRR)[::-1]
        suvivorsFirstRound = joinedPopulation[permRR[0:nFirstRound], :]
        suvivorsFirstRound = np.apply_along_axis(self.localOpt, 1, suvivorsFirstRound)
        suvivorsFirstVals = np.apply_along_axis(self.objectiveVal, 1, suvivorsFirstRound)

        nSecondRound = keep - nFirstRound
        suvivorsSecondRound = np.empty([nSecondRound, self.nLocation])
        indLeft = np.setdiff1d(np.array(range(nJoined)), permRR[0:nFirstRound]) # should be permRR instead
        remain = joinedPopulation[indLeft]
        remainVals = np.apply_along_axis(self.objectiveVal, 1, remain, population=suvivorsFirstRound,
                                         fitPop=suvivorsFirstVals)

        permRemain = np.argsort(remainVals)
        suvivorsSecondRound = remain[permRemain[0: nSecondRound], :]
        suvivorsSecondVals = finalVals[indLeft[permRemain[0:nSecondRound]]]
        suvivors = np.vstack((suvivorsFirstRound, suvivorsSecondRound))

        allVals = np.concatenate((suvivorsFirstVals, suvivorsSecondVals))
        self.meanObj = np.mean(allVals)
        return suvivors

    """ Given the tour, output the number of wins for Round Robin tournament """
    @staticmethod
    @jit(nopython=True)
    def rrTournament(k, tourInd, popVals):
        rrResults = np.empty(tourInd.shape[0], dtype = np.int32)
        nComp = popVals.shape[0]
        ind = np.arange(nComp)

        for i in tourInd:
            tourVal = popVals[i]
            np.random.shuffle(ind)
            participantsVal = popVals[ind[0:k]]
            win = np.sum(tourVal <= participantsVal, dtype = np.int32)
            rrResults[i] = win
        return rrResults

    """ Return the index of nearest neighbors """
    def getNN(self, cityInd, neighborSize):
        nnVals = self.distanceM[cityInd, :]
        perm = np.argsort(nnVals)
        nearestNeighbors = perm[1:neighborSize + 1]
        return nearestNeighbors

    """ Heuristic NN: Returns the path and cost of the found solution """
    def NN(self, startingPath, startingFit):
        totalCost = startingFit
        N = self.nLocation
        path = np.empty(N, dtype = int)
        path[0] = startingPath[0]
        path[1] = startingPath[1]

        mask = np.ones(N, dtype=bool)
        mask[startingPath] = False
        for i in range(N - 2):
            last = path[i + 1]
            next_ind = np.argmin(self.distanceM[last, :][mask])
            next_loc = np.arange(N)[mask][next_ind]
            path[i + startingPath.size] = next_loc
            mask[next_loc] = False
            totalCost += self.distanceM[last, next_loc]
        totalCost += self.distanceM[path[-1], path[0]]
        return path, totalCost

    """Given a candidate solution, generate a neighbor that has a
    fitness value greater or equal than the given candidate's fitness value"""
    def localOpt(self, x):
        distanceM = self.distanceM
        neighbors = self.localOpt2(distanceM, x)
        champion = min(neighbors, key=lambda x: self.objectiveVal(x))
        if (self.objectiveVal(champion) > self.objectiveVal(x)): return x
        return champion

    @staticmethod
    @jit(nopython = True)
    def localOpt2(distanceM, x): # could add n neighbors!
        neighbors = []
        for i in range(x.size):
            iNext = i + 1
            if iNext >= x.size:
                iNext = 0
            for j in range(x.size):
                jNext = j + 1
                if jNext >= x.size:
                    jNext = 0
                if ((x[jNext] != x[iNext] or x[i] != x[j]) and
                    distanceM[x[i], x[iNext]] + distanceM[x[j], x[jNext]] > distanceM[x[jNext], x[iNext]] + distanceM[x[j], x[i]]):
                        candidate = np.copy(x)
                        if i < jNext: candidate[i:jNext + 1] = np.flipud(candidate[i:jNext + 1]) # reverse operator, including i and j + 1
                        else: candidate[jNext:i + 1] = np.flipud(candidate[jNext:i+1])
                        neighbors.append(candidate)
        return neighbors

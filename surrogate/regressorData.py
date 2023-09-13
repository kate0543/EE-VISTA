from enum import Enum

import numpy as np

from vlga.chromosome import Chromosome


class SelectionType(str, Enum):
    ROULETTE = "roulette"
    RANDOM = "random"


class regData():
    def __init__(self, chromosome_chunk_size, max_n_chunk=10,  # variable-length params  # normal GA params
                 random_state=1, log_detail=True):
        """
        chromosome:
            length = n_chunk * chromosome_chunk_size
            |xxxx xxxx xxxx| -> length = 12, n_chunk = 3, chunk_size = 4
        """

        self.log_detail = log_detail
        self.max_n_chunk = max_n_chunk
        self.chromosome_chunk_size = chromosome_chunk_size
        np.random.seed(random_state)
        self.population = []

    def createTrainData(self, problem):
        xdata = []
        ydata = []
        self.initialize_population(problem)
        for p in self.population:
            xdata.append(p.config.tolist())
            ydata.append(p.fitness)
        return xdata, ydata

    def saveTrainData(self, xdata, ydata):
        with open('datasets/reg_xdata.dat', 'w') as file:
            for x in xdata:
                s = str(x)
                # s.replace('[','')
                # s.replace(']','')
                file.write(s[1:-1])
                file.write('\n')
        with open('datasets/reg_ydata.dat', 'w') as file:
            for y in ydata:
                file.write(str(y))
                file.write('\n')
        exit()

    def initialize_population(self, problem, pop_size):
        print('Initialize population')
        self.population = []
        for i_population in range(pop_size):
            n_chunk = np.random.randint(1, self.max_n_chunk + 1)  # interval [1, max_n_chunk]
            chromosome_size = n_chunk * self.chromosome_chunk_size

            config = np.random.randint(2, size=0)

            for i_chunk in range(n_chunk):
                while True:
                    new_chunk = np.random.randint(2, size=self.chromosome_chunk_size)
                    if np.sum(new_chunk) > 0:
                        config = np.concatenate((config, new_chunk))
                        break

            fitness = problem.fitness(config)

            chromosome = Chromosome(config, fitness)

            self.population.append(chromosome)
            print(i_population, ',', fitness)
        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
        return self.population
        print('Done initializing')

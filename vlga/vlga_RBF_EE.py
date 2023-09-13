import copy
import datetime
from enum import Enum

import numpy as np

from surrogate.smt_RBF_EE import smtWrapper
# from surrogate.smt_RBF_LSTM import smtWrapper

from vlga.chromosome import Chromosome
from vlga.ga_utils import crossover, mutate
from vlga.result_collector import ResultCollector


class SelectionType(str, Enum):
    ROULETTE = "roulette"
    RANDOM = "random"
    TOURNAMENT = "tournament"


class VLGA:

    def __init__(self, chromosome_chunk_size, max_n_chunk=10, n_lstm=10, # variable-length params
                 n_generations=50, prob_crossover=0.9, prob_mutation=0.1, n_population=100,  # normal GA params
                 smt_train_iter=1, smt_estimate_iter=10,
                 selection_type=SelectionType.ROULETTE, random_state=1, log_detail=True):
        """
        chromosome:
            length = n_chunk * chromosome_chunk_size
            |xxxx xxxx xxxx| -> length = 12, n_chunk = 3, chunk_size = 4
        """

        self.log_detail = log_detail
        self.selection_type = selection_type
        self.max_n_chunk = max_n_chunk
        self.chromosome_chunk_size = chromosome_chunk_size
        self.n_population = n_population
        self.prob_mutation = prob_mutation
        self.prob_crossover = prob_crossover
        self.n_generations = n_generations

        np.random.seed(random_state)
        self.population = []
        self.result_collector = ResultCollector(log_detail)

        """
        smt: a list of regressors
        smt_estimate_iter: estimate number of iterations 
        smt_train_iter: train number of iterations
        surrogate: flag use to switch between estimate and train for surrogate model 
        """
        self.n_lstm=n_lstm
        self.smt = smtWrapper(self.n_lstm,self.chromosome_chunk_size)
        self.smt_estimate_iter = smt_estimate_iter
        self.smt_train_iter = smt_train_iter
        self.surrogate = False

    def dummy_solve(self, problem):
        np.random.seed(1)
        n_chunk = np.random.randint(1, self.max_n_chunk)
        chromosome_size = n_chunk * self.chromosome_chunk_size

        random_candidate = np.random.randint(2, size=chromosome_size)
        return random_candidate, problem.fitness(random_candidate)

    def solve(self, problem):

        self.initialize_population(problem)
        self.smt.initTrainData(self.population)
        self.smt.fit()
        self.smt.train()

        archive = copy.deepcopy(self.population)
        smt_estimate_count = 0
        smt_train_count = 0
        for i_generation in range(self.n_generations):

            # use real fitness in last iteration
            if i_generation == self.n_generations - 1:
                self.surrogate = False
                smt_train_count +=1
                print('use real fitness in last iteration')
            else:
                # count estimate/train number of iterations
                if self.surrogate:
                    smt_estimate_count += 1
                else:
                    smt_train_count += 1
                # switch surrogate status
                if smt_train_count == self.smt_train_iter:
                    # retrain regressors model when have enough data
                    # reset train count
                    self.surrogate = not self.surrogate
                    print(smt_train_count, ',training surrogate model')
                    self.smt.updateModel()
                    smt_train_count = 0
                if smt_estimate_count == self.smt_estimate_iter:
                    # reset estimate count
                    self.surrogate = not self.surrogate
                    print(smt_estimate_count, ',stop use surrogate model')
                    smt_estimate_count = 0


            print(datetime.datetime.now(), ' At generation {}'.format(i_generation))

            population_fitness = np.array([c.get_fitness() for c in self.population])

            population_mean = population_fitness.mean()
            population_best = population_fitness[0]
            self.result_collector.add_generation_average_fitness(population_mean)
            self.result_collector.add_generation_best_fitness(population_best)

            print('surrogate', self.surrogate, '\tsmt train', smt_train_count, '\tsmt estimate', smt_estimate_count)
            print('Current generation best: {}'.format(population_best))
            print('Current generation mean: {}'.format(population_mean))

            selection_probability = population_fitness / population_fitness.sum()

            while True:
                # 1. crossover process
                if self.selection_type == SelectionType.RANDOM:
                    id1, id2 = np.random.choice(self.n_population, 2, replace=False)
                elif self.selection_type == SelectionType.ROULETTE:
                    # print('  gen ', i_generation,
                    #       ' ,pop size=', len(self.population),
                    #       # ' ,sel list size=', len(selection_probability),
                    #       ' ,surrogate ', surrogate)
                    # print(population_fitness)
                    id1, id2 = np.random.choice(self.n_population, 2, replace=False,
                                                p=selection_probability)
                elif self.selection_type == SelectionType.TOURNAMENT:
                    id1, id2 = self.tournament_selection()
                r_crossover = np.random.rand()
                if r_crossover < self.prob_crossover:
                    new_chromosome1, new_chromosome2 = crossover(self.population[id1], self.population[id2],
                                                                 self.chromosome_chunk_size)

                    self.population.append(new_chromosome1)
                    self.population.append(new_chromosome2)

                    # 2. mutation process
                    # NOTE: new_chromosome1 & new_chromosome2 still refer to those in population
                    r_mutation = np.random.rand()
                    if r_mutation < self.prob_mutation:
                        # Apply mutation on the 2 new offsprings
                        mutate(new_chromosome1, self.chromosome_chunk_size)
                        mutate(new_chromosome2, self.chromosome_chunk_size)

                    # 3. recalculate fitness for the 2 new offsprings
                    config1 = new_chromosome1.get_config()
                    config2 = new_chromosome2.get_config()

                    # use surrogate model to predict fitness
                    if self.surrogate:
                        fitness1, fitness2 = self.smt.predict([config1, config2])
                    else:
                        fitness1 = problem.fitness(config1)
                        fitness2 = problem.fitness(config2)

                    new_chromosome1.set_fitness(fitness1)
                    new_chromosome2.set_fitness(fitness2)

                # print(len(self.population))
                if len(self.population) >= 2 * self.n_population:
                    break

            # 4. natural selection 2*L -> L
            self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
            del self.population[self.n_population:]
            if self.surrogate:
                # when surrogate, update surrogate model with selection strategy
                subpopulation = self.smt.selection(self.population, 'best')
                for chromosome in subpopulation:
                    fitness = problem.fitness(chromosome.config)
                    chromosome.set_fitness(fitness)
                    self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
                self.smt.updateData(subpopulation)
                self.smt.updateModel()
            else:
                # when not using surrogate, update surrogate database
                self.smt.updateData(self.population)

        best_candidates = []
        best_fitness = self.population[0].get_fitness()

        for i in range(len(self.population)):
            if self.population[i].get_fitness() == best_fitness:
                best_candidates.append(self.population[i])
            else:
                break

        return best_candidates

    # tournament selecton, tournament size =2
    def tournament_selection(self):
        index = []
        while len(index) < 2:
            id1, id2 = np.random.choice(self.n_population, 2, replace=False)
            if self.population[id1].get_fitness() > self.population[id2].get_fitness():
                if id1 not in index:
                    index.append(id1)
            else:
                if id2 not in index:
                    index.append(id2)
        return index[0], index[1]

    def initialize_population(self, problem):
        print('Initialize population')
        self.population = []
        for i_population in range(self.n_population):
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

        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)

        print('Done initializing')

    def get_result_collector(self):
        return self.result_collector

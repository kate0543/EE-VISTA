class Chromosome:
    def __init__(self, config, fitness):
        self.config = config
        self.fitness = fitness

    def get_config(self):
        return self.config

    def get_fitness(self):
        return self.fitness

    def set_config(self, config):
        self.config = config

    def set_fitness(self, fitness):
        self.fitness = fitness

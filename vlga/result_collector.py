class ResultCollector:

    def __init__(self, log_details):
        """ If not log_details then do nothing

        """
        self.active = log_details
        if not self.active:
            pass
        else:
            self.generations_average_fitness = []
            self.generations_best_fitness = []

    def add_generation_average_fitness(self, fitness):
        if not self.active:
            pass
        else:
            self.generations_average_fitness.append(fitness)

    def add_generation_best_fitness(self, fitness):
        if not self.active:
            pass
        else:
            self.generations_best_fitness.append(fitness)

    def get_generations_average_fitness(self):
        return self.generations_average_fitness

    def get_generations_best_fitness(self):
        return self.generations_best_fitness

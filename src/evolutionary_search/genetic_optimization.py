from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

'''
boundaries for hyperparameters:
----
layer 1: CNN [8 to 128] | RNN [8 to 128]
layer 2: CNN [0 to 128] | RNN [0 to 128]
layer 3: CNN [0 to 128] | -
layer 4: CNN [0 to 64]  | -
layer 5: CNN [0 to 32]  | -
solver: [Adam, RMSProps, SGD] as [0, 1, 2]
learning_rate: [1-e6 to 1-e1] as [1,2,3,4,5,6]
lr_decay: [constant, exponential] as [0,1]
callback(early stopping): [true, false] as [0,1]
expochs: [20 to 50]
----
CNN:
kernel_size: [2 to 32]
activation: [relu, tanh] as [0,1]
RNN:
layer_type: [lstm, gru] as [0,1]
'''

BOUNDS_CNN_LOW = [8,  -32, -32, -48, -64,     0,     1,  0,  0,  10,  2, 0]
BOUNDS_CNN_HIGH = [128,  128,  128,  64, 32, 2.999, 6.999,  1,  1,  50, 32, 1]
BOUNDS_RNN_LOW = [8,  -32, -32, -32, -32,     0,     1,  0,  0,  10,  0]
BOUNDS_RNN_HIGH = [128,  128,  0,   0,   0, 2.999, 6.999,  1,  1,  50,  1]


class GeneticSearch():
    # save config params to the class object
    def __init__(self, ga_config, model, dataset) -> None:

        self.nn_type = ga_config['NN_TYPE']

        self.population_size = ga_config['POPULATION_SIZE']
        self.max_generations = ga_config['MAX_GENERATIONS']
        self.p_crossover = ga_config['P_CROSSOVER']
        self.p_mutation = ga_config['P_MUTATION']
        self.hall_of_fame_size = ga_config['HALL_OF_FAME_SIZE']
        self.crowding_factor = ga_config['CROWDING_FACTOR']

        self.population = None
        self.hof = None

        self.BOUNDS_LOW = BOUNDS_CNN_LOW if self.nn_type == "CNN" else BOUNDS_RNN_LOW
        self.BOUNDS_HIGH = BOUNDS_CNN_HIGH if self.nn_type == "CNN" else BOUNDS_RNN_HIGH
        self.NUM_PARAMS = len(self.BOUNDS_HIGH)

        self.model = model
        self.dataset = dataset
        self.attributes = ()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

    def initial_setup(self):  # create strategy
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        for i in range(self.NUM_PARAMS):
            self.toolbox.register("attribute_" + str(i),
                                  random.uniform,
                                  self.BOUNDS_LOW[i],
                                  self.BOUNDS_HIGH[i])
        for i in range(self.NUM_PARAMS):
            self.attributes = self.attributes + \
                (self.toolbox.__getattribute__("attribute_" + str(i)),)

    def get_fitness(self, individ):
        return [self.model.model_generator(self.dataset, individ)]

    def create_population(self):  # create population
        # create the individual operator to fill up an Individual instance:
        self.toolbox.register("individualCreator",
                              tools.initCycle,
                              creator.Individual,
                              self.attributes,
                              n=1)

        # create the population operator to generate a list of individuals:
        self.toolbox.register("populationCreator",
                              tools.initRepeat,
                              list,
                              self.toolbox.individualCreator)
        self.toolbox.register("evaluate", self.get_fitness)

    def define_operators(self):

        # pick up the best available parents
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        # make crossover between 2 picked parents
        self.toolbox.register("mate",
                              tools.cxSimulatedBinaryBounded,
                              low=self.BOUNDS_LOW,
                              up=self.BOUNDS_HIGH,
                              eta=self.crowding_factor)
        # mutate
        self.toolbox.register("mutate",
                              tools.mutPolynomialBounded,
                              low=self.BOUNDS_LOW,
                              up=self.BOUNDS_HIGH,
                              eta=self.crowding_factor,
                              indpb=1.0/self.NUM_PARAMS)

    def print_solution(self):
        print("Best solution: \n",
              self.model.format_params(self.hof.items[0]),
              "\n Accuracy = ",
              self.model.model_generator(self.dataset, self.hof.items[0]))

    def ga_elitism(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                   halloffame=None, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is None:
            raise ValueError("halloffame is empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - hof_size)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population:
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def solver(self):
        # create initial population (generation 0):
        population = self.toolbox.populationCreator(n=self.population_size)

        # prepare the statistics object:
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("max", numpy.max)
        self.stats.register("avg", numpy.mean)

        # define the hall-of-fame object:
        self.hof = tools.HallOfFame(self.hall_of_fame_size)

        # perform the Genetic Algorithm flow with hof feature added:
        population, logbook = self.ga_elitism(population,
                                              self.toolbox,
                                              cxpb=self.p_crossover,
                                              mutpb=self.p_mutation,
                                              ngen=self.max_generations,
                                              stats=self.stats,
                                              halloffame=self.hof,
                                              verbose=True)
        self.print_solution()
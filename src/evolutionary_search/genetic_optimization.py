from deap import base
from deap import creator
from deap import tools

import random

class Genetic_Search():
    def __init__(self, ga_config, nn_type) -> None: #save config params to the class object
        self.layer_type = ga_config['layer_type']
        self.layers = ga_config['layers']
        self.units = ga_config['units']
        self.activation = ga_config['activation']
        self.learing_rates = ga_config['learing_rates']
        self.epochs = ga_config['epochs']
        self.nn_type = nn_type

        self.gen_len = ga_config['gen_len']
        self.population = ga_config['population']
        self.max_generations = ga_config['generations']
        self.p_crossover = ga_config['prob_crossover']
        self.p_mutation = ga_config['prob_mutation']
        self.fitness = None 
        self.best = None
        self.history = []
        
        self.toolbox = base.Toolbox()


    def initial_setup(self): #create strategy

        ...

    def create_population(self): # create population

        ...
    
    def pick_parents(self): #pick up the best available parents

        ...
    
    def crossover(self, ind1, ind2): # make crossover between 2 picked parents

        ...

    def mutate(self, ind1, ind2): # mutate 2 childen

        ...

    def get_best_ind(self): 
        
        ...

def main():

    ...

if __name__ == "__main__":
    main()
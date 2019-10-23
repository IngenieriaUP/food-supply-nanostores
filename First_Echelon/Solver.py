#!/usr/bin/env python
# coding: utf-8

# # Main Imports

import random
import array
import vrp as VRP
import elitism
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle
from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm


# ## Experiment Dictionary

# In[2]:


combinations = [(n_vehicles, hof) for n_vehicles in np.linspace(
    1, 620, num=32, dtype=int) for hof in [5, 10, 20, 30]]

experiment_dict = {'experiment_{}'.format(i): {
    'num_vehicles': combinations[i][0],
    'hof': combinations[i][1],
} for i in range(len(combinations))}


# ## Randomness control

# In[3]:


# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ## Constants for TSP instance

# In[4]:


TSP_NAME = "cost_matrix"
COORDENADAS = "demanda_bodegas"
DEPOT_LOCATION = 0
POPULATION_SIZE = 500


# ## Genetic Algorithm Constants

# In[ ]:


P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.2   # probability for mutating an individual
MAX_GENERATIONS = 10000


# In[ ]:


toolbox = base.Toolbox()


# ## Objective function for distance minimization

# In[ ]:


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))


# ## Instance individual classes

# In[ ]:


creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)


# ## Fitness function definition

# In[ ]:


def vrpDistance(individual):
    return vrp.getMaxDistance(individual),


# In[ ]:


toolbox.register("evaluate", vrpDistance)


# In[ ]:


def genetic_algorithm_vrp(population_size, hall_of_fame_size, max_generations, vrp_fn, best_genetic_fn,
                          min_fitness_fn, mean_fitness_fn, fitness_fn, route_fn, total_veh, save=False, verbose=False, plot=False):
    '''
    @param population_size: population size for experiment
    @param hall_of_fame_size: hof size for experiment
    @param max_generations: max_gens for experiment
    @param vrp_fn: filename to save vrp object
    @param best_genetic_fn: filename to save best_genetic solution object
    @param min_fitness_fn: filename to save min fitness object
    @param mean_fitness_fn: filename to save mean fitness object
    @param fitness_fn: filename to save fitness function plot
    @param route_fn: filename to save optimal route plot
    @param total_veh: total number of vehicles for VRP
    '''
    # Create initial population (generation 0):
    population = toolbox.populationCreator(n=population_size)

    # Prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Define the hall-of-fame object:
    hof = tools.HallOfFame(hall_of_fame_size)

    # Perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                      ngen=1, stats=stats, halloffame=hof, verbose=True)

    # Best individual stats
    best = hof.items[0]
    if verbose:
        print("-- Best Ever Individual = ", best)
        print("-- Best Ever Fitness = ", best.fitness.values[0])

        print("-- Route Breakdown = ", vrp.getRoutes(best))
        print("-- total distance = ", vrp.getTotalDistance(best))
        print("-- max distance = ", vrp.getMaxDistance(best))

    # Main statistics
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    if save:
        cPickle.dump(vrp, open('./Output/pkl_files/{}'.format(vrp_fn), 'wb'), -1)
        cPickle.dump(best, open('./Output/pkl_files/{}'.format(best_genetic_fn), 'wb'), -1)
        cPickle.dump(minFitnessValues, open(
            './Output/pkl_files/{}'.format(min_fitness_fn), 'wb'), -1)
        cPickle.dump(meanFitnessValues, open(
            './Output/pkl_files/{}'.format(mean_fitness_fn), 'wb'), -1)

    if plot:
        # Plot Best Solution
        plt.figure(1)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Best Route with {} vehicles'.format(total_veh))
        vrp.plotData(best)
        plt.savefig('./Output/plots/{}'.format(route_fn))
        plt.show()

        # Plot solution
        plt.figure(2)
        plt.plot(minFitnessValues, color='red')
        plt.plot(meanFitnessValues, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Min / Average Fitness')
        plt.title('Min and Average fitness vs. Generation')
        plt.savefig('./Output/plots/{}'.format(fitness_fn), dpi=300)
        plt.show()


# In[ ]:


failed_experiment = dict()

for key in tqdm(experiment_dict.keys()):
    try:
        vrp = VRP.VehicleRoutingProblem(
            TSP_NAME, COORDENADAS, experiment_dict[key]['num_vehicles'], DEPOT_LOCATION)
        toolbox.register("randomOrder", random.sample,  range(len(vrp)), len(vrp)
                         )  # Operator for randomly shuffled indices in GA
        # Individual creation operator to fill up an Individual instance with shuffled indices
        toolbox.register("individualCreator", tools.initIterate,
                         creator.Individual, toolbox.randomOrder)
        toolbox.register("populationCreator", tools.initRepeat, list,
                         toolbox.individualCreator)  # Population creation operator

        # Genetic operators
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(vrp))
        toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0/len(vrp))

        genetic_algorithm_vrp(POPULATION_SIZE,
                              experiment_dict[key]['hof'],
                              MAX_GENERATIONS,
                              'vrp_solution_{}.pkl'.format(key),
                              'best_genetic_solution_{}.pkl'.format(key),
                              'min_fitness_{}.pkl'.format(key),
                              'mean_fitness_{}.pkl'.format(key),
                              'fitness_function_{}.png'.format(key),
                              'route_plot_{}.png'.format(key),
                              experiment_dict[key]['num_vehicles'], save=True)
    except:
        print(experiment_dict[key])
        failed_experiment[key] = experiment_dict[key]


# In[ ]:

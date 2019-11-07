from deap import tools
from deap import algorithms
import numpy as np


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, epsilon_es=0.0001255, stats=None,
                        halloffame=None, verbose=__debug__, minimize=True):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implment an elitism mechanism. Th eindividuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

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

        # Early Stopping implementation
        # only accumulate consectutive ocurrences

        actual_mean_fitness = np.array(logbook.select("avg"))

        if gen != 1:
            avg_fitness_diff = actual_mean_fitness[-2] - actual_mean_fitness[-1]

            if minimize == True:
                if avg_fitness_diff < actual_mean_fitness[-1]*epsilon_es:
                    iter_counter_es += 1
                else:
                    iter_counter_es = 0  # reset counter when condition is not met
            else:
                if avg_fitness_diff > actual_mean_fitness[-1]*epsilon_es:
                    iter_counter_es += 1
                else:
                    iter_counter_es = 0  # reset counter when condition is not met
        else:
            previous_mean_fitness = actual_mean_fitness
            iter_counter_es = 0

        if iter_counter_es >= 100:
            return population, logbook

    return population, logbook

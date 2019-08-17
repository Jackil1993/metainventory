import numpy as np
import random
import matplotlib.pyplot as plt
from simulation2_sample import Simulator
from simulation import Simulation
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


class SimpleOpt:
    def __init__(self, pop_size, generations, m_rate, c_rate, t_size, type=1):
        self.pop_size = pop_size #population size
        self.generations = generations #number of generations
        self.m_rate = m_rate #mutation rate
        self.c_rate = c_rate #cross-over rate
        self.t_size = t_size #tournament size
        self.type = type

    def ev(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, 400)

        # Structure initializers
        if type == 1:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 2)
        else:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 16)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval(individual, n=1):
            fitness = []
            if self.type == 1:
                for i in range(n):
                    sim1 = Simulation([[0, 0]],  # to_plot, to_report
                                      [[0.1, [0.5, 0.1], [4, 0.5], [20, 2]]],
                                      # interarrivals, demand, replenishment_lead, expiry
                                      [[70.0, 200.0, 1.0, 30.0, 100.0, 100.0]],
                                      # purchase price, sales price, handling, backorder, overflow, recycle
                                      [[individual[0], individual[1]]])  # storage, reorder point

                    fitness.append(sim1.simulate())
            else:
                for i in range(n):
                    sim1 = Simulator(250,  # planning horizon
                                     [10, 8, 7, 10],  # demands
                                     1.0,  # interarrivals
                                     [individual[0], individual[1], individual[2], individual[3]],  # supplies
                                     [individual[4], individual[5], individual[6], individual[7]],  # initial inventory levels
                                     1000,  # total capacity
                                     10,  # overflow_fee
                                     [50, 50, 50, 50],  # prices
                                     [15, 15, 15, 15],  # production costs
                                     [500, 500, 500, 500],  # launch costs
                                     [individual[8], individual[9], individual[10], individual[11]],
                                     # condition to start production
                                     [individual[12], individual[13], individual[14], individual[15]],
                                     # condition to stop production
                                     [30, 30, 30, 30]  # lost-sales fees
                                     )
                    fitness.append(sim1.simulate())

            return (sum(fitness) / len(fitness)),

        toolbox.register("evaluate", eval)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=400, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.t_size)

        def evolve():
            random.seed(64)

            pop = toolbox.population(n=self.pop_size)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("max", np.max)
            stats.register("min", np.min)

            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=self.c_rate, mutpb=self.m_rate, ngen=self.generations,
                                           stats=stats, halloffame=hof, verbose=True)

            return log, hof

        log, hof = evolve()
        print('The fittest solution: ', hof)
        p = Plotter(log.select("gen"), log.select("max"))
        p.plot()


class CustomOpt:
    def __init__(self):
        pass

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)
    toolbox.register("attr_bool", random.randint, 0, 400)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, 16)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    def eval(individual, n=1):
        fitness = []
        if self.type == 1:
            for i in range(n):
                sim1 = Simulation([[0, 0]],  # to_plot, to_report
                                  [[0.1, [0.5, 0.1], [4, 0.5], [20, 2]]],
                                  # interarrivals, demand, replenishment_lead, expiry
                                  [[70.0, 200.0, 1.0, 30.0, 100.0, 100.0]],
                                  # purchase price, sales price, handling, backorder, overflow, recycle
                                  [[individual[0], individual[1]]])  # storage, reorder point

                fitness.append(sim1.simulate())
        else:
            for i in range(n):
                sim1 = Simulator(250,  # planning horizon
                                 [10, 8, 7, 10],  # demands
                                 1.0,  # interarrivals
                                 [individual[0], individual[1], individual[2], individual[3]],  # supplies
                                 [individual[4], individual[5], individual[6], individual[7]],
                                 # initial inventory levels
                                 1000,  # total capacity
                                 10,  # overflow_fee
                                 [50, 50, 50, 50],  # prices
                                 [15, 15, 15, 15],  # production costs
                                 [500, 500, 500, 500],  # launch costs
                                 [individual[8], individual[9], individual[10], individual[11]],
                                 # condition to start production
                                 [individual[12], individual[13], individual[14], individual[15]],
                                 # condition to stop production
                                 [30, 30, 30, 30]  # lost-sales fees
                                 )
                fitness.append(sim1.simulate())

        return (sum(fitness) / len(fitness)),

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", eval)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutUniformInt, indpb=0.05, low=0, up=400)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ----------

    def main():
        random.seed(64)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=40)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.2

        print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Begin the evolution
        while g < 30:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


class Plotter:
    def __init__(self, gens, fits):
        self.gen = gens
        self.fits = fits

    def plot(self):
        fits_plot = [max(0, self.fits[0])]
        for i in range(1, len(self.fits)):
            fits_plot.append(max(self.fits[i], fits_plot[i - 1]))
        plt.grid()
        plt.plot([i + 1 for i in self.gen], fits_plot)
        plt.xlabel("Generation")
        plt.ylabel("Net profit")
        plt.show()


evolver = SimpleOpt(30, 30, 0.05, 0.3, 3, type=1)
evolver.ev()
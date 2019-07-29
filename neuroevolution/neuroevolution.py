import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score, f1_score
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


class Regressors:
    def __init__(self, individual):
        depths = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        widths = [10, 15, 20, 25, 30, 35, 40, 50, 55]
        afs = ['relu', 'elu', 'selu', 'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'sigmoid']
        opts = ['sgd', 'adam', 'adagrad', 'adamax', 'nadam', 'adam', 'adagrad', 'adamax', 'nadam']
        drops = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        l1s = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        l2s = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        self.depth = depths[individual[0]]
        self.width = widths[individual[1]]
        self.af = afs[individual[2]]
        self.opt = opts[individual[3]]
        self.drop = drops[individual[4]]
        self.l1 = l1s[individual[5]]
        self.l2 = l2s[individual[6]]

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(self.width, input_dim=75, kernel_initializer='normal', activation=self.af)) #25
        for layer in range(self.depth):
            model.add(Dense(self.width, kernel_initializer='normal', activation=self.af))
            model.add(Dropout(self.drop, noise_shape=None, seed=None))
            model.add(ActivityRegularization(l1=self.l1, l2=self.l2))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mse', optimizer=self.opt)
        return model


class Classifiers:
    def __init__(self, individual):
        depths = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        widths = [10, 15, 20, 25, 30, 35, 40, 50, 55]
        afs = ['relu', 'elu', 'selu', 'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'sigmoid']
        opts = ['sgd', 'adam', 'adagrad', 'adamax', 'nadam', 'adam', 'adagrad', 'adamax', 'nadam']
        drops = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        l1s = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        l2s = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        self.depth = depths[individual[0]]
        self.width = widths[individual[1]]
        self.af = afs[individual[2]]
        self.opt = opts[individual[3]]
        self.drop = drops[individual[4]]
        self.l1 = l1s[individual[5]]
        self.l2 = l2s[individual[6]]

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(self.width, input_dim=75, kernel_initializer='normal', activation=self.af))  # 17
        for layer in range(self.depth):
            model.add(Dense(self.width, kernel_initializer='normal', activation=self.af))
            model.add(Dropout(self.drop, noise_shape=None, seed=None))
            model.add(ActivityRegularization(l1=self.l1, l2=self.l2))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.opt)
        return model


class BlindWatchmaker:
    def __init__(self, pop_size, generations, m_rate, c_rate, t_size, epochs, b_size, cv, type):
        self.pop_size = pop_size #population size
        self.generations = generations #number of generations
        self.m_rate = m_rate #mutation rate
        self.c_rate = c_rate #cross-over rate
        self.t_size = t_size #tournament size
        self.epochs = epochs #number of epochs
        self.b_size = b_size #batch size
        self.cv = cv #folds in cross validation
        self.type = type

    def get_data(self):
        if self.type == 'regressor':
            #df = pd.read_excel('simulation2_trainingset_full.xlsx')
            df = pd.read_excel('new_10_trainingset_full.xlsx')
            dataset = scale(df.values)
            X = dataset[:, :75]
            Y = dataset[:, 75]
        else:
            # df = pd.read_excel('simulation2_trainingset_class.xlsx')
            df = pd.read_excel('10_class.xlsx')
            dataset = df.values
            X = scale(dataset[:, :75])
            Y = dataset[:, 75]

        return X, Y

    def ev(self):
        X, Y = self.get_data()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, 8)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 7)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval(individual):
            if self.type == 'regressor':
                r = Regressors(individual)
                model = r.baseline_model
                estimator = KerasRegressor(build_fn=model, epochs=self.epochs, batch_size=self.b_size, verbose=0)
                results = cross_val_predict(estimator, X, Y, cv=self.cv)
                r2 = r2_score(Y, results)
                return r2,
            else:
                c = Classifiers(individual)
                model = c.baseline_model
                estimator = KerasClassifier(build_fn=model, epochs=self.epochs, batch_size=self.b_size, verbose=0)
                results = cross_val_predict(estimator, X, Y, cv=self.cv)
                f1 = f1_score(Y, results)
                return f1,

        toolbox.register("evaluate", eval)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=8, indpb=0.2)
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
        print('Praise the fittest one!', hof)
        def plot():
            gen = log.select("gen")
            fits = log.select("avg")
            std = log.select("std")
            std = [i/2 for i in std]
            plt.grid()
            plt.errorbar([i+1 for i in gen], fits, std, uplims=True, fmt='-o', color='blue')
            plt.xlabel("Generation")
            plt.ylabel("F1 score")
            plt.show()
        plot()


evolver = BlindWatchmaker(15, 9, 0.1, 0.4, 3, 40, 4, 2, 'regressor')
evolver.ev()


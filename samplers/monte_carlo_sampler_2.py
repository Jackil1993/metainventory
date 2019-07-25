from simulations import simulation2_sample
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = []
def to_sample(n, k):
    global pos, neg
    def generate_candidate_solution():
        global sim, parameters
        parameters = []
        for j in range(4):
            parameters.append(np.random.uniform(5, 20))
        for j in range(4):
            parameters.append(np.random.uniform(7, 25))
        for j in range(4):
            parameters.append(np.random.uniform(20, 200)) #initial I
        parameters.append(np.random.uniform(0, 100))  # overflow fee
        for j in range(4):
            parameters.append(np.random.uniform(10, 200))  # prices
        for j in range(4):
            parameters.append(np.random.uniform(10, 200))  # production costs
        for j in range(4):
            parameters.append(np.random.uniform(5, 40))  # lost-sasles

        sim = simulation2_sample.Simulator(250,  # planning horizon
                         [parameters[0], parameters[1], parameters[2], parameters[3]],  # demands
                         1.0,  # interarrivals
                         [parameters[4], parameters[5], parameters[6], parameters[7]],  # supplies
                         [parameters[8], parameters[9], parameters[10], parameters[11]],  # initial inventory levels
                         1000,  # total capacity
                         parameters[12],  # overflow_fee
                         [parameters[13], parameters[14], parameters[15], parameters[16]],  # prices
                         [parameters[17], parameters[18], parameters[19], parameters[20]],  # production costs
                         [100, 100, 100, 100],  # launch costs
                         [100, 100, 100, 100],  # condition to start production
                         [200, 200, 200, 200],  # condition to stop production
                         [parameters[21], parameters[22], parameters[23], parameters[24]]) # lost-sales fees

    pos = 0
    neg = 0

    for i in range(k):
        generate_candidate_solution()
        fitness = []
        for j in range(n):
            fitness.append(sim.simulate())
        parameters.append(sum(fitness)/len(fitness))
        if parameters[-1] <0:
            parameters.append(0)
            neg += 1
        else:
            parameters.append(1)
            pos += 1
        if parameters[-2] > -1000000:
            dataset.append(parameters)
        print(i)

to_sample(30, 1000)
df = pd.DataFrame(dataset)
print(df.iloc[:,-2].mean())
print(pos, neg)
ax = sns.boxplot(df.iloc[:,-2])
plt.show()
writer = pd.ExcelWriter('simulation2_trainingset_full.xlsx')
df.to_excel(writer,'Sheet1', header=False, index=False)
writer.save()
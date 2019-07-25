from simulations import simulation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def evaluate(q_r, c, par):
    costs = []
    for sim in range(10):
        s = simulation.Simulation([[0, 0] for i in range(10)],  # to_plot, to_report
                                  par,  # interarrivals, demand, replenishment_lead, expiry
                                  c,  # purchase price, sales price, handling, backorder, overflow, recycle
                                  q_r)  # storage, reorder point
        costs.append(s.simulate())
    return sum(costs)/len(costs)

#strategy
q_r = []
#environment
c = []
par = []
#output
profit = 0.0
dataset=[]

for i in range(500):
    dataset.append([])
    for j in range(5):
        q_r.append([np.random.randint(10, 50), np.random.randint(8, 40)])
        while q_r[j][1] > q_r[j][0]:
            q_r[j][1] = np.random.randint(8, 30)
        dataset[i].append(q_r[-1][0])
        dataset[i].append(q_r[-1][1])

        par.append([np.random.uniform(0.2, 0.7), [np.random.uniform(0.2, 0.7), np.random.uniform(0.05, 0.3)],
                    [np.random.randint(7, 12), np.random.uniform(0.2, 3)], [np.random.randint(8, 15), np.random.uniform(0.2, 3)]])

        while par[j][1][0] <= 2*par[j][1][1]:
            par[j][1][1] = np.random.uniform(0.05, 0.3)

        dataset[i].append(par[-1][0])
        dataset[i].append(par[-1][1][0])
        dataset[i].append(par[-1][1][1])
        dataset[i].append(par[-1][2][0])
        dataset[i].append(par[-1][2][1])
        dataset[i].append(par[-1][3][0])
        dataset[i].append(par[-1][3][1])

        c.append([np.random.randint(60, 150), np.random.randint(150, 200), np.random.uniform(2, 7),
                  np.random.randint(10, 50), np.random.randint(30, 130), np.random.randint(30, 130)])

        while c[j][0] > c[j][1]:
            c[j][0] = np.random.randint(60, 150)

        dataset[i].append(c[-1][0])
        dataset[i].append(c[-1][1])
        dataset[i].append(c[-1][2])
        dataset[i].append(c[-1][3])
        dataset[i].append(c[-1][4])
        dataset[i].append(c[-1][5])

    profit = evaluate(q_r, c, par)
    dataset[i].append(profit)

    q_r = []
    c = []
    par = []
    profit = 0.0
    print(i)

df = pd.DataFrame(dataset)
ax = sns.boxplot(df.iloc[:,-1])
plt.show()
writer = pd.ExcelWriter('new_10_trainingset_full.xlsx')
df.to_excel(writer,'Sheet1', header=False, index=False)
writer.save()
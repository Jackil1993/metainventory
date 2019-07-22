import nolds
from simulations import simulation, simulation2
import matplotlib.pyplot as plt


# running the simulations generating the time series
sim = simulation2.Simulator(10000)
sim.simulate()
s = simulation.Simulation([[1, 1], [0, 0]],  # to_plot, to_report
                          [[0.1, [0.4, 0.2], [10, 2], [7, 2]], [0.1, [0.5, 0.2], [10, 2], [10, 2]]],  # interarrivals, demand, replenishment_lead, expiry
                          [[70.0, 110.0, 5.0, 30.0, 100.0, 100.0], [70.0, 110.0, 5.0, 30.0, 100.0, 100.0]],  # purchase price, sales price, handling, backorder, overflow, recycle
                          [[50, 35], [50, 25]])  # storage, reorder point
s.simulate()

# storing the simulation result into the vector

df2 = sim.stats.individual_inventory_vector
df1 = s.w.products[0].stats.storage[0:3000]
df2 = s.w.products[0].stats.storage[0:6000]
df1 = df1[0::3]
df2 = df2[0::6]


def write_down(df): #save the generated time series
    file = open("time_series1.txt", "w")
    for index in range(len(df)):
        file.write(str(df2[index]) + "\n")
    file.close()


def hurst(df1, df2): #calculating the Hurst exponent and plotiing R/S statistic on log-log plot
    print('Hurst exponent: ', nolds.hurst_rs(df1, debug_plot='True'))
    print('Hurst exponent: ', nolds.hurst_rs(df2, debug_plot='True'))


hurst(df1, df2)


def plot_cd(df1, df2): # calculating and plotting the correlation integal for embeding dimensions in range (1-10)
    cd1 = []
    cd2 = []
    n = []
    for i in range(1, 11):
        cd1.append(nolds.corr_dim(df1, i, fit='RANSAC'))
        cd2.append(nolds.corr_dim(df2, i, fit='RANSAC'))
        n.append(i)
        print(i)
    plt.grid()
    plt.plot(n, cd1, color='red', label='Model 1')
    plt.scatter(n, cd1, color='red')
    plt.plot(n, cd2, color='green', label='Model 2')
    plt.scatter(n, cd2, color='green')
    plt.xlabel('Embedding dimension')
    plt.ylabel('Correlation dimension')
    plt.legend()
    plt.show()
    print('Model 1 max: ', max(cd1))
    print('Model 2 max: ', max(cd2))


def plot_entropy(df1, df2): # calculating and plotting the sample entropy for embeding dimensions in range (1-10)
    cd1 = []
    cd2 = []
    n = []
    for i in range(1, 11):
        cd1.append(nolds.sampen(df1, emb_dim=i))
        cd2.append(nolds.sampen(df2, emb_dim=i))
        n.append(i)
        print(i)
    plt.grid()
    plt.plot(n, cd1, color='red', label='Model 1')
    plt.scatter(n, cd1, color='red')
    plt.plot(n, cd2, color='green', label='Model 2')
    plt.scatter(n, cd2, color='green')
    plt.xlabel('Embedding dimension')
    plt.ylabel('Sample entropy')
    plt.legend()
    plt.show()
    print('Model 1 max: ', max(cd1))
    print('Model 2 max: ', max(cd2))


plot_cd(df1, df2)
plot_entropy(df1, df2)
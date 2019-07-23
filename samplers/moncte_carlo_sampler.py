from simulations import simulation, simulation2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def sample(type):
    dist = []
    if type==1:
        for i in range(10000):
            s = simulation.Simulation([[0, 0]],  # to_plot, to_report
                                      [[0.1, [0.5, 0.1], [4, 0.5], [20, 2]]],  # interarrivals, demand, replenishment_lead, expiry
                                      [[70.0, 100.0, 1.0, 30.0, 100.0, 100.0]],
                                      # purchase price, sales price, handling, backorder, overflow, recycle
                                      [[35, 20]])  # storage, reorder point
            dist.append(s.simulate())
            print(i)


        mu, sigma = stats.norm.fit(dist)
        # create a normal distribution with loc and scale
        test_sample = np.random.normal(mu, sigma, 10000)
        n = stats.norm(loc=mu, scale=sigma)
        print(stats.kstest(dist, n.cdf))
        print(stats.chisquare(dist, test_sample))
        print(stats.anderson(dist, dist='norm'))
        left, right = stats.norm.interval(0.95, loc=mu, scale=(sigma/np.sqrt(len(dist))))


        sns.distplot(np.asarray(dist), bins=100, kde=True,
                     kde_kws={"color": "r", "lw": 3, "label": "Kernel density estimation"})
        plt.axvspan(left, right, alpha=0.35, color='red', label='95% Confidence interval')
        plt.legend()
        plt.xlabel('Cost function')
        plt.show()
        ax = sns.boxplot(np.asarray(dist))
        plt.xlabel('Cost function')
        plt.show()

    else:
        for i in range(10000):
            s = simulation2.Simulator(250)
            dist.append(s.simulate())
            print(i)

        mu, sigma = stats.norm.fit(dist)
        # create a normal distribution with loc and scale
        test_sample = np.random.normal(mu, sigma, 10000)
        print('mu: ', mu, 'n/sigma: ', sigma)
        print(stats.anderson(dist, dist='norm'))
        left, right = stats.norm.interval(0.95, loc=mu, scale=(sigma / np.sqrt(len(dist))))

        sns.distplot(np.asarray(dist), bins=100, kde=True,
                     kde_kws={"color": "r", "lw": 3, "label": "Kernel density estimation"})
        plt.axvspan(left, right, alpha=0.35, color='red', label='95% Confidence interval')
        plt.legend()
        plt.xlabel('Cost function')
        plt.show()
        ax = sns.boxplot(np.asarray(dist))
        plt.xlabel('Cost function')
        plt.show()


sample(1)
import numpy as np
import matplotlib.pyplot as plt


class Generator:
    def normal(self, mu, sigma):
        return np.random.normal(mu, sigma)

    def exponential(self, betta):
        return np.random.exponential(betta)

    def uniform(self):
        return np.random.uniform()


class Plant:
    def __init__(self, markets, interarrival, rate, T, markovian=False):
        self.g = Generator()
        self.time = 0.0
        self.markets = []
        self.T = T
        self.rate = rate
        self.status = [True, True, True, True]
        for market in markets:
            self.markets.append(Market(market, T, markovian))
        self.inventory = [100, 100, 100, 100]
        self.total_capacity = 1000
        self.overflow_fee = 10
        self.prices = [20, 20, 20, 20]
        self.production_costs = [15, 15, 15, 15]
        self.launch_costs = [500, 500, 500, 500]
        self.start = [100, 100, 100, 100]
        self.stop = [200, 200, 200, 200]
        self.queue = []
        self.max_orders = 5
        self.backlog_fees = [30, 30, 30, 30]
        self.interarrival = interarrival
        self.next_demand = self.g.exponential(self.interarrival)
        self.profit = 0.0
        for i,j in zip(self.inventory, self.production_costs):
            self.profit -= i*j

    def produce(self, lag):
        for i in range(len(self.inventory)):
            if self.status[i] == True:
                produced = self.g.normal(self.rate[i][0], self.rate[i][0]*self.rate[i][1])*lag
                self.profit -= produced*self.production_costs[i]
                self.inventory[i] += produced
            if self.inventory[i] >= self.stop[i] and self.status[i]==True:
                self.status[i] = False
                #print('!!! production of product {} has been interrupted'.format(i))
            if sum(self.inventory) > self.total_capacity:
                #print('overflow!')
                self.profit -= (sum(self.inventory) - self.total_capacity)*self.overflow_fee

    def advance_time(self):
        lag = self.next_demand - self.time
        self.time = self.next_demand
        self.produce(lag)
        self.next_demand = self.time + self.g.exponential(self.interarrival)
        self.queue.append(self.markets[0].demand_arises())
        for i in range(len(self.queue)):
            if all(self.inventory[j] >= self.queue[i][j] for j in range(len(self.inventory))) is True:
                #print('{} pieces are sold and shipped'.format(sum(self.queue[i])))
                self.inventory = list(map(lambda x, y: x-y, self.inventory, self.queue[i]))
                for j,k in zip(self.queue[i], self.prices):
                    self.profit += j*k
                self.queue[i] = [0 for i in range(len(self.queue[i]))]

        for i in range(len(self.queue)):
            try:
                if self.queue[i][0] == 0:
                    self.queue.pop(i)
            except:
                pass

        if len(self.queue) > self.max_orders:
            #print('backlog!')
            tmp = self.queue.pop(-1)
            for i,j in zip(tmp, self.backlog_fees):
                self.profit -= i*j

        for i in range(len(self.inventory)):
            if self.inventory[i] < self.start[i] and self.status[i] == False:
                self.status[i] = True
                self.profit -= self.launch_costs[i]
                #print('!!! production of product {} has been launched'.format(i))


class Market:
    def __init__(self, parameters, transitions, markovian):
        self.g = Generator()
        self.demands = parameters[0]
        self.markovian = markovian
        self.transitions = transitions
        self.states = [0, 0, 0, 0]

    def shift(self):
        for i in range(len(self.demands)):
            r = self.g.uniform()
            if r <= self.transitions[i][self.states[i]][0]:
                self.states[i] = 0
            elif self.transitions[i][self.states[i]][0] < r <= self.transitions[i][self.states[i]][1] + self.transitions[i][self.states[i]][0]:
                self.states[i] = 1
                self.demands[i][0] = self.demands[i][0]*1.01
            else:
                self.states[i] = 2
                self.demands[i][0] = self.demands[i][0] * 0.99

    def demand_arises(self):
        demand = []
        if self.markovian == False:
            self.shift()
        for i in range(len(self.demands)):
            demand.append(self.g.normal(self.demands[i][0], self.demands[i][0]*self.demands[i][1]))
        #print('demand arises {}'.format(demand))
        return demand


class Statistics:
    def __init__(self):
        self.time_vector = []
        self.inventory_vector = []
        self.individual_inventory_vector = []
        self.profit_vector = []
        self.statuses = [[], [], [], []]

    def gather_inventory_stats(self, time, inventory, individual_inventory, profit, status):
        self.time_vector.append(time)
        self.inventory_vector.append(inventory)
        self.individual_inventory_vector.append(individual_inventory)
        self.profit_vector.append(profit)
        for i in range(len(status)):
            self.statuses[i].append(status[i])

    def plot_inventory(self, x, y):
        plt.grid()
        plt.plot(x, y, color='orange')
        plt.xlabel('time')
        plt.ylabel('inventory level')
        plt.show()

    def plot_individual_inventory(self, x, y, down, up):

        fig, axs = plt.subplots(2, 1)
        axs[0].grid(True)
        axs[0].plot(x, y, color='orange', label='inventory dynamics')
        axs[0].hlines(down, 0, max(x)-1, color='green', linestyles=':', linewidth=3, label='resume production')
        axs[0].hlines(up, 0, max(x) - 1, color='red', linestyles=':', linewidth=3, label='interrupt production')
        axs[0].set_ylabel('inventory level')
        #axs[0].set_xlabel('time')
        axs[0].legend()

        axs[1].grid(True)
        axs[1].plot(x, self.statuses[0], color='blue', label='status')
        axs[1].set_yticks([0.0, 1.0])
        axs[1].set_yticklabels(["False", "True"])
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('production status')
        axs[1].legend()
        fig.tight_layout()
        plt.show()

    def plot_phase(self, x):
        plt.grid()
        plt.plot(x[0:-1], x[1:], color='blue')
        plt.xlabel('inventory level (t)')
        plt.ylabel('inventory level (t+1)')
        plt.show()

    def plot_profit(self, x, y):
        plt.grid()
        plt.plot(x, y, color='green', label='monetary dynamics')
        plt.hlines(0,0,self.time_vector[-1], color='red', linestyles=':', linewidth=3, label='break-even point')
        plt.xlabel('time')
        plt.ylabel('net profit')
        plt.legend()
        plt.show()

    def plot_status(self, x, y):
        plt.grid()
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(len(y)):
            plt.plot(x, y[i], color=colors[i], label='status {}'.format(i+1))
        plt.xlabel('time')
        plt.ylabel('production status')
        plt.legend()
        plt.show()


class Simulator:
    def __init__(self, planning_horizon):

        self.plant = Plant([[[[10, 0.2], [8, 0.2], [7, 0.2], [10, 0.2]]]],
    1.0,
    [[12, 0.2], [10, 0.2], [9, 0.2], [12, 0.2]],
                           [np.array([[0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33]]),
                            np.array([[0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33]]),
                            np.array([[0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33]]),
                            np.array([[0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33],
                                      [0.33, 0.34, 0.33]])
                            ])
        self.stats = Statistics()
        self.planning_horizon = planning_horizon

    def simulate(self):
        while self.plant.time < self.planning_horizon:
            self.plant.advance_time()

            self.stats.gather_inventory_stats(self.plant.time, sum(self.plant.inventory), self.plant.inventory[0],
                                              self.plant.profit, self.plant.status)
        self.stats.plot_inventory(self.stats.time_vector, self.stats.inventory_vector)
        self.stats.plot_individual_inventory(self.stats.time_vector, self.stats.individual_inventory_vector,
                                             self.plant.start[0], self.plant.stop[0])
        self.stats.plot_profit(self.stats.time_vector, self.stats.profit_vector)
        #self.stats.plot_status(self.stats.time_vector, self.stats.statuses)
        self.stats.plot_phase(self.stats.inventory_vector)
        print(self.plant.markets[0].demands)
        return self.plant.profit

#sim1 = Simulator(250)
#sim1.simulate()
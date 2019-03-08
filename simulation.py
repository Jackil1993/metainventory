import numpy as np
import matplotlib.pyplot as plt

# create a .txt file to write down all the discrete events for further  tracing)
file = open("protocol.txt","w")

# a static class containing all the utilized random number generators


class Generator:
    def normal(self, mu, sigma):
        return np.random.normal(mu, sigma)

    def exponential(self, betta):
        return np.random.exponential(betta)

# the "Warehouse" includes the parameters shared by all the operable products


class Warehouse:
    def __init__(self, adjustments, parameters, costs, strategy, total_capacity=300):
        self.total_capacity = total_capacity
        self.free_capacity = total_capacity
        self.time = 0.0
        self.products = []

        self.to_report = bool(adjustments[0][1])

        # we generate the operable products
        for product in range(len(parameters)):
            self.products.append(Product(product, adjustments[product], parameters[product], costs[product], strategy[product],
                                         total_capacity))

    def check_free_capacity(self):
        used = 0.0
        for product in range(len(self.products)):
            used += sum(self.products[product].lots)
        self.free_capacity = self.total_capacity - used
        if self.free_capacity < 0.0:
            self.free_capacity = 0.0
        if self.to_report is True:
            file.write('\nFree capacity = {}'.format(round(self.free_capacity, 2)))

    def advance_time(self):
        demands = list(map(lambda x: x.next_demand, self.products))
        replenishments = list(map(lambda x: x.next_rep, self.products))
        self.check_free_capacity()
        if min(demands) <= min(replenishments):
            self.time = self.products[np.argmin(demands)].handle_demand(self.time)
        else:
            self.time = self.products[np.argmin(replenishments)].handle_replenishment(self.time, self.free_capacity)


class Product:
    def __init__(self, id, adjustments, parameters, costs, strategy, total_capacity):
        self.g = Generator()
        self.id = id

        self.interarrivals = parameters[0]
        self.demand = parameters[1]
        self.replenishment_lead = parameters[2]
        self.expiry = parameters[3]

        self.lots = [strategy[0]]
        self.expiry_date = [self.g.normal(parameters[3][0], parameters[3][1])]
        self.status = False

        self.next_demand = self.g.exponential(self.interarrivals)
        self.next_rep = float('inf')
        self.next_event = self.next_demand

        self.reorder_point = strategy[1]
        self.reorder_size = strategy[0]

        #counters
        self.backorders = 0.0
        self.overflows = 0.0
        self.expired = 0.0

        #costs
        self.purchase_prise = costs[0] #includes delivery
        self.sales_prise = costs[1]
        self.total_capacity = total_capacity # needed to calculate return to scale
        self.purchase_prise = self.purchase_prise if self.reorder_size < 0.5*self.total_capacity else 0.7*self.purchase_prise
        self.handling_cost = costs[2]
        self.backorder_fee = costs[3]
        self.overflow_fee = costs[4]
        self.recycle_fee = costs[5]
        self.income = 0.0
        self.costs = 0.0

        self.to_plot = bool(adjustments[0])
        self.to_report = bool(adjustments[1])
        if self.to_plot == True:
            self.stats = Statistics(self.reorder_point, self.reorder_size)


    def check_expiry(self, time):
        for lot in range(len(self.expiry_date)):
            if self.expiry_date[lot] <= 0.0 and self.lots[lot] > 0:
                if self.to_report is True:
                    file.write('\nThe lot № {} of product {} is expired. {} pieces will be recycled'.format(lot, self.id,
                                                                                                        round(float(self.lots[lot]), 2)))
                try:
                    self.stats.expires.append(time)
                    self.stats.expires_time.append(sum(self.lots))
                except AttributeError:
                    pass
                if len(self.lots) > 1:
                    self.expiry_date.pop(lot)
                    tmp = self.lots.pop(lot)
                    self.expired += tmp
                    self.costs += self.recycle_fee * tmp
                else:
                    self.expiry_date[0] = 0
                    self.expired += self.lots[0]
                    self.costs += self.recycle_fee * self.lots[0]
                    self.lots[0] = 0
                break

    def handle_demand(self, time):
        to_handle = abs(self.g.normal(self.demand[0], self.demand[1]))
        self.expiry_date = list(map(lambda x: x - (self.next_demand - time), self.expiry_date))
        self.check_expiry(time)
        self.costs += (self.next_demand - time)*sum(self.lots)*self.handling_cost #handling costs
        time = self.next_demand
        self.next_demand = time + self.g.exponential(self.interarrivals)
        if self.to_report is True:
            file.write('\ntime {}: {} pieces of product {} have been demanded'.format(round(time, 2), round(to_handle, 2), self.id))
        empty_lots = 0
        for lot in range(len(self.lots)):
            if self.lots[lot] >= to_handle > 0.0:
                self.lots[lot] -= to_handle
                self.income += to_handle * self.sales_prise
                if self.to_report is True:
                    file.write('\nNo backorder arose')
                break
            else:
                to_handle -= self.lots[lot]
                self.income += self.lots[lot] * self.sales_prise
                self.backorders += to_handle
                empty_lots += 1
                try:
                    self.stats.backorders.append(time)
                except AttributeError:
                    pass
                if self.to_report is True:
                    file.write('\nBackorder of {} arose'.format(round(to_handle, 2)))
        if empty_lots != 0.0 and len(self.lots) > 1:
            for i in range(empty_lots):
                self.lots.pop(i)
                self.expiry_date.pop(i)
                break
            empty_lots = 0.0
        if self.to_report is True:
            file.write('\nStorge {} is {}. Backorder is {}'.format(self.id, [round(element) for element in self.lots], round(self.backorders)))
        if sum(self.lots) <= self.reorder_point and self.status == False:
            self.replenish(time)
            self.costs += self.purchase_prise * self.reorder_size #product is orderd
        try:
            self.stats.update_storage(time, sum(self.lots),
                                      (self.income - self.costs)) #gather stats
        except AttributeError:
            pass
        return time

    def replenish(self, time):
        self.status = True
        self.next_rep = time + self.g.normal(self.replenishment_lead[0], self.replenishment_lead[1])
        if self.to_report is True:
            file.write('\n{} pieces of product {} have been ordered'.format(round(self.reorder_size, 2), self.id))

    def handle_replenishment(self, time, free_capacity):
        self.expiry_date = list(map(lambda x: x - (self.next_rep - time), self.expiry_date))
        self.check_expiry(time)
        if free_capacity >= self.reorder_size:
            self.lots.append(self.reorder_size)
            self.expiry_date.append(self.g.normal(self.expiry[0], self.expiry[1]))
            if self.to_report is True:
                file.write('\ntime {}: {} pieces of product {} have been replehished'.format(round(time, 2),
                                                                                             round(self.reorder_size, 2), self.id))
        else:
            self.lots.append(free_capacity)
            self.expiry_date.append(self.g.normal(self.expiry[0], self.expiry[1]))
            self.overflows += (self.reorder_size - free_capacity)
            self.costs += (self.reorder_size - free_capacity) * self.overflow_fee #fee for overflow
            if self.to_report is True:
                file.write('\nStorage overflow {} pieces of product {} are sent back'.format((self.reorder_size - free_capacity), self.id))
        self.costs += (self.next_rep - time) * sum(self.lots) * self.handling_cost #handling costs
        time = self.next_rep
        self.status = False
        self.next_rep = float('inf')
        if free_capacity < self.reorder_size or free_capacity < 0.0:
            free_capacity = 0.0
        return time


class Statistics:
    def __init__(self, reorder_line, size_line):
        self.time = []
        self.storage = []
        self.profit = []
        self.reorder_line = reorder_line
        self.size_line = size_line
        self.backorders = []
        self.expires = []
        self.expires_time =[]
        self.overflows = []

    def update_storage(self, time, storage, profit):
        self.time.append(time)
        self.storage.append(storage)
        self.profit.append(profit)

    def plot_storage(self):
        storage_dynamics = plt.figure()
        plt.plot(self.time, self.storage, color='orange', label='storage')
        plt.axhline(self.reorder_line, color='red', linestyle='--', label='reorder point')
        plt.axhline(self.size_line, color='black', linestyle='--', label='reorder size')
        plt.scatter(self.expires, self.expires_time, color='black', marker='x', label='goods are expired')
        plt.scatter(self.backorders, [0 for i in range(len(self.backorders))], color='red', s=2, label='backorders')
        plt.legend()
        plt.xlabel('modeling time')
        plt.ylabel('inventory')
        plt.show()

    def plot_profit(self):
        money_dynamics = plt.figure()
        plt.plot(self.time, self.profit, color='orange')
        plt.axhline(self.reorder_line, color='red', linestyle='--', label='break-even point')
        plt.legend()
        plt.xlabel('modeling time')
        plt.ylabel('net profit')
        plt.show()

    def plot_phase(self):
        phase = plt.figure()
        #plt.plot(self.storage, self.profit)
        plt.plot(self.storage[0:-1], self.storage[1:])
        #plt.plot(self.profit[0:-1], self.profit[1:])
        plt.title('Pseudo phase portait')
        plt.xlabel('I(t)')
        plt.ylabel('I(t+1)')
        plt.show()


class Simulation:
    def __init__(self, adjustments, parameters, costs, strategy, horizon=120.0):
        #np.random.seed(seed)
        self.w = Warehouse(adjustments, parameters, costs, strategy)
        self.horizon = horizon

    def simulate(self):
        while self.w.time < self.horizon:
            self.w.advance_time()
        try:
            self.w.products[0].stats.plot_storage()
            self.w.products[0].stats.plot_profit()
            self.w.products[0].stats.plot_phase()
        except AttributeError:
            pass
        total_cost = 0.0
        for i in range(len(self.w.products)):
            total_cost += (self.w.products[i].income - self.w.products[i].costs - self.w.products[i].backorders*self.w.products[i].backorder_fee)
        return total_cost
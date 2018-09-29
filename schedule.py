import numpy as np
from mesa.time import SimultaneousActivation


class MarketActivation(SimultaneousActivation):

    def __init__(self, model):
        super().__init__(model)

    def agent_count(self):
        self.num = len(self.agents)

    def benefit(self, p_matrix, a_matrix):
        for i in range(0, self.num):
            self.agents[i].benefit_table(p_matrix[i], a_matrix[i])

    def learn_d(self, p_matrix):
        price_sum = np.sum(p_matrix)
        non_zero = np.nonzero(p_matrix)[0].shape[0]  # get the total number of non-zeros

        if non_zero != 0:
            price_avg = price_sum / non_zero
            for i in range(0, self.num):
                z = np.array(p_matrix[i])
                if np.sum(z) > 0:
                    self.agents[i].learn()  # learn the outflow, water use and outflow to maximize the benefit
                else:
                    self.agents[i].learn_price(price_avg)  # only learn the price
        else:  # No transaction occurs in the market
            for i in range(0, self.num):
                self.agents[i].learn_by_random()

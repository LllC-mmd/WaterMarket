import numpy as np
from mesa.time import SimultaneousActivation


class MarketActivation(SimultaneousActivation):

    def __init__(self, model):
        super().__init__(model)
        self.num = len(self.agents)

    def control(self):
        for i in range(0, self.num):
            self.agents[i].control()
            self.model.f_matrix[i] = self.agents[i].outflow

    def benefit(self):
        for i in range(0, self.num):
            self.agents[i].benefit_table()

    def learn_d(self, p_matrix):
        total_cost = np.sum(p_matrix)
        non_zero = np.nonzero(p_matrix)[0].shape[0]

        if total_cost != 0:
            average_cost = total_cost / non_zero
            for i in range(0, self.num):
                z = np.array(p_matrix[i])
                if np.sum(z) > 0:
                    tau = np.min(z[z>0])
                    self.agents[i].learn(tau)
                else:
                    tau = average_cost
                    self.agents[i].learn(tau)
        else:
            for i in range(0, self.num):
                self.agents[i].learn_by_random()

import numpy as np
from mesa.time import SimultaneousActivation


class MarketActivation(SimultaneousActivation):

    def __init__(self, model):
        super().__init__(model)

    def learn_d(self, p_matrix):
        total_cost = np.sum(p_matrix)
        non_zero = np.nonzero(p_matrix)[0].shape[0]
        t = len(self.agents)
        if total_cost != 0:
            average_cost = total_cost / non_zero
            for i in range(0, t):
                z = np.array(p_matrix[i])
                if np.sum(z) > 0:
                    tau = np.min(z[z>0])
                    self.agents[i].learn(tau)
                else:
                    tau = average_cost
                    self.agents[i].learn(tau)
        else:
            for i in range(0, t):
                self.agents[i].learn_by_random()

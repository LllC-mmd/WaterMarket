import numpy as np
from mesa import Agent


w = 0.1  # market_transaction_ratio


class WaterUser(Agent):

    def __init__(self, unique_id, model,
                 x, u_a, u_b, u_c, w, L,
                 outflow, inflow,
                 transaction_size, res,
                 beta, mu):
        super().__init__(unique_id, model)
        self.x = x  # water use
        self.u_a = u_a
        self.u_b = u_b
        self.u_c = u_c
        self.u = u_a*self.x*self.x + u_b*self.x + u_c  # benefit brought by the use of water
        self.permit = w  # water permit of user
        self.store = L  # local water available
        self.outflow = outflow
        self.inflow = inflow
        self.limit = np.sum(inflow) - np.sum(outflow) + L
        self.role_choose()
        self.res = res  # parameter for reservation price, represents marginal profit of the water
        self.transaction_size = transaction_size
        self.beta = beta
        self.mu = mu
        if self.limit < self.permit:
            self.label = 'over'  # there are some water rights which surely can't be use
        else:
            self.label = 'normal'

    def role_choose(self):
        x = self.x
        q = self.permit
        if x > q:
            self.market_role = 'buyer'
        elif x < q:
            self.market_role = 'seller'
        else:
            self.market_role = 'sider'

    def buy(self):
        self.bid_amount = self.x - self.permit
        self.reservation_price = self.res / (1+w)
        self.bid_price = (1 - self.mu) * self.reservation_price

    def sell(self):
        self.bid_amount = self.permit - self.x
        self.reservation_price = self.res / (1 - w)
        self.bid_price = (1 + self.mu) * self.reservation_price

    def learn(self, tau):  # tau is the transaction cost
        self.x = max(self.x + self.beta * (2 * self.u_a * self.x + self.u_b - (1 + w) * tau), 0)
        if self.market_role == 'buyer':
            self.mu = self.mu - self.beta * (tau - self.bid_price) / self.reservation_price
        else:
            self.mu = self.mu + self.beta * (tau - self.bid_price) / self.reservation_price

    def learn_by_random(self):
        ratio = np.random.uniform(0.5, 1)
        self.mu = self.mu * ratio

    def balance_check(self):
        pass

    def control(self):
        pass

    def step(self):
        self.role_choose()
        if self.market_role == 'buyer':
            self.buy()
        elif self.market_role == 'seller':
            self.sell()
        else:
            pass

    def advance(self):
        pass


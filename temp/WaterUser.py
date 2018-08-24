import numpy as np
from mesa import Agent


w = 0.1  # market_transaction_ratio


class WaterUser(Agent):

    def __init__(self, unique_id, model,
                 x, u_a, u_b, u_c, w, L,
                 out_link, in_link,
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
        self.out_link = out_link  # the unique_id of users who have waterways in the downstream of the user
        self.in_link = in_link
        self.role_choose()
        self.res = res  # parameter for reservation price, represents marginal profit of the water
        self.transaction_size = transaction_size
        self.beta = beta
        self.mu = mu
        self.precipitation = self.model.f_matrix[self.unique_id][self.unique_id]

    def balance(self):
        # water_balance holds true
        while self.x > self.limit:
            ratio = np.random.uniform(0.5, 1)
            # If water use exceeds its net flow_in, water use should be decreased
            if self.x > np.sum(self.inflow) + self.store + self.precipitation:
                self.x = self.x * ratio
            # Else, decrease the outflow to random out_links
            else:
                choice_num = len(self.out_link)
                d = np.random.randint(0, choice_num, 1)
                self.outflow[d - 1] = self.outflow[d - 1] * ratio
                # re-calculate the water table
                self.water_table()

    def control(self, f):  # decide the outflow
        i = 0
        for user in self.out_link:
            self.model.f_matrix[self.unique_id][user] = f[i]
            i += 1

    def water_table(self):
        self.outflow = self.model.f_matrix[self.unique_id]  # array of outflow, including the flow from i to i
        self.inflow = self.model.f_matrix.transpose()[self.unique_id]
        self.limit = np.sum(self.inflow) - np.sum(self.outflow) + self.store + self.precipitation  # water use limit

    def role_choose(self):
        x = self.x
        q = self.permit
        if x > q:  # water use > permit
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

    def label_choose(self):
        if self.limit < self.permit:
            self.label = 'over'  # there are some water permits which can't be used
            # under water balance (use <= limit), 'over' user must be a seller
        else:
            self.label = 'normal'

    def learn(self, tau):  # tau is the transaction cost
        self.x = max(self.x + self.beta * (2 * self.u_a * self.x + self.u_b - (1 + w) * tau), 0)
        if self.market_role == 'buyer':
            self.mu = self.mu - self.beta * (tau - self.bid_price) / self.reservation_price
        else:
            self.mu = self.mu + self.beta * (tau - self.bid_price) / self.reservation_price
        # how to learn the water use and outflow ???

    def learn_by_random(self):
        ratio = np.random.uniform(0.5, 1, 1)
        self.mu = self.mu * ratio

    def step(self):
        self.water_table()  # calculate the outflow, the inflow, the water use limit
        self.balance()  # check if the water balance holds; if not, re-balance the water table
        self.role_choose()  # choose the role in the market for the water user
        if self.market_role == 'buyer':
            self.buy()
        elif self.market_role == 'seller':
            self.sell()
        else:
            pass
        self.label_choose()  # choose the label for the water user

    def advance(self):
        pass


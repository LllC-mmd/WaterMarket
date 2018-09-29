import numpy as np
import random
from scipy.stats import truncnorm
from mesa import Agent


w = 0.05  # market_transaction_ratio
phi = 0.1  # regency ratio
pi = np.pi  # 3.1415926
excess_fee = -1000  # fine charged for the unit excess water use


def propensity(x, mu, sheet, ini):   # the sheet is a record of (x, mu, benefit)
    q = ini
    for i in range(1, len(sheet)):
        E = (sheet[i][2]-sheet[i-1][2])/abs(sheet[i-1][2])
        E = E*1/(2*pi)*np.exp(-0.5*((x-sheet[i][0])/sheet[i][0])**2-0.5*((mu-sheet[i][1])/sheet[i][0])**2)
        q = (1-phi)*q + E
    return q

# metropolis_hastings sampling algorithms
def metropolis_hastings(user):
   # initialization
    x = user.x
    mu = user.mu
    sheet = user.sheet
    # burn-in process
    for i in range(0, 10000):
        # select a candidate for x, mu
        x_candidate = truncnorm.rvs(0, user.limit)
        if user.market_role == 'buyer':
            mu_candidate = truncnorm.rvs(0, 1)
        else:
            mu_candidate = truncnorm.rvs(0, 10)
        # compute the acceptance rate
        q_candidate = propensity(x=x_candidate, mu=mu_candidate, sheet=sheet, ini=user.p_ini)
        q_t = propensity(x=x, mu=mu, sheet=sheet, ini=user.p_ini)
        rate = min(1, q_candidate/q_t)
        u = random.uniform(0, 1)
        if u < rate:
            x = x_candidate
            mu = mu_candidate
    # do sampling
    while True:
        x_candidate = truncnorm.rvs(0, user.limit)
        print("User limit: " + str(user.limit))
        if user.market_role == 'buyer':
            mu_candidate = truncnorm.rvs(0, 1)
        else:
            mu_candidate = truncnorm.rvs(0, 10)
        q_candidate = propensity(x=x_candidate, mu=mu_candidate, sheet=sheet, ini=user.p_ini)
        q_t = propensity(x=x, mu=mu, sheet=sheet, ini=user.p_ini)
        rate = min(1, q_candidate / q_t)
        u = random.uniform(0, 1)
        if u < rate:
            break
        #if t > 100000:
            #break
        #t += 1
    return x_candidate, mu_candidate


class WaterUser(Agent):

    def __init__(self, unique_id, model,
                 x, u_a, u_b, u_c, w, L,
                 out_link, in_link, out_min, penalty,
                 transaction_size, res,
                 beta, mu):
        super().__init__(unique_id, model)
        self.x = x  # water use
        self.u_a = u_a
        self.u_b = u_b
        self.u_c = u_c
        self.permit = w  # water permit of user
        self.store = L  # local water available
        self.out_link = out_link  # the unique_id of users who have waterways in the downstream of the user
        self.in_link = in_link
        self.out_min = out_min  # the minimum outflow on each out_link
        self.penalty = penalty
        self.role_choose()
        self.res = res  # parameter for reservation price, represents marginal profit of the water
        self.transaction_size = transaction_size
        self.beta = beta
        self.mu = mu
        self.precipitation = self.model.f_matrix[self.unique_id][self.unique_id]
        self.sheet = [[0,0,-10000]]  # self.sheet is a record of [x, mu, benefit] for every successful transaction
        self.time = 0

    def balance(self):
        # water_balance holds true
        self.water_table()  # calculate the water table to start the computation
        choice_num = len(self.out_link)
        while self.x > self.limit:
            ratio = np.random.uniform(0.5, 1)
            # If water use exceeds its net flow_in
            # or there is no out_link, water use should be decreased
            if self.x > np.sum(self.inflow) + self.store or choice_num == 0:  # self.inflow contains the precipitation
                self.x = self.x * ratio
            # Else, decrease the outflow to random out_links
            else:
                d = np.random.randint(0, choice_num, 1)
                self.outflow[d - 1] = self.outflow[d - 1] * ratio
                # re-calculate the water table
            self.water_table()

    def water_table(self):  # water table set a constraint for water use x
        self.outflow = self.model.f_matrix[self.unique_id]  # array of outflow, including the flow from i to i
        self.inflow = self.model.f_matrix.transpose()[self.unique_id]
        self.limit = np.sum(self.inflow) - np.sum(self.outflow) + self.store + self.precipitation  # water use limit

    def outflow_initialize(self):  # decide the outflow based on the minimum flow constraints
        n = self.out_link.shape[0]  # the number of out_links
        if n == 0:
            pass
        else:
            outflow_sum = np.sum(self.inflow) + self.store - self.x
            if outflow_sum < 0:
                self.x = np.sum(self.inflow) + self.store
            else:
                min_sum = np.sum(self.out_min)
                if min_sum > 0:
                    for link in self.out_link:
                        q = self.out_min[link]*outflow_sum/min_sum
                        self.outflow[link] = q
                        self.model.f_matrix[self.unique_id][link] = q
                else:
                    q_avg = outflow_sum/n
                    for link in self.out_link:
                        self.outflow[link] = q_avg
                        self.model.f_matrix[self.unique_id][link] = q_avg

    def benefit_table(self, price, amount):
        # utility brought by the use of water
        u = self.u_a * self.x ** 2 + self.u_b * self.x + self.u_c
        # transaction income/cost (here, we all use 'income', which is negative for buyers)
        # Note: a_matrix[i][j] is positive if i is the buyer and j is the seller
        income = -np.sum(price*amount)
        income = income - w*np.sum(price*np.abs(amount))
        # penalty caused by the violation of minimum outflow
        delta = self.outflow-self.out_min
        f = self.penalty*delta
        fine_min = np.sum(f[f<0])
        # penalty caused by the excess water use
        fine_excess = excess_fee*max(0, self.x-self.permit-np.sum(amount))
        # calculate the net benefit
        self.benefit = u + income + fine_min + fine_excess
        # if the transaction happens and the , record it into the agent's sheet
        if np.sum(amount) == self.x-self.permit:
            self.sheet_up()

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
        self.bid_price = (1 - self.mu) * self.reservation_price  # mu is in (0, 1)

    def sell(self):
        self.bid_amount = self.permit - self.x
        self.reservation_price = self.res / (1 - w)
        self.bid_price = (1 + self.mu) * self.reservation_price  # mu is in (0, infinity)

    def label_choose(self):
        if self.limit < self.permit:
            self.label = 'over'  # there are some water permits which can't be used
            # under water balance (use <= limit), 'over' user must be a seller
        else:
            self.label = 'normal'

    # learn the outflow, water use and outflow
    def learn(self):
        self.x, self.mu = metropolis_hastings(self)
        self.balance()

    # learn the price
    def learn_price(self, tau):
        if self.market_role == 'buyer':
            mu = min(self.mu - self.beta * (tau - self.bid_price) / self.reservation_price, 1)
            mu = max(mu, 0)
            self.mu = mu
        else:
            self.mu = max(self.mu + self.beta * (tau - self.bid_price) / self.reservation_price, 0)

    def learn_by_random(self):
        ratio = np.random.uniform(0.5, 1)
        self.mu = self.mu * ratio

    def sheet_up(self):
        self.sheet.append([self.x, self.mu, self.benefit])

    def propensity_initialization(self):
        if self.market_role == 'buyer':
            self.p_ini = 1/(self.limit)
        else:
            self.p_ini = 1/(10*self.limit)

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
        # initialize the propensity
        if self.time == 0:
            self.propensity_initialization()
        self.label_choose()  # choose the label for the water user
        self.time += 1

    def advance(self):
        pass

import numpy as np
import copy
from WaterUser import WaterUser
from schedule import MarketActivation
from mesa import Model


def local_optimal(upper_bound, u):
    c_1 = upper_bound
    c_2 = 0
    c_3 = -u[1]/(2*u[0])  # -b/2a
    v_1 = u[0]*c_1*c_1 + u[1]*c_1 + u[2]
    v_2 = u[2]
    if c_3 <= c_1:
        return c_3
    elif v_1 > v_2:
        return c_1
    else:
        return c_2


def role_check(water_users, basin_matrix, role):
    for i in range(0, basin_matrix.shape[0]):
        if water_users[i].x >= water_users[i].limit:
            for j in range(0, basin_matrix.shape[0]):
                if basin_matrix[i][j] > 0:
                    water_users[j].market_role='sider'
                    role[j] = 'sider'
                    water_users[j].x = local_optimal(upper_bound=water_users[i].limit,
                                                     u=[water_users[j].u_a,water_users[j].u_b,water_users[j].u_c])


def limit_check(market):
    i = 0
    for user in market.users:
        if user.x > user.limit:
            user.x = local_optimal(upper_bound=user.limit, u=[user.u_a, user.u_b, user.u_c])
        i += 1
    x_update(market)
    role_update(market)


def role_update(market):
    i = 0
    for user in market.users:
        market.role[i] = user.market_role
        i += 1


def x_update(market):
    i = 0
    for user in market.users:
        market.x[i] = user.x
        i += 1


class WaterMarket(Model):

    def __init__(self, basin_matrix, u, water_permit, res, beta, mu, market):
        # basin_matrix is a adjacent matrix (np.array) of a directed graph for a basin outflow
        # e.g. for a Y-shape river system, the basin_matrix is
        # [[Q_11, 0, Q_13, 0],
        #  [0, Q_22, Q_23, 0],
        #  [0, 0, Q_33, Q_34],
        #  [0, 0, 0, Q_44]]
        super().__init__()
        self.schedule = MarketActivation(self)
        self.user_amount = basin_matrix.shape[0]
        self.basin_matrix = basin_matrix

        sample_size = np.random.randint(self.user_amount, size=self.user_amount)

        for i in range(0, self.user_amount):
            water_user = WaterUser(unique_id=i, model=self, x=0,
                                   u_a=u[i][0], u_b=u[i][1], u_c=u[i][2],
                                   w=water_permit[i], L=basin_matrix[i][i],
                                   outflow=basin_matrix[i], inflow=basin_matrix.transpose()[i],
                                   res=res[i],
                                   transaction_size=sample_size[i],
                                   beta=beta[i], mu=mu[i])
            self.schedule.add(water_user)

        self.market = market
        self.users = self.schedule.agents
        self.users_keys = list(self.schedule.agents_keys)
        # p_matrix[i][j] is the transaction price between agent i and agent j
        # p_matrix[i][j] = 0 if no transaction happens
        self.p_matrix = np.zeros((self.user_amount, self.user_amount))
        self.p_old = np.zeros((self.user_amount, self.user_amount))
        # a_matrix[i][j] is the transaction amount between agent i and agent j
        # a_matrix[i][j] is positive if i is the buyer and j is the seller
        # a_matrix[i][j] = 0 if no transaction happens
        self.a_matrix = np.zeros((self.user_amount, self.user_amount))

        self.role = np.array([user.market_role for user in self.users])
        self.x = [user.x for user in self.users]

        self.running = True

    def step(self):
        print(self.p_old)
        self.schedule.step()  # user.step() for all users in self.users
        flag = self.check()
        if not flag:
            self.transaction()
            if self.market == 'discriminatory-price':
                self.schedule.learn_d(self.p_matrix)
            if self.schedule.time % 10 == 1:
                p_new = copy.deepcopy(self.p_matrix)  # deep copy: allocation a new address
                p_delta = p_new - self.p_old
                if np.sum(np.abs(p_delta)) < 10**(-8):
                    self.running = False
                else:
                    self.p_old = p_new
        else:
            self.running = False

    def transaction(self):
        print(self.role)
        #print(self.x)
        # if the market is the discriminatory-price double auction market
        if self.market == 'discriminatory-price':
            buyer_price = np.array([user.bid_price for user in self.users if user.market_role == 'buyer'])
            b_index = np.array([user.unique_id for user in self.users if user.market_role == 'buyer'])
            #print(buyer_price)
            buyer_amount = np.array([user.bid_amount for user in self.users if user.market_role == 'buyer'])

            seller_price = np.array([user.bid_price for user in self.users if user.market_role == 'seller'])
            s_index = np.array([user.unique_id for user in self.users if user.market_role == 'seller'])
            #print(seller_price)
            seller_amount = np.array([user.bid_amount for user in self.users if user.market_role == 'seller'])

            # for buyers, we sort it by their price offers in descending order
            buyer_sort = -buyer_price
            buyer_order = buyer_sort.argsort()
            # for sellers, we sort it by their price offers in ascending order
            seller_order = seller_price.argsort()

            j = 0  # j is the seller order
            for i in buyer_order:  # i is the buyer order in buyer_price/amount
                while buyer_amount[i] > 0 and j<seller_order.shape[0]:
                    buyer_id = b_index[i]  # i is the unique_id of ith buyer
                    j_index = seller_order[j]  # seller_order[j] is the index of jth seller in seller_price/amount
                    seller_id = s_index[j_index]  # j is the unique_id of jth seller
                    if buyer_price[i] > seller_price[j_index]:  # the jth seller complete its transaction
                        cost = 0.5*(buyer_price[i]+seller_price[j_index])
                        self.p_matrix[buyer_id][seller_id] = cost
                        self.p_matrix[seller_id][buyer_id] = cost
                        if buyer_amount[i] >= seller_amount[j_index]:
                            buyer_amount[i] -= seller_amount[j_index]
                            seller_amount[j_index] = 0
                            # self.users[seller].market_role = 'sider'
                            role_update(self)
                            j += 1
                        else:   # the ith buyer complete its transaction
                            seller_amount[j_index] -= buyer_amount[i]
                            buyer_amount[i] = 0
                            # self.users[buyer].market_role = 'sider'
                            role_update(self)
                    else:  # transaction fail
                        break
            print(self.schedule.time)  # iteration times
            #print(self.p_matrix)
        # if the market is the bilateral negotiation market
        elif self.market == 'bilateral negotiations':
            pass

    def check(self):
        limit_check(self)
        # all sider
        if np.sum(self.role=='sider') == self.user_amount:
            return True
        # only seller and sider in the market
        elif np.sum(self.role=='seller') + np.sum(self.role=='sider') == self.user_amount:
            # randomly choose a user until he is NOT an 'over' user
            # because he can't use such much water
            while True:
                index = np.random.randint(0, self.user_amount)
                if self.users[index].label != 'over':
                    break
            ratio = np.random.uniform(1, 1.5)
            self.users[index].x = min(self.users[index].permit*ratio, self.users[index].limit)
            self.users[index].market_role = 'buyer'
            role_update(self)
            x_update(self)
            self.users[index].buy()
        # only buyer and sider in the market
        elif np.sum(self.role == 'buyer') + np.sum(self.role == 'sider') == self.user_amount:
            # randomly choose a user until he is an 'over' user
            # because he has the motivation to sell his extra water
            while True:
                index = np.random.randint(0, self.user_amount)
                if self.users[index].label == 'over':
                    break
            ratio = np.random.uniform(0.5, 1)
            self.users[index].x = self.users[index].permit * ratio
            self.users[index].market_role = 'seller'
            role_update(self)
            x_update(self)
            self.users[index].sell()
        sum_x = np.sum(self.x)
        sum_w = np.sum([user.permit for user in self.users])
        # print(sum_x - sum_w)
        if sum_x == sum_w:
            return True
        else:
            return False


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


def flow_update(market):
    i = 0
    for user in market.users:
        market.f_matrix[i] = user.outflow
        i += 1


def label_get(market, label):
    l_list = [user.unique_id for user in market.users if user.label == label]
    return l_list


class WaterMarket(Model):

    def __init__(self, basin_matrix, precipitation, out_min, penalty, res, u, water_permit, beta, mu, market):
        # basin_matrix is a adjacent matrix (np.array) of a directed graph for a waterway
        # e.g. for a Y-shape river system, the basin_matrix is
        # [[0, 0, 1, 0],
        #  [0, 0, 1, 0],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        super().__init__()
        self.user_amount = basin_matrix.shape[0]
        self.basin_matrix = basin_matrix

        sample_size = np.random.randint(self.user_amount, size=self.user_amount)
        x_initial = [-u_i[1]/(2*u_i[0]) for u_i in u]
        # f_matrix[i][j] is the flow from agent i to agent j
        self.f_matrix = np.diag(precipitation)

        self.schedule = MarketActivation(self)
        for i in range(0, self.user_amount):
            water_user = WaterUser(unique_id=i, model=self,
                                   u_a=u[i][0], u_b=u[i][1], u_c=u[i][2], x=x_initial[i],
                                   w=water_permit[i], L=basin_matrix[i][i],
                                   out_link=np.nonzero(basin_matrix[i])[0], in_link=np.nonzero(basin_matrix.transpose()[i])[0],
                                   out_min=out_min[i], penalty=penalty[i], res=res[i],
                                   transaction_size=sample_size[i], beta=beta[i], mu=mu[i])
            self.schedule.add(water_user)

        self.schedule.agent_count()
        self.market = market
        self.users = self.schedule.agents
        self.users_keys = list(self.schedule.agents_keys)

        # p_matrix[i][j] is the transaction price between agent i and agent j
        # p_matrix[i][j] = 0 if no transaction happens
        self.p_matrix = np.zeros((self.user_amount, self.user_amount))
        self.p_old = np.zeros((self.user_amount, self.user_amount))

        # a_matrix[i][j] is the transaction amount from agent i to agent j
        # a_matrix[i][j] is positive if i is the buyer and j is the seller
        # a_matrix[i][j] = 0 if no transaction happens
        self.a_matrix = np.zeros((self.user_amount, self.user_amount))
        # initialize the outflow
        for user in self.users:
            user.water_table()
            user.outflow_initialize()

        self.role = np.array([user.market_role for user in self.users])
        self.x = [user.x for user in self.users]
        self.running = True

    def step(self):
        self.schedule.step()  # user.step() for all users in self.users
        flag = self.check()  # check if all sider, only sider and buyer, or only sider and seller
        if not flag:
            self.transaction()
            self.schedule.benefit(self.p_matrix, self.a_matrix)
            if self.market == 'discriminatory-price':
                self.schedule.learn_d(self.p_matrix)
            if self.schedule.time % 50 == 49:  # if choose 1, easily stop at time 1
                p_new = copy.deepcopy(self.p_matrix)  # deep copy
                p_delta = p_new - self.p_old
                if np.sum(np.abs(p_delta)) < 10**(-8):
                    self.running = False
                else:
                    self.p_old = p_new
        else:
            self.running = False

    def transaction(self):
        print(self.role)
        self.p_matrix = np.zeros((self.user_amount, self.user_amount))
        self.a_matrix = np.zeros((self.user_amount, self.user_amount))
        # if the market is the discriminatory-price double auction market
        if self.market == 'discriminatory-price':
            buyer_price = np.array([user.bid_price for user in self.users if user.market_role == 'buyer'])
            b_index = np.array([user.unique_id for user in self.users if user.market_role == 'buyer'])
            print(buyer_price)
            buyer_amount = np.array([user.bid_amount for user in self.users if user.market_role == 'buyer'])

            seller_price = np.array([user.bid_price for user in self.users if user.market_role == 'seller'])
            s_index = np.array([user.unique_id for user in self.users if user.market_role == 'seller'])
            print(seller_price)
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
                    if buyer_price[i] > seller_price[j_index]:
                        # update the p_matrix
                        price = 0.5*(buyer_price[i]+seller_price[j_index])
                        self.p_matrix[buyer_id][seller_id] = price
                        self.p_matrix[seller_id][buyer_id] = price
                        # update the a_matrix
                        amount = min(buyer_amount[i],seller_amount[j_index])
                        self.a_matrix[buyer_id][seller_id] = amount
                        self.a_matrix[seller_id][buyer_id] = -amount
                        buyer_amount[i] -= amount
                        seller_amount[j_index] -= amount

                        # update users' property
                        self.users[buyer_id].step()
                        self.users[seller_id].step()
                        # update market_role
                        role_update(self)
                        if amount == seller_amount[j_index]:
                            j += 1  # turn to the next seller
                    else:  # transaction fail
                        break
            print(self.schedule.time)  # iteration times
            print(self.p_matrix)
            print(self.f_matrix)
        # if the market is the bilateral negotiation market
        elif self.market == 'bilateral negotiations':
            pass

    def check(self):
        # all sider
        if np.sum(self.role=='sider') == self.user_amount:
            return True
        # only seller and sider in the market
        elif np.sum(self.role=='seller') + np.sum(self.role=='sider') == self.user_amount:
            # randomly choose a user until he is NOT an 'over' user
            # because he need water permit
            list = label_get(self, 'normal')
            list_l = len(list)
            if list_l == 0:
                num = self.user_amount
                index = np.random.randint(0, num)
                ratio = np.random.uniform(1, 1.5)
                self.users[index].x = min(self.users[index].permit*ratio, self.users[index].limit)
                self.users[index].step()
                role_update(self)
                x_update(self)
            else:
                num = list_l
                index = list[np.random.randint(0, num)]
                ratio = np.random.uniform(1, 1.5)
                self.users[index].x = min(self.users[index].permit*ratio, self.users[index].limit)
                self.users[index].step()
                role_update(self)
                x_update(self)
        # only buyer and sider in the market
        elif np.sum(self.role == 'buyer') + np.sum(self.role == 'sider') == self.user_amount:
            # randomly choose a user
            num = self.user_amount
            index = np.random.randint(0, num)
            ratio = np.random.uniform(0.5, 1)
            self.users[index].x = self.users[index].permit * ratio
            self.users[index].step()
            role_update(self)
            x_update(self)
        # if we have buyer, seller and sider, just update the role and water use after every agent steps
        role_update(self)
        x_update(self)
        # check if the water permits are fully used
        sum_x = np.sum(self.x)
        sum_w = np.sum([user.permit for user in self.users])
        if sum_x == sum_w:
            return True
        else:
            return False


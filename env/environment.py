import numpy as np
from env.loader import Loader


class PortfolioEnv:

    def __init__(self, start_date=None, end_date=None, action_scale=1):
        self.loader = Loader()
        self.historical_data = self.loader.load(start_date, end_date)
        self.n_stocks = len(self.historical_data)
        self.prices = np.zeros(self.n_stocks)
        self.shares = np.zeros(self.n_stocks).astype(np.int)
        self.balance = 0
        self.current_row = 0
        self.end_row = 0
        self.action_scale = action_scale

    def state_shape(self):
        return 2 * self.n_stocks + 1,

    def n_actions(self):
        return self.n_stocks

    def reset(self, start_date=None, end_date=None, initial_balance=1000000):
        if start_date is None:
            self.current_row = 0
        else:
            self.current_row = self.historical_data[0].index.get_loc(start_date)
        if end_date is None:
            self.end_row = self.historical_data[0].index.size - 1
        else:
            self.end_row = self.historical_data[0].index.get_loc(end_date)
        self.prices = self.get_prices()
        self.shares = np.zeros(self.n_stocks).astype(np.int)
        self.balance = initial_balance

        return self.get_state()

    def get_prices(self):
        return np.array([stock['Adj Close'][self.current_row] for stock in self.historical_data])

    def get_state(self):
        return [self.balance] + self.prices.tolist() + self.shares.tolist()

    def is_finished(self):
        return self.current_row == self.end_row

    def get_date(self):
        return self.historical_data[0].index[self.current_row]

    def get_wealth(self):
        return self.prices.dot(self.shares) + self.balance

    def step(self, action):
        actions = np.maximum(np.round(np.array(action) * self.action_scale), -self.shares)
        cost = self.prices.dot(actions)
        if cost > self.balance:
            actions = np.floor(actions * self.balance / cost)
            cost = self.prices.dot(actions)
        self.shares = self.shares + actions.astype(np.int)
        self.balance -= cost
        self.current_row += 1
        new_prices = self.get_prices()
        reward = (new_prices - self.prices).dot(self.shares)
        self.prices = new_prices

        return self.get_state(), reward, self.is_finished(), self.get_date(), self.get_wealth()

    # def step(self, action):
    #     actions = np.clip(action, -1, +1)
    #     positive = actions > 0
    #     negative = actions < 0
    #     actions[negative] = np.ceil(actions[negative] * self.shares[negative])
    #     k = (self.balance - np.sum(self.prices[negative] * actions[negative])) \
    #         / np.sum(self.prices[positive] * actions[positive])
    #     actions[positive] = np.floor(actions[positive] * k)
    #     # print("positive coefficient", k)
    #     # print("new actions:", actions.tolist())
    #     cost = self.prices.dot(actions)
    #     self.shares = self.shares + actions
    #     self.balance -= cost
    #     self.current_row += 1
    #     new_prices = self.get_prices()
    #     reward = (new_prices - self.prices).dot(self.shares)
    #     self.prices = new_prices
    #
    #     return self.get_state(), reward, self.is_finished(), self.get_date(), self.get_wealth()

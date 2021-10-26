import numpy as np
import torch as T
import torch.nn.functional as F
from env.loader import Loader
from finta import TA
import pandas as pd


class PortfolioEnv:

    def __init__(self, start_date=None, end_date=None, action_scale=1, action_interpret='portfolio',
                 state_type='only prices', djia_year=2019):
        self.loader = Loader(djia_year=djia_year)
        self.historical_data = self.loader.load(start_date, end_date)
        for stock in self.historical_data:
            stock['MA20'] = TA.SMA(stock, 20)
            stock['MA50'] = TA.SMA(stock, 50)
            stock['MA200'] = TA.SMA(stock, 200)
            stock['ATR'] = TA.ATR(stock)
        self.n_stocks = len(self.historical_data)
        self.prices = np.zeros(self.n_stocks)
        self.shares = np.zeros(self.n_stocks).astype(np.int)
        self.balance = 0
        self.current_row = 0
        self.end_row = 0
        self.action_scale = action_scale
        self.action_interpret = action_interpret
        self.state_type = state_type

    def state_shape(self):
        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            return self.n_stocks,
        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            return 6 * self.n_stocks,
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            return 2 * self.n_stocks + 1,
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            return 7 * self.n_stocks + 1,

    def action_shape(self):
        if self.action_interpret == 'portfolio':
            return self.n_stocks + 1,
        if self.action_interpret == 'transactions':
            return self.n_stocks,

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

        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            return self.prices.tolist()

        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            state = []
            for stock in self.historical_data:
                state.extend(stock[['Adj Close', 'MA20', 'MA50', 'MA200', 'ATR', 'Volume']].iloc[self.current_row])
            return np.array(state)
        
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            return [self.balance] + self.prices.tolist() + self.shares.tolist()
        
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            state = [self.balance] + self.shares.tolist()
            for stock in self.historical_data:
                state.extend(stock[['Adj Close', 'MA20', 'MA50', 'MA200', 'ATR', 'Volume']].iloc[self.current_row])
            return np.array(state)

    def is_finished(self):
        return self.current_row == self.end_row

    def get_date(self):
        return self.historical_data[0].index[self.current_row]

    def get_wealth(self):
        return self.prices.dot(self.shares) + self.balance
    
    def get_balance(self):
        return self.balance
    
    def get_shares(self):
        return self.shares

    def buy_hold_history(self, start_date=None, end_date=None):
        if start_date is None:
            start_row = 0
        else:
            start_row = self.historical_data[0].index.get_loc(start_date)
        if end_date is None:
            end_row = self.historical_data[0].index.size - 1
        else:
            end_row = self.historical_data[0].index.get_loc(end_date)
        
        values = [sum([stock['Adj Close'][row] for stock in self.historical_data])
                  for row in range(start_row, end_row + 1)]
        dates = self.historical_data[0].index[start_row:end_row+1]

        return pd.Series(values, index=dates)

    def get_intervals(self, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        index = self.historical_data[0].index

        if self.state_type == 'only prices':
            size = len(index)
            train_begin = 0
            train_end = int(np.round(train_ratio * size - 1))
            valid_begin = train_end + 1
            valid_end = valid_begin + int(np.round(valid_ratio * size - 1))
            test_begin = valid_end + 1
            test_end = -1
        
        if self.state_type == 'indicators':
            size = len(index) - 199
            train_begin = 199
            train_end = train_begin + int(np.round(train_ratio * size - 1))
            valid_begin = train_end + 1
            valid_end = valid_begin + int(np.round(valid_ratio * size - 1))
            test_begin = valid_end + 1
            test_end = -1
        
        intervals = {'training': (index[train_begin], index[train_end]),
             'validation': (index[valid_begin], index[valid_end]),
             'testing': (index[test_begin], index[test_end])}

        return intervals

    def step(self, action, softmax=True):
        
        if self.action_interpret == 'portfolio':
            current_wealth = self.get_wealth()
            if softmax:
                action = F.softmax(T.tensor(action, dtype=T.float), -1).numpy()
            else:
                action = np.array(action)
            new_shares = np.floor(current_wealth * action[1:] / self.prices)
            actions = new_shares - self.shares
            cost = self.prices.dot(actions)
            self.shares = self.shares + actions.astype(np.int)
            self.balance -= cost
            self.current_row += 1
            new_prices = self.get_prices()
            reward = (new_prices - self.prices).dot(self.shares)
            self.prices = new_prices
        
        if self.action_interpret == 'transactions':
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

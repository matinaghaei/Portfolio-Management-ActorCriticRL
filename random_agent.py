#%%
import warnings
warnings.filterwarnings('ignore')

from env.environment import PortfolioEnv
import numpy as np
import pandas as pd
from pyfolio import timeseries

env = PortfolioEnv(state_type='indicators')
action_shape = env.action_shape()
intervals = env.get_intervals()
buy_hold_history = env.buy_hold_history(*intervals['testing'])
stats = []

iteration = 0
while iteration < 50:
    print(f'interation {iteration}')
    observation = env.reset(*intervals['testing'])
    wealth_history = [env.get_wealth()]
    done = False
    while not done:
        action = np.eye(*action_shape)[np.random.choice(*action_shape)]
        observation_, reward, done, info, wealth = env.step(action, softmax=False)
        observation = observation_
        # print(f"random - Date: {info.date()},\tBalance: {int(env.get_balance())},\t"
        #       f"Cumulative Return: {int(wealth) - 1000000},\tShares: {env.get_shares()}")
        wealth_history.append(wealth)

    returns = pd.Series(wealth_history, buy_hold_history.index).pct_change().dropna()
    stats.append(timeseries.perf_stats(returns))

    iteration += 1

#%%
annual = [data['Annual return'] for data in stats]
print(f'Annual return\tmean: {np.mean(annual)}, std: {np.std(annual)}')
sharp = [data['Sharpe ratio'] for data in stats]
print(f'Sharpe ratio\tmean: {np.mean(sharp)}, std: {np.std(sharp)}')
drawdown = [data['Max drawdown'] for data in stats]
print(f'Max drawdown\tmean: {np.mean(drawdown)}, std: {np.std(drawdown)}')

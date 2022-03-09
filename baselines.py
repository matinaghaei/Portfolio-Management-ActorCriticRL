import warnings
warnings.filterwarnings('ignore')

from env.environment import PortfolioEnv
import plot
from plot import add_curve, save_plot
import os
import pandas as pd
from pyfolio import timeseries
from pypfopt import EfficientFrontier, risk_models, expected_returns

plot.initialize()


def test(env, weights, name):
    intervals = env.get_intervals()

    return_history = [0]

    env.reset(*intervals['testing'])
    wealth_history = [env.get_wealth()]
    done = False
    while not done:
        observation_, reward, done, info, wealth = env.step(weights, softmax=False)
        # print(f"random - Date: {info.date()},\tBalance: {int(env.get_balance())},\t"
        #       f"Cumulative Return: {int(wealth) - 1000000},\tShares: {env.get_shares()}")
        return_history.append(wealth - 1000000)
        wealth_history.append(wealth)

    add_curve(return_history, name)

    returns = pd.Series(wealth_history, buy_hold_history.index).pct_change().dropna()
    stats = timeseries.perf_stats(returns)
    stats.to_csv(f'plots/{name}_perf.csv')
    

file = open(f'env/data/DJIA_2019/tickers.txt', 'r')
tickers = [line.strip() for line in file.readlines()]

table = pd.DataFrame()
for i in range(len(tickers)):
    data = pd.read_csv(f'env/data/DJIA_2019/ticker_{tickers[i]}.csv', parse_dates=True, index_col='Date')
    table[data['ticker'][0]] = data['Adj Close']

env = PortfolioEnv(state_type='indicators')
intervals = env.get_intervals()
start = table.index.get_loc(intervals['training'][0])
end = table.index.get_loc(intervals['training'][1])
train_set = table[start:end+1]

buy_hold_history = env.buy_hold_history(*intervals['testing'])
add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy&Hold')

mu = expected_returns.mean_historical_return(train_set)
S = risk_models.sample_cov(train_set)

ef = EfficientFrontier(mu, S)
weights = [0] + list(ef.max_sharpe().values())
test(env, weights, 'Max-Sharpe')

ef = EfficientFrontier(mu, S)
weights = [0] + list(ef.min_volatility().values())
test(env, weights, 'Min-Volatility')

save_plot('plots/baselines_testing.png',
            title=f"Testing - {intervals['testing'][0].date()} to {intervals['testing'][1].date()}",
            x_label='Days', y_label='Cumulative Return (Dollars)')

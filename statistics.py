#%%
import numpy as np
import pandas as pd

a2c_stats = [pd.read_csv(f'plots/a2c/{i}3_perf.csv', index_col=0) for i in range(50)]
ddpg_stats = [pd.read_csv(f'plots/ddpg/{i}3_perf.csv', index_col=0) for i in range(50)]
ppo_stats = [pd.read_csv(f'plots/ppo/{i}3_perf.csv', index_col=0) for i in range(50)]

#%%
print('A2C')
a2c_annual = [data.loc['Annual return'][0] for data in a2c_stats]
print(f'Annual return\tmean: {np.mean(a2c_annual)}, std: {np.std(a2c_annual)}')
a2c_sharp = [data.loc['Sharpe ratio'][0] for data in a2c_stats]
print(f'Sharpe ratio\tmean: {np.mean(a2c_sharp)}, std: {np.std(a2c_sharp)}')
a2c_drawdown = [data.loc['Max drawdown'][0] for data in a2c_stats]
print(f'Max drawdown\tmean: {np.mean(a2c_drawdown)}, std: {np.std(a2c_drawdown)}')
print()

#%%
print('DDPG')
ddpg_annual = [data.loc['Annual return'][0] for data in ddpg_stats]
print(f'Annual return\tmean: {np.mean(ddpg_annual)}, std: {np.std(ddpg_annual)}')
ddpg_sharp = [data.loc['Sharpe ratio'][0] for data in ddpg_stats]
print(f'Sharpe ratio\tmean: {np.mean(ddpg_sharp)}, std: {np.std(ddpg_sharp)}')
ddpg_drawdown = [data.loc['Max drawdown'][0] for data in ddpg_stats]
print(f'Max drawdown\tmean: {np.mean(ddpg_drawdown)}, std: {np.std(ddpg_drawdown)}')
print()

#%%
print('PPO')
ppo_annual = [data.loc['Annual return'][0] for data in ppo_stats]
print(f'Annual return\tmean: {np.mean(ppo_annual)}, std: {np.std(ppo_annual)}')
ppo_sharp = [data.loc['Sharpe ratio'][0] for data in ppo_stats]
print(f'Sharpe ratio\tmean: {np.mean(ppo_sharp)}, std: {np.std(ppo_sharp)}')
ppo_drawdown = [data.loc['Max drawdown'][0] for data in ppo_stats]
print(f'Max drawdown\tmean: {np.mean(ppo_drawdown)}, std: {np.std(ppo_drawdown)}')
print()

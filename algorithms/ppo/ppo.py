from env.environment import PortfolioEnv
from algorithms.ppo.agent import Agent
from plot import add_curve, add_hline, save_plot
import os
import pandas as pd
from pyfolio import timeseries


class PPO:

    def __init__(self, load=False, alpha=0.0003, n_epochs=10,
                 batch_size=64, layer1_size=512, layer2_size=512, t_max=256,
                 state_type='only prices', djia_year=2019, repeat=0, entropy=1e-2):

        self.figure_dir = 'plots/ppo'
        self.checkpoint_dir = 'checkpoints/ppo'
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.t_max = t_max
        self.repeat = repeat

        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        elif djia_year == 2012:
            self.intervals = self.env.get_intervals(train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05)
        self.agent = Agent(action_dims=self.env.action_shape(), batch_size=batch_size, alpha=alpha,
                           n_epochs=n_epochs, input_dims=self.env.state_shape(),
                           fc1_dims=layer1_size, fc2_dims=layer2_size, entropy=entropy)

        if load:
            self.agent.load_models(self.checkpoint_dir)

    def train(self, verbose=False):
        training_history = []
        validation_history = []
        iteration = 1
        max_wealth = 0

        while True:
            n_steps = 0
            observation = self.env.reset(*self.intervals['training'])
            done = False
            while not done:
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, info, wealth = self.env.step(action)
                n_steps += 1
                self.agent.remember(observation, action, prob, val, reward, done)
                if n_steps % self.t_max == 0:
                    self.agent.learn()
                observation = observation_
                if verbose:
                    print(f"PPO training - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                          f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            self.agent.memory.clear_memory()

            print(f"PPO training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"PPO validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
            validation_history.append(validation_wealth - 1000000)
            if validation_wealth > max_wealth:
                self.agent.save_models(self.checkpoint_dir)
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-5:].count(max_wealth - 1000000) != 1:
                break
            iteration += 1

        self.agent.load_models(self.checkpoint_dir)

        buy_hold_history = self.env.buy_hold_history(*self.intervals['training'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(training_history, 'PPO')
        save_plot(filename=self.figure_dir + f'/{self.repeat}0_training.png',
                  title=f"Training - {self.intervals['training'][0].date()} to {self.intervals['training'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(validation_history, 'PPO')
        save_plot(filename=self.figure_dir + f'/{self.repeat}1_validation.png',
                  title=f"Validation - {self.intervals['validation'][0].date()} to {self.intervals['validation'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
            if verbose:
                print(f"PPO validation - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                      f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
        return wealth

    def test(self, verbose=True):
        return_history = [0]
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy&Hold')
        n_steps = 0

        observation = self.env.reset(*self.intervals['testing'])
        wealth_history = [self.env.get_wealth()]
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            n_steps += 1
            self.agent.remember(observation, action, prob, val, reward, done)
            if n_steps % self.t_max == 0:
                self.agent.learn()
            observation = observation_
            if verbose:
                print(f"PPO testing - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                    f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            return_history.append(wealth - 1000000)
            wealth_history.append(wealth)
        self.agent.memory.clear_memory()

        add_curve(return_history, 'PPO')
        save_plot(self.figure_dir + f'/{self.repeat}2_testing.png',
                  title=f"Testing - {self.intervals['testing'][0].date()} to {self.intervals['testing'][1].date()}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

        returns = pd.Series(wealth_history, buy_hold_history.index).pct_change().dropna()
        stats = timeseries.perf_stats(returns)
        stats.to_csv(self.figure_dir + f'/{self.repeat}3_perf.csv')

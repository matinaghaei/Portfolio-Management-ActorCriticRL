from env.environment import PortfolioEnv
from algorithms.a2c.agent import ActorCritic, Agent
from torch.multiprocessing import Pipe, Lock
from plot import add_curve, add_hline, save_plot
import os
import pandas as pd
from pyfolio import timeseries


class A2C:

    def __init__(self, n_agents, load=False, alpha=1e-3, gamma=0.99,
                 layer1_size=512, layer2_size=512, t_max=64,
                 state_type='only prices', djia_year=2019, repeat=0, entropy=1e-4):

        self.n_agents = n_agents
        self.figure_dir = 'plots/a2c'
        self.checkpoint_dir = 'checkpoints/a2c'
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.t_max = t_max
        self.state_type = state_type
        self.djia_year = djia_year
        self.repeat = repeat

        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        elif djia_year == 2012:
            self.intervals = self.env.get_intervals(train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05)
        self.network = ActorCritic(input_dims=self.env.state_shape(), action_dims=self.env.action_shape(),
                                   gamma=gamma, fc1_dims=layer1_size, fc2_dims=layer2_size, lr=alpha, entropy=entropy)
        self.network.share_memory()
        self.network.train()
        
        if load:
            self.network.load_checkpoint(self.checkpoint_dir)

    def train(self, verbose=False):
        training_history = [[] for i in range(self.n_agents)]
        validation_history = []
        iteration = 1
        max_wealth = 0

        while True:
            pipes = [Pipe() for i in range(self.n_agents)]
            local_conns, remote_conns = list(zip(*pipes))
            lock = Lock()
            workers = [Agent(self.network,
                             self.intervals['training'],
                             conn=remote_conns[i],
                             lock=lock,
                             name=f'worker {i}',
                             t_max=self.t_max,
                             verbose=verbose,
                             state_type=self.state_type,
                             djia_year=self.djia_year) for i in range(self.n_agents)]
            [w.start() for w in workers]

            self.network.done = False
            while not self.network.done:
                losses = []
                for c in local_conns:
                    self.network.set_memory(*c.recv())
                    losses.append(self.network.calc_loss())
                total_loss = sum(losses)
                self.network.zero_grad()
                total_loss.backward()
                self.network.optimizer.step()
                [c.send('resume') for c in local_conns]

            wealth = [c.recv() for c in local_conns]
            for i in range(self.n_agents):
                print(f"A2C training - worker {i} - Iteration: {iteration},\t"
                      f"Cumulative Return: {int(wealth[i]) - 1000000}")
                training_history[i].append(wealth[i] - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"A2C validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
            validation_history.append(validation_wealth - 1000000)
            if validation_wealth > max_wealth:
                self.network.save_checkpoint(self.checkpoint_dir)
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-5:].count(max_wealth - 1000000) != 1:
                break
            iteration += 1

        self.network.load_checkpoint(self.checkpoint_dir)

        buy_hold_history = self.env.buy_hold_history(*self.intervals['training'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        for i in range(self.n_agents):
            add_curve(training_history[i], f'A2C - worker {i}')
        save_plot(filename=self.figure_dir + f'/{self.repeat}0_training.png',
                  title=f"Training - {self.intervals['training'][0].date()} to {self.intervals['training'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(validation_history, 'A2C')
        save_plot(filename=self.figure_dir + f'/{self.repeat}1_validation.png',
                  title=f"Validation - {self.intervals['validation'][0].date()} to {self.intervals['validation'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action = self.network.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
            if verbose:
                print(f"A2C validation - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                      f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
        return wealth

    def test(self, verbose=True):
        return_history = [0]
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy&Hold')

        done = False
        observation = self.env.reset(*self.intervals['testing'])
        wealth_history = [self.env.get_wealth()]
        t_step = 0
        while not done:
            action = self.network.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            self.network.remember(observation, action, reward)
            if t_step % self.t_max == 0 or done:
                loss = self.network.calc_loss(done)
                self.network.zero_grad()
                loss.backward()
                self.network.optimizer.step()
                self.network.clear_memory()
            t_step += 1
            observation = observation_
            if verbose:
                print(f"A2C testing - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                    f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            return_history.append(wealth - 1000000)
            wealth_history.append(wealth)

        add_curve(return_history, 'A2C')
        save_plot(self.figure_dir + f'/{self.repeat}2_testing.png',
                  title=f"Testing - {self.intervals['testing'][0].date()} to {self.intervals['testing'][1].date()}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

        returns = pd.Series(wealth_history, buy_hold_history.index).pct_change().dropna()
        stats = timeseries.perf_stats(returns)
        stats.to_csv(self.figure_dir + f'/{self.repeat}3_perf.csv')

from env.environment import PortfolioEnv
from algorithms.ddpg.agent import Agent
import numpy as np
from plot import add_curve, add_hline, save_plot
import os


class DDPG:

    def __init__(self, load=False, alpha=0.000025, beta=0.00025, tau=0.001,
                 batch_size=64, layer1_size=400, layer2_size=300, layer3_size=None,
                 action_interpret='portfolio', state_type='only prices', djia_year=2019,
                 train_eval='new', bn_drop='only bn', action_input_layer='nothing',
                 state_activation=False, action_activation=True, repeat=0):

        self.figure_dir = f'plots/ddpg/{train_eval}_{bn_drop}_{action_input_layer}_{state_activation}_{action_activation}'
        os.makedirs(self.figure_dir, exist_ok=True)
        self.repeat = repeat
        self.env = PortfolioEnv(action_scale=1000, action_interpret=action_interpret,
                                state_type=state_type, djia_year=djia_year)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        elif djia_year == 2012:
            self.intervals = self.env.get_intervals(train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05)
        self.agent = Agent(alpha=alpha, beta=beta, input_dims=self.env.state_shape(), 
                           action_dims=self.env.action_shape(), tau=tau, batch_size=batch_size, 
                           layer1_size=layer1_size, layer2_size=layer2_size, layer3_size=layer3_size,
                           action_interpret=action_interpret, train_eval=train_eval, bn_drop=bn_drop,
                           action_input_layer=action_input_layer, state_activation=state_activation,
                           action_activation=action_activation)
        if load:
            self.agent.load_models()

        np.random.seed(0)

    def train(self, verbose=False):
        training_history = []
        validation_history = []
        iteration = 1
        max_wealth = 0

        while True:
            observation = self.env.reset(*self.intervals['training'])
            done = False
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info, wealth = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, int(done))
                self.agent.learn()
                observation = observation_
                if verbose:
                    print(f"DDPG training - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                          f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            self.agent.memory.clear_buffer()

            print(f"DDPG training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"DDPG validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
            validation_history.append(validation_wealth - 1000000)
            if validation_wealth > max_wealth:
                self.agent.save_models()
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-5:].count(max_wealth - 1000000) == 0:
                break
            # if iteration == 10:
            #     break
            iteration += 1

        self.agent.load_models()

        buy_hold_history = self.env.buy_hold_history(*self.intervals['training'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(training_history, 'DDPG')
        save_plot(filename=self.figure_dir + f'/training_{self.repeat}.png',
                  title=f"Training - {self.intervals['training'][0]} to {self.intervals['training'][1]}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(validation_history, 'DDPG')
        save_plot(filename=self.figure_dir + f'/validation_{self.repeat}.png',
                  title=f"Validation - {self.intervals['validation'][0]} to {self.intervals['validation'][1]}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
            if verbose:
                print(f"DDPG validation - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                      f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
        return wealth

    def test(self, verbose=True):
        return_history = []
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy&Hold')

        observation = self.env.reset(*self.intervals['testing'])
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            self.agent.remember(observation, action, reward, observation_, int(done))
            self.agent.learn()
            observation = observation_
            if verbose:
                print(f"DDPG testing - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                    f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            return_history.append(wealth - 1000000)
        self.agent.memory.clear_buffer()

        add_curve(return_history, 'DDPG')
        save_plot(self.figure_dir + f'/testing_{self.repeat}.png',
                  title=f"Testing - from {self.intervals['testing'][0]} to {self.intervals['testing'][1]}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

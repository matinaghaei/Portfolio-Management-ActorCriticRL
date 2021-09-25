from env.environment import PortfolioEnv
from algorithms.ddpg.agent import Agent
import numpy as np
from plot import add_curve, save_plot


class DDPG:

    def __init__(self, intervals, load=False, alpha=0.000025, beta=0.00025, tau=0.001,
                 batch_size=64, layer1_size=400, layer2_size=300):

        self.intervals = intervals
        self.figure_dir = 'plots/ddpg'
        self.env = PortfolioEnv(action_scale=1000)
        self.agent = Agent(alpha=alpha, beta=beta, input_dims=self.env.state_shape(), tau=tau,
                           batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                           n_actions=self.env.n_actions())
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
                    print(f"DDPG training - Date: {info.date()},\tBalance: {int(observation[0])},\t"
                          f"Cumulative Return: {int(wealth) - 1000000},\tShares: {observation[31:61]}")
            self.agent.memory.clear_buffer()

            print(f"DDPG training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"DDPG validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
            validation_history.append(validation_wealth - 1000000)
            if validation_wealth > max_wealth:
                self.agent.save_models()
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-2:].count(max_wealth - 1000000) == 0:
                break
            iteration += 1

        self.agent.load_models()

        buy_hold_history = self.env.buy_hold_history(*self.intervals['training'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_curve([buy_hold_final for i in range(iteration)], 'Buy & Hold')
        add_curve(training_history, 'DDPG training')
        save_plot(filename=self.figure_dir + '/training.png',
                  x_label='Iterations', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_curve([buy_hold_final for i in range(iteration)], 'Buy & Hold')
        add_curve(validation_history, 'DDPG validation')
        save_plot(filename=self.figure_dir + '/validation.png',
                  x_label='Iterations', y_label='Cumulative Return (Dollars)')

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
            if verbose:
                print(f"DDPG validation - Date: {info.date()},\tBalance: {int(observation[0])},\t"
                      f"Cumulative Return: {int(wealth) - 1000000},\tShares: {observation[31:61]}")
        return wealth

    def test(self):
        return_history = []
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy & Hold')

        observation = self.env.reset(*self.intervals['testing'])
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            self.agent.remember(observation, action, reward, observation_, int(done))
            self.agent.learn()
            observation = observation_

            print(f"DDPG testing - Date: {info.date()},\tBalance: {int(observation[0])},\t"
                  f"Cumulative Return: {int(wealth) - 1000000},\tShares: {observation[31:61]}")
            return_history.append(wealth - 1000000)
        self.agent.memory.clear_buffer()

        add_curve(return_history, 'DDPG testing')
        save_plot(self.figure_dir + '/testing.png', x_label='Days', y_label='Cumulative Return (Dollars)')

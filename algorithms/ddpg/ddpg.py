from env.environment import PortfolioEnv
from algorithms.ddpg.agent import Agent
import numpy as np
from plot import add_curve, save_plot


class DDPG:

    def __init__(self, load=False, alpha=0.000025, beta=0.00025, tau=0.001,
                 batch_size=64, layer1_size=400, layer2_size=300):

        self.figure_file = 'plots/ddpg.png'
        self.env = PortfolioEnv(action_scale=1000)
        self.agent = Agent(alpha=alpha, beta=beta, input_dims=self.env.state_shape(), tau=tau,
                           batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                           n_actions=self.env.n_actions())
        if load:
            self.agent.load_models()

        np.random.seed(0)

    def train(self):
        score_history = []
        buy_hold_history = self.env.buy_hold_history()
        add_curve(buy_hold_history / buy_hold_history[0], 'Buy & Hold')

        observation = self.env.reset()
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            self.agent.remember(observation, action, reward, observation_, int(done))
            self.agent.learn()
            observation = observation_
            print(f"DDPG - Date: {info},\tBalance: {int(observation[0])},\tWealth: {int(wealth)},\t"
                  f"Shares: {observation[31:61]}")
            score_history.append(wealth / 1000000)

        self.agent.save_models()

        add_curve(score_history, 'DDPG')
        save_plot(self.figure_file)

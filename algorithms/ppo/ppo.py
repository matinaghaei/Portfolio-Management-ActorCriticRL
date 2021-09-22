from env.environment import PortfolioEnv
from algorithms.ppo.agent import Agent
from plot import add_curve, save_plot


class PPO:

    def __init__(self, load=False, alpha=0.0003, n_epochs=4,
                 batch_size=5, layer1_size=512, layer2_size=512, t_max=20):

        self.figure_file = 'plots/ppo.png'
        self.t_max = t_max
        self.env = PortfolioEnv(action_scale=1000)

        self.agent = Agent(n_actions=self.env.n_actions(), batch_size=batch_size, alpha=alpha,
                           n_epochs=n_epochs, input_dims=self.env.state_shape(),
                           fc1_dims=layer1_size, fc2_dims=layer2_size)

        if load:
            self.agent.load_models()

    def train(self):
        score_history = []
        buy_hold_history = self.env.buy_hold_history()
        add_curve(buy_hold_history / buy_hold_history[0], 'Buy & Hold')
        n_steps = 0

        observation = self.env.reset()
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            n_steps += 1
            self.agent.remember(observation, action, prob, val, reward, done)
            if n_steps % self.t_max == 0:
                self.agent.learn()
            observation = observation_
            print(f"PPO - Date: {info},\tBalance: {int(observation[0])},\tWealth: {int(wealth)},\t"
                  f"Shares: {observation[31:61]}")
            score_history.append(wealth / 1000000)

        self.agent.save_models()

        add_curve(score_history, 'PPO')
        save_plot(self.figure_file)

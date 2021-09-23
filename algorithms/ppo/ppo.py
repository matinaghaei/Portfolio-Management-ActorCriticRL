from env.environment import PortfolioEnv
from algorithms.ppo.agent import Agent
from plot import add_curve, save_plot


class PPO:

    def __init__(self, intervals, load=False, alpha=0.0003, n_epochs=4,
                 batch_size=5, layer1_size=512, layer2_size=512, t_max=20):

        self.intervals = intervals
        self.figure_dir = 'plots/ppo'
        self.t_max = t_max
        self.env = PortfolioEnv(action_scale=1000)

        self.agent = Agent(n_actions=self.env.n_actions(), batch_size=batch_size, alpha=alpha,
                           n_epochs=n_epochs, input_dims=self.env.state_shape(),
                           fc1_dims=layer1_size, fc2_dims=layer2_size)

        if load:
            self.agent.load_models()

    def train(self):
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
            self.agent.memory.clear_memory()

            print(f"PPO training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate()
            print(f"PPO validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
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
        add_curve(training_history, 'PPO training')
        save_plot(filename=self.figure_dir + '/training.png',
                  x_label='Iterations', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_curve([buy_hold_final for i in range(iteration)], 'Buy & Hold')
        add_curve(validation_history, 'PPO validation')
        save_plot(filename=self.figure_dir + '/validation.png',
                  x_label='Iterations', y_label='Cumulative Return (Dollars)')

    def validate(self):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
        return wealth

    def test(self):
        return_history = []
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy & Hold')
        n_steps = 0

        observation = self.env.reset(*self.intervals['testing'])
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            n_steps += 1
            self.agent.remember(observation, action, prob, val, reward, done)
            if n_steps % self.t_max == 0:
                self.agent.learn()
            observation = observation_

            print(f"PPO testing - Date: {info.date()},\tBalance: {int(observation[0])},\t"
                  f"Cumulative Return: {int(wealth) - 1000000},\tShares: {observation[31:61]}")
            return_history.append(wealth - 1000000)
        self.agent.memory.clear_memory()

        add_curve(return_history, 'PPO testing')
        save_plot(self.figure_dir + '/testing.png', x_label='Days', y_label='Cumulative Return (Dollars)')

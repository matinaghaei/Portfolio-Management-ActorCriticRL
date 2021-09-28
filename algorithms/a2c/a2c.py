from env.environment import PortfolioEnv
from algorithms.a2c.agent import ActorCritic, Agent
from torch.multiprocessing import Pipe, Lock
from plot import add_curve, add_hline, save_plot


class A2C:

    def __init__(self, intervals, n_agents, load=False, alpha=1e-3, gamma=0.99, layer1_size=128, t_max=5):

        self.intervals = intervals
        self.n_agents = n_agents
        self.figure_dir = 'plots/a2c'
        self.t_max = t_max
        self.env = PortfolioEnv()
        self.network = ActorCritic(input_dims=self.env.state_shape(), action_dims=self.env.action_shape(),
                                   gamma=gamma, fc1_dims=layer1_size, lr=alpha)
        self.network.share_memory()

        if load:
            self.network.load_checkpoint()

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
                             verbose=verbose) for i in range(self.n_agents)]
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
                self.network.save_checkpoint()
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-5:].count(max_wealth - 1000000) == 0:
                break
            iteration += 1

        self.network.load_checkpoint()

        buy_hold_history = self.env.buy_hold_history(*self.intervals['training'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        for i in range(self.n_agents):
            add_curve(training_history[i], f'A2C - worker {i}')
        save_plot(filename=self.figure_dir + '/training.png',
                  title=f"Training - {self.intervals['training'][0]} to {self.intervals['training'][1]}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(validation_history, 'A2C')
        save_plot(filename=self.figure_dir + '/validation.png',
                  title=f"Validation - {self.intervals['validation'][0]} to {self.intervals['validation'][1]}",
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

    def test(self):
        return_history = []
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy&Hold')

        done = False
        observation = self.env.reset(*self.intervals['testing'])
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

            print(f"A2C testing - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                  f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            return_history.append(wealth - 1000000)

        add_curve(return_history, 'A2C')
        save_plot(self.figure_dir + '/testing.png',
                  title=f"Testing - {self.intervals['testing'][0]} to {self.intervals['testing'][1]}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

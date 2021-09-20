from env.environment import PortfolioEnv
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import multiprocessing as mp
from plot import add_curve, save_plot


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, fc1_dims=128, lr=1e-3, entropy_coef=1):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.rewards = []
        self.actions = []
        self.states = []
        self.done = False

        self.pi1 = nn.Linear(*input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.pi1.weight.data.size()[0])
        T.nn.init.uniform_(self.pi1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.pi1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.v1 = nn.Linear(*input_dims, fc1_dims)
        f2 = 1. / np.sqrt(self.v1.weight.data.size()[0])
        T.nn.init.uniform_(self.v1.weight.data, -f2, f2)
        T.nn.init.uniform_(self.v1.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc1_dims)

        self.mu = nn.Linear(fc1_dims, n_actions)
        f3 = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.var = nn.Linear(fc1_dims, n_actions)
        f4 = 0.003
        T.nn.init.uniform_(self.var.weight.data, -f4, f4)
        T.nn.init.uniform_(self.var.bias.data, -f4, f4)

        self.v = nn.Linear(fc1_dims, 1)
        f5 = 0.003
        T.nn.init.uniform_(self.v.weight.data, -f5, f5)
        T.nn.init.uniform_(self.v.bias.data, -f5, f5)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def set_memory(self, states, actions, rewards, done):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.done = done

    def forward(self, state):
        pi1 = F.relu(self.bn1(self.pi1(state)))
        v1 = F.relu(self.bn2(self.v1(state)))

        mu = self.mu(pi1)
        var = F.softplus(self.var(pi1))
        v = self.v(v1)

        return mu, var, v

    def calc_R(self):
        states = T.tensor(self.states, dtype=T.float).to(self.device)
        mu, var, v = self.forward(states[-1])

        R = v * (1 - int(self.done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).to(self.device)

        return batch_return

    def calc_loss(self):
        states = T.tensor(self.states, dtype=T.float).to(self.device)
        actions = T.tensor(self.actions, dtype=T.float).to(self.device)

        returns = self.calc_R()

        mu, var, values = self.forward(states)
        values = values.squeeze()
        critic_loss = ((returns - values) ** 2).mean()

        dist = Normal(mu, var.clamp(min=1e-3))
        log_probs = dist.log_prob(actions)
        actor_loss = (-log_probs * (returns-values).unsqueeze(-1)).mean()

        # entropy_loss = -dist.entropy().mean()

        total_loss = critic_loss + actor_loss

        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        mu, var, v = self.forward(state)
        dist = Normal(mu, var.clamp(min=1e-3))
        action = dist.sample()
        return action.numpy()[0]


class Agent(mp.Process):

    def __init__(self, network, lock, conn, name, t_max):
        super(Agent, self).__init__()

        self.rewards = []
        self.actions = []
        self.states = []
        self.score_history = []
        self.network = network
        self.lock = lock
        self.conn = conn
        self.name = name
        self.figure_file = f'plots/a2c/{name}.png'
        self.env = PortfolioEnv(action_scale=1000)
        self.t_max = t_max
        self.t_step = 0
        self.done = False
        self.observation = self.env.reset()

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def reset(self):
        self.done = False
        self.observation = self.env.reset()
        self.t_step = 0
        self.score_history = []

    def run(self):
        buy_hold_history = self.env.buy_hold_history()
        add_curve(buy_hold_history / buy_hold_history[0], 'Buy & Hold')

        while not self.done:
            action = self.network.choose_action(self.observation)
            observation_, reward, self.done, info, wealth = self.env.step(action)
            self.remember(self.observation, action, reward)
            if self.t_step % self.t_max == 0 or self.done:
                self.conn.send((self.states, self.actions, self.rewards, self.done))
                self.conn.recv()
                self.clear_memory()
            self.t_step += 1
            self.observation = observation_
            with self.lock:
                print(f"{self.name} Date: {info},\tBalance: {int(self.observation[0])},\tWealth: {int(wealth)},\t"
                      f"Shares: {self.observation[31:61]}")
            self.score_history.append(wealth/1000000)

        add_curve(self.score_history, 'A2C')
        save_plot(self.figure_file)

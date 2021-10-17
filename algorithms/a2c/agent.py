import os
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
    def __init__(self, input_dims, action_dims, gamma=0.99, fc1_dims=128, fc2_dims=128, lr=1e-3, entropy=0,
                 chkpt_dir='checkpoints/a2c'):
        super(ActorCritic, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.name = 'network'

        self.gamma = gamma
        self.entropy = entropy

        self.rewards = []
        self.actions = []
        self.states = []
        self.done = False

        self.drop = nn.Dropout()

        self.pi1 = nn.Linear(*input_dims, fc1_dims)
        # f1 = 1. / np.sqrt(self.pi1.weight.data.size()[0])
        # T.nn.init.uniform_(self.pi1.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.pi1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.v1 = nn.Linear(*input_dims, fc1_dims)
        # f2 = 1. / np.sqrt(self.v1.weight.data.size()[0])
        # T.nn.init.uniform_(self.v1.weight.data, -f2, f2)
        # T.nn.init.uniform_(self.v1.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc1_dims)

        self.pi2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn3 = nn.LayerNorm(fc2_dims)

        self.v2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn4 = nn.LayerNorm(fc2_dims)

        self.mu = nn.Linear(fc2_dims, *action_dims)
        # f3 = 0.003
        # T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        # T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.var = nn.Linear(fc2_dims, *action_dims)
        # f4 = 0.003
        # T.nn.init.uniform_(self.var.weight.data, -f4, f4)
        # T.nn.init.uniform_(self.var.bias.data, -f4, f4)

        self.v = nn.Linear(fc2_dims, 1)
        # f5 = 0.003
        # T.nn.init.uniform_(self.v.weight.data, -f5, f5)
        # T.nn.init.uniform_(self.v.bias.data, -f5, f5)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def set_memory(self, states, actions, rewards, done):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.done = done

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.bn1(self.pi1(state)))
        # pi1 = self.drop(pi1)
        v1 = F.relu(self.bn2(self.v1(state)))
        # v1 = self.drop(v1)
        pi2 = F.relu(self.bn3(self.pi2(pi1)))
        # pi2 = self.drop(pi2)
        v2 = F.relu(self.bn4(self.v2(v1)))
        # v2 = self.drop(v2)

        mu = self.mu(pi2)
        var = F.softplus(self.var(pi2))
        v = self.v(v2)

        return mu, var, v

    def calc_R(self, done):
        self.eval()
        states = T.tensor(self.states, dtype=T.float).to(self.device)
        mu, var, v = self.forward(states[-1])

        R = v * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).to(self.device)

        self.train()
        return batch_return

    def calc_loss(self, done=None):
        states = T.tensor(self.states, dtype=T.float).to(self.device)
        actions = T.tensor(self.actions, dtype=T.float).to(self.device)

        if done is None:
            done = self.done
        returns = self.calc_R(done)

        mu, var, values = self.forward(states)
        values = values.squeeze()
        critic_loss = ((returns - values) ** 2).mean()

        dist = Normal(mu, var.clamp(min=1e-3))
        log_probs = dist.log_prob(actions)
        actor_loss = (-log_probs * (returns-values).unsqueeze(-1)).mean()

        entropy_loss = -dist.entropy().mean()

        total_loss = critic_loss + actor_loss + self.entropy * entropy_loss

        return total_loss

    def choose_action(self, observation):
        self.eval()
        state = T.tensor([observation], dtype=T.float).to(self.device)
        mu, var, v = self.forward(state)
        dist = Normal(mu, var.clamp(min=1e-3))
        action = dist.sample()
        self.train()
        return action[0].detach().cpu().numpy().tolist()

    def save_checkpoint(self, address=None):
        print('... saving models ...')
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}')

    def load_checkpoint(self, address=None):
        print('... loading models ...')
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}'))


class Agent(mp.Process):

    def __init__(self, network, interval, conn, lock, name, t_max, verbose=False,
                 state_type='only prices', djia_year=2019):
        super(Agent, self).__init__()

        self.rewards = []
        self.actions = []
        self.states = []
        self.network = network
        self.interval = interval
        self.conn = conn
        self.lock = lock
        self.name = name
        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year)
        self.t_max = t_max
        self.verbose = verbose

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def run(self):
        done = False
        observation = self.env.reset(*self.interval)
        t_step = 0

        while not done:
            action = self.network.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            self.remember(observation, action, reward)
            if t_step % self.t_max == 0 or done:
                self.conn.send((self.states, self.actions, self.rewards, done))
                self.conn.recv()
                self.clear_memory()
            t_step += 1
            observation = observation_
            if self.verbose:
                with self.lock:
                    print(f"A2C training - {self.name} - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                          f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
        self.conn.send(wealth)

from env.environment import PortfolioEnv
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from plot import add_curve


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, fc1_dims=128, entropy_coef=1):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.entropy_coef = entropy_coef

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

        self.rewards = []
        self.actions = []
        self.states = []

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
        v1 = F.relu(self.bn2(self.v1(state)))

        mu = self.mu(pi1)
        var = F.softplus(self.var(pi1))
        v = self.v(v1)

        return mu, var, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        mu, var, v = self.forward(states[-1])

        R = v * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

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
        state = T.tensor([observation], dtype=T.float)
        mu, var, v = self.forward(state)
        dist = Normal(mu, var.clamp(min=1e-3))
        action = dist.sample()
        return action.numpy()[0]


class Agent:
    n_dones = 0
    n_gradients = 0
    score_history = []

    def __init__(self, global_actor_critic, input_dims, n_actions,
                 gamma, name, t_max, layer1_size=128):
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma, layer1_size)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.env = PortfolioEnv(action_scale=1000)
        self.t_max = t_max
        self.t_step = 1
        self.local_network_gradient = None
        self.done = False
        self.observation = self.env.reset()
        self.djia_initial = sum(self.observation[1:31])
        self.waiting = False
        self.score_history = []
        self.djia = []

    def resume(self):
        if not self.done:
            self.local_network_gradient = None
            Agent.n_gradients -= 1
            self.local_actor_critic.load_state_dict(
                self.global_actor_critic.state_dict())
            self.local_actor_critic.clear_memory()
            self.waiting = False

    def reset(self):
        self.done = False
        self.observation = self.env.reset()
        self.t_step = 1

    def iterate(self):
        if self.waiting:
            return
        action = self.local_actor_critic.choose_action(self.observation)
        observation_, reward, done, info, wealth = self.env.step(action)
        self.local_actor_critic.remember(self.observation, action, reward)
        if self.t_step % self.t_max == 0 or done:
            loss = self.local_actor_critic.calc_loss(done)
            self.local_actor_critic.zero_grad(set_to_none=True)
            loss.backward()
            self.local_network_gradient =\
                [param.grad for param in self.local_actor_critic.parameters()]
            self.waiting = True
            Agent.n_gradients += 1
        self.t_step += 1
        self.observation = observation_
        print(f"{self.name} Date: {info},\tBalance: {int(self.observation[0])},\tWealth: {int(wealth)},\t"
              f"Shares: {self.observation[31:61]}")
        self.score_history.append(wealth/1000000)
        self.djia.append(sum(self.observation[1:31])/self.djia_initial)

        if done:
            add_curve(self.score_history, f'A2C {self.name}')
            Agent.n_dones += 1
            # self.reset()

    def get_gradient(self):
        return self.local_network_gradient

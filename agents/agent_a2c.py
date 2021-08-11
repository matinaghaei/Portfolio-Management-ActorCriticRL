from env.environment import PortfolioEnv
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from plot import plot

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, layer1_size=128, layer2_size=128):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, layer1_size)
        self.v1 = nn.Linear(*input_dims, layer1_size)
        self.mu = nn.Linear(layer1_size, n_actions)
        self.var = nn.Linear(layer1_size, n_actions)
        self.v = nn.Linear(layer1_size, 1)

        # self.pi1 = nn.Linear(*input_dims, layer1_size)
        # self.pi2 = nn.Linear(layer1_size, layer2_size)
        # self.mu = nn.Linear(layer2_size, n_actions)
        # self.var = nn.Linear(layer2_size, n_actions)
        # self.v1 = nn.Linear(*input_dims, layer1_size)
        # self.v2 = nn.Linear(layer1_size, layer2_size)
        # self.v = nn.Linear(layer2_size, 1)

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
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        mu = T.tanh(self.mu(pi1))
        var = F.softplus(self.var(pi1))
        v = self.v(v1)

        # pi1 = F.relu(self.pi1(state))
        # pi2 = F.relu(self.pi2(pi1))
        # mu = self.mu(pi2)
        # var = F.softplus(self.var(pi2))
        #
        # v1 = F.relu(self.v1(state))
        # v2 = F.relu(self.v2(v1))
        # v = self.v(v2)

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
                 gamma, name, t_max, layer1_size=128, layer2_size=128):
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma, layer1_size, layer2_size)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.env = PortfolioEnv(action_scale=1)
        self.t_max = t_max
        self.figure_file = f'plots/a2c/{self.name}.png'
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
              f"َActions: {[int(act) for act in action]}")
        self.score_history.append(wealth/1000000)
        self.djia.append(sum(self.observation[1:31])/self.djia_initial)

        if done:
            plot(self.score_history, 'A2C', self.djia, 'DJIA', self.figure_file)
            Agent.n_dones += 1
            # self.reset()

    def get_gradient(self):
        return self.local_network_gradient

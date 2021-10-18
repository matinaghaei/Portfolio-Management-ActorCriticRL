import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, action_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, *action_shape))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def clear_buffer(self):
        self.mem_cntr = 0


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, action_dims, fc1_dims, fc2_dims, name,
                 chkpt_dir='checkpoints/ddpg'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.name = name

        self.drop = nn.Dropout()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(*action_dims, fc2_dims)
        self.bna = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        # state_value = self.drop(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = self.action_value(action)
        # action_value = self.bna(action_value)
        action_value = F.relu(action_value)
        # action_value = self.drop(action_value)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}')

    def load_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, action_dims, fc1_dims, fc2_dims, name,
                 chkpt_dir='checkpoints/ddpg'):
        super(ActorNetwork, self).__init__()
        
        self.checkpoint_dir = chkpt_dir
        self.name = name

        self.drop = nn.Dropout()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.mu = nn.Linear(fc2_dims, *action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.drop(x)
        x = self.mu(x)
        # x = T.tanh(x)

        return x

    def save_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}')

    def load_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}'))


class Agent(object):
    def __init__(self, alpha, beta, input_dims, action_dims, tau, gamma=0.99,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, action_dims)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, action_dims,
                                  layer1_size, layer2_size, name='actor')
        self.critic = CriticNetwork(beta, input_dims, action_dims,
                                    layer1_size, layer2_size, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, action_dims,
                                         layer1_size, layer2_size, name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, action_dims,
                                           layer1_size, layer2_size, name='target_critic')
        self.actor.train()
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()

        self.noise = OUActionNoise(mu=np.zeros(action_dims))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation)
        mu = mu + T.tensor(self.noise(),
                                dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu.detach().cpu().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)
        
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.critic.train()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self, address=None):
        print('... saving models ...')
        self.actor.save_checkpoint(address)
        self.target_actor.save_checkpoint(address)
        self.critic.save_checkpoint(address)
        self.target_critic.save_checkpoint(address)

    def load_models(self, address=None):
        print('... loading models ...')
        self.actor.load_checkpoint(address)
        self.target_actor.load_checkpoint(address)
        self.critic.load_checkpoint(address)
        self.target_critic.load_checkpoint(address)

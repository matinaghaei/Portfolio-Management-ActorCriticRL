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
    def __init__(self, beta, input_dims, action_dims, fc1_dims, fc2_dims, fc3_dims, name,
                 bn_drop='only bn', action_input_layer='nothing', chkpt_dir='checkpoints/ddpg',
                 state_activation=False, action_activation=True):
        super(CriticNetwork, self).__init__()

        self.fc3_dims = fc3_dims
        self.bn_drop = bn_drop
        self.action_input_layer = action_input_layer
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.state_activation = state_activation
        self.action_activation = action_activation

        self.drop = nn.Dropout()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        # f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        # T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        if fc3_dims is None:
            self.action_value = nn.Linear(*action_dims, fc2_dims)
            self.bn3 = nn.LayerNorm(fc2_dims)
            self.q = nn.Linear(fc2_dims, 1)
            # f3 = 0.003
            # T.nn.init.uniform_(self.q.weight.data, -f3, f3)
            # T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        else:
            self.fc3 = nn.Linear(fc2_dims, fc3_dims)
            self.bn3 = nn.LayerNorm(fc3_dims)
            self.action_value = nn.Linear(*action_dims, fc3_dims)
            self.q = nn.Linear(fc3_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        if self.bn_drop == 'only bn' or self.bn_drop == 'bn drop':
            state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        if self.bn_drop == 'only drop' or self.bn_drop == 'bn drop':
            state_value = self.drop(state_value)
        state_value = self.fc2(state_value)
        if self.bn_drop == 'only bn' or self.bn_drop == 'bn drop':
            state_value = self.bn2(state_value)
        if self.state_activation:
            state_value = F.relu(state_value)
            if self.bn_drop == 'only drop' or self.bn_drop == 'bn drop':
                state_value = self.drop(state_value)
        if self.fc3_dims is not None:
            state_value = F.relu(state_value)
            state_value = self.fc3(state_value)
            state_value = self.bn3(state_value)
        
        action_value = self.action_value(action)
        if self.action_input_layer == 'bn_drop' and (self.bn_drop == 'only bn' or self.bn_drop == 'bn drop'):
            action_value = self.bn3(action_value)
        if self.action_activation:
            action_value = F.relu(action_value)
            if self.action_input_layer == 'bn_drop' and (self.bn_drop == 'only drop' or self.bn_drop == 'bn drop'):
                action_value = self.drop(action_value)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        # print(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, action_dims, fc1_dims, fc2_dims, fc3_dims, name,
                 bn_drop='only bn', chkpt_dir='checkpoints/ddpg'):
        super(ActorNetwork, self).__init__()
        
        self.fc3_dims = fc3_dims
        self.bn_drop = bn_drop
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.drop = nn.Dropout()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        # f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        # T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        if fc3_dims is None:
            self.mu = nn.Linear(fc2_dims, *action_dims)
            # f3 = 0.003
            # T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
            # T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        else:
            self.fc3 = nn.Linear(fc2_dims, fc3_dims)
            self.bn3 = nn.LayerNorm(fc3_dims)
            self.mu = nn.Linear(fc3_dims, *action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        if self.bn_drop == 'only bn' or self.bn_drop == 'bn drop':
            x = self.bn1(x)
        x = F.relu(x)
        if self.bn_drop == 'only drop' or self.bn_drop == 'bn drop':
            x = self.drop(x)
        x = self.fc2(x)
        if self.bn_drop == 'only bn' or self.bn_drop == 'bn drop':
            x = self.bn2(x)
        x = F.relu(x)
        if self.bn_drop == 'only drop' or self.bn_drop == 'bn drop':
            x = self.drop(x)
        if self.fc3_dims is not None:
            x = self.fc3(x)
            x = self.bn3(x)
            x = F.relu(x)
        x = self.mu(x)

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, alpha, beta, input_dims, action_dims, tau, gamma=0.99,
                 max_size=1000000, layer1_size=400, layer2_size=300, layer3_size=None, 
                 batch_size=64, action_interpret='portfolio', train_eval='new', bn_drop='only bn', 
                 action_input_layer='nothing', state_activation=False, action_activation=True):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, action_dims)
        self.batch_size = batch_size
        self.action_interpret = action_interpret
        self.train_eval = train_eval

        self.actor = ActorNetwork(alpha, input_dims, action_dims,
                                  layer1_size, layer2_size, layer3_size, 
                                  name='actor', bn_drop=bn_drop)
        self.critic = CriticNetwork(beta, input_dims, action_dims,
                                    layer1_size, layer2_size, layer3_size, name='critic', 
                                    bn_drop=bn_drop, action_input_layer=action_input_layer,
                                    state_activation=state_activation,
                                    action_activation=action_activation)

        self.target_actor = ActorNetwork(alpha, input_dims, action_dims,
                                         layer1_size, layer2_size, layer3_size,
                                         name='target_actor', bn_drop=bn_drop)
        self.target_critic = CriticNetwork(beta, input_dims, action_dims,
                                           layer1_size, layer2_size, layer3_size,
                                           name='target_critic', bn_drop=bn_drop,
                                           action_input_layer=action_input_layer,
                                           state_activation=state_activation,
                                           action_activation=action_activation)
        if train_eval == 'new':
            self.actor.train()
            self.critic.train()
            self.target_actor.eval()
            self.target_critic.eval()

        self.noise = OUActionNoise(mu=np.zeros(action_dims))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu = mu + T.tensor(self.noise(),
                                dtype=T.float).to(self.actor.device)
        # print(mu)
        if self.action_interpret == 'portfolio':
            mu = F.softmax(mu, -1)
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

        if self.train_eval == 'old':
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)
        
        if self.train_eval == 'old':
            self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        if self.train_eval == 'old':
            self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        if self.train_eval == 'new':
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

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

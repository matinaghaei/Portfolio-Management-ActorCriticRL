import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, action_dims, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='checkpoints/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.name = 'actor'

        self.drop = nn.Dropout()

        self.pi1 = nn.Linear(*input_dims, fc1_dims)
        # f1 = 1. / np.sqrt(self.pi1.weight.data.size()[0])
        # T.nn.init.uniform_(self.pi1.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.pi1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.pi2 = nn.Linear(fc1_dims, fc2_dims)
        # f2 = 1. / np.sqrt(self.pi2.weight.data.size()[0])
        # T.nn.init.uniform_(self.pi2.weight.data, -f2, f2)
        # T.nn.init.uniform_(self.pi2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.mu = nn.Linear(fc2_dims, *action_dims)
        # f3 = 0.003
        # T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        # T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.var = nn.Linear(fc2_dims, *action_dims)
        # f4 = 0.003
        # T.nn.init.uniform_(self.var.weight.data, -f4, f4)
        # T.nn.init.uniform_(self.var.bias.data, -f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = F.relu(self.bn1(self.pi1(state)))
        # state = self.drop(state)
        state = F.relu(self.bn2(self.pi2(state)))
        # state = self.drop(state)
        mu = self.mu(state)
        var = F.softplus(self.var(state))

        return mu, var

    def save_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}')

    def load_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}'))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,fc1_dims=400, fc2_dims=300,
                 chkpt_dir='checkpoints/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.name = 'critic'

        self.drop = nn.Dropout()

        self.v1 = nn.Linear(*input_dims, fc1_dims)
        # f1 = 1. / np.sqrt(self.v1.weight.data.size()[0])
        # T.nn.init.uniform_(self.v1.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.v1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.v2 = nn.Linear(fc1_dims, fc2_dims)
        # f2 = 1. / np.sqrt(self.v2.weight.data.size()[0])
        # T.nn.init.uniform_(self.v2.weight.data, -f2, f2)
        # T.nn.init.uniform_(self.v2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.v = nn.Linear(fc2_dims, 1)
        # f3 = 0.003
        # T.nn.init.uniform_(self.v.weight.data, -f3, f3)
        # T.nn.init.uniform_(self.v.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = F.relu(self.bn1(self.v1(state)))
        # value = self.drop(value)
        value = F.relu(self.bn2(self.v2(value)))
        # value = self.drop(value)
        value = self.v(value)

        return value

    def save_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}')

    def load_checkpoint(self, address=None):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}'))


class Agent:
    def __init__(self, action_dims, input_dims, fc1_dims=256, fc2_dims=256, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10, entropy=0):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy = entropy

        self.actor = ActorNetwork(action_dims, input_dims, alpha, fc1_dims, fc2_dims)
        self.critic = CriticNetwork(input_dims, alpha, fc1_dims, fc2_dims)

        self.actor.train()
        self.critic.train()

        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, address=None):
        print('... saving models ...')
        self.actor.save_checkpoint(address)
        self.critic.save_checkpoint(address)

    def load_models(self, address=None):
        print('... loading models ...')
        self.actor.load_checkpoint(address)
        self.critic.load_checkpoint(address)

    def choose_action(self, observation):
        self.actor.eval()
        self.critic.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        mu, var = self.actor(state)
        dist = Normal(mu, var.clamp(min=1e-3))
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).detach().cpu().numpy().tolist()
        action = T.squeeze(action).detach().cpu().numpy().tolist()
        value = T.squeeze(value).item()

        self.actor.train()
        self.critic.train()
        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                mu, var = self.actor(states)
                dist = Normal(mu, var.clamp(min=1e-3))
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch].unsqueeze(-1) * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch].unsqueeze(-1)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                entropy_loss = -dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss + self.entropy * entropy_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

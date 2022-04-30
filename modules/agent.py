import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .utils import ReplayMemory
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else: self.device=device
        self.layers = nn.ModuleList()
        for i in range(len(config['hid_ds'])-1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(config['hid_ds'][i], 
                              config['hid_ds'][i+1]),
                    nn.ReLU()))
        self.layers = nn.Sequential(
            *self.layers)
        
    def forward(self, x):
        x = torch.tensor(x).float().to(self.device)
        return F.relu(self.layers(x))
    

class Agent:
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.policy_network = Network(config, self.device)    
        self.tgt_network = Network(config, self.device)
        self.policy_network.to(self.device)
        self.tgt_network.to(self.device)
        
        for param in self.tgt_network.parameters():
            param.requires_grad = False
        
        self.reward_discount = config['reward_discount']
        self.batch_size = config['batch_size']
        self.update_tgt = config['update_tgt'] 
        self.memory = ReplayMemory()
        self.optimizer = Adam(self.policy_network.parameters(), lr=config['lr'])
        self.loss = nn.SmoothL1Loss()
        
    def get_action(self, observations, search_epsilon):
        '''
        param: observations: All the next possible future states
        param: search_epsilon: probability threshold for search exploit 
        '''
        # Explore
        if np.random.uniform() < search_epsilon:
            action = torch.tensor(random.sample(range(observations.shape[0]), 1))
        # Exploit
        else:
            q_values = self.policy_network(observations)
            action = torch.argmax(q_values)
        return action
    
    def update_params(self, step):
        '''
        1. Sample state, action, next_states & reward from memory
        2. Calculate the reward using the policy_network 
            - Policy network predicts the reward given a state
            - The predicted reward is used to select an action
        3. Calculate the value of the predicted next_state using tgt_network
        4. Calculate target rewards (predicted_next_state_values * reward_discount) + ground truth reward
        5. Optimise policy network parameters based on the predicted state action values (policy network) and target rewards
        '''
        if self.memory.__len__ <= self.batch_size:
            print('Memory Too short')
            return 
        # Used values
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # Predicted values
        predicted_rewards = torch.zeros(self.batch_size, 1, requires_grad=False)
        next_state_values = torch.zeros(self.batch_size, 1, requires_grad=False)
        for i in range(self.batch_size):
            # Predicted reward based on action taken (chosen randomly or not)
            predicted_reward = self.policy_network(actions[i])
            predicted_rewards[i] = predicted_reward
            # Calculate rewards for the following state
            next_state_values[i] = torch.max(self.tgt_network(next_states[i]))
        # Target rewards are discounted future reward + actual reward for action
        target_rewards = (next_state_values * self.reward_discount) + torch.tensor(rewards)
        # Minimise the distance between predicted reward and target
        # Therefore model gets better at predicting future reward
        loss = self.loss(predicted_rewards, target_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            # Clip gradients
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()    
            
        if step % self.update_tgt == 0:
            print(f'Current loss: {loss}')
            print('Updating Target Network')
            self.tgt_network.load_state_dict(self.policy_network.state_dict())

        return loss    
            
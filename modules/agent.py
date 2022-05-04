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
        
        self.search_epsilon = config['search_epsilon']
        self.search_epsilon_scalar = config['search_epsilon_scalar']
        self.tgt_net_weight_multiplier = config['tgt_net_weight_multiplier']
        self.reward_discount = config['reward_discount']
        self.batch_size = config['batch_size']
        self.update_tgt = config['update_tgt'] 
        self.memory = ReplayMemory()
        self.optimizer = Adam(self.policy_network.parameters(), lr=config['lr'])
        self.loss = nn.SmoothL1Loss()
        
    def get_action(self, observations):
        '''
        param: observations: All the next possible future states
        param: search_epsilon: probability threshold for search exploit 
        '''
        # Explore
        if random.randint(0,100) < int(self.search_epsilon*100):
            action = torch.tensor(random.sample(range(observations.shape[0]), 1))
        # Exploit
        else:
            q_values = self.policy_network(observations)
            action = torch.argmax(q_values)
        self.search_epsilon *= self.search_epsilon_scalar
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
            return 
        # all possible actions, action taken, reward recieved, next possible actions
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # Predicted values
        predicted_rewards = torch.zeros(self.batch_size, 1, requires_grad=False).to(self.device)
        next_state_values = torch.zeros(self.batch_size, 1, requires_grad=False).to(self.device)
        for i in range(self.batch_size):
            # Get predicted reward from actions taken
            predicted_reward = self.policy_network(actions[i])
            predicted_rewards[i] = predicted_reward
            # Calculate rewards for the following state
            next_state_values[i] = torch.max(self.tgt_network(next_states[i]))
            
        dones = torch.FloatTensor(dones).to(self.device)
        masked_next_state_values = (1-dones) * next_state_values.to(self.device)
        target_rewards = (masked_next_state_values * self.reward_discount) + torch.tensor(rewards).to(self.device)

        error = predicted_rewards - target_rewards
        loss = torch.where(
            torch.abs(error) < 1.0,
            0.5 * error**2,
            1.0 * (torch.abs(error) - 0.5))
        
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()    
        
        with torch.no_grad():
            for p, p_targ in zip(self.policy_network.parameters(), self.tgt_network.parameters()):
                p_targ.data.mul_(self.tgt_net_weight_multiplier)
                p_targ.data.add_((1 - self.tgt_net_weight_multiplier) * p.data)

        return loss    
            
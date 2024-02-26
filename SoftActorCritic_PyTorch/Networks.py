import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

data_type = torch.float64

class Q_Network(nn.Module):
    def __init__(self, obs_dim, actions_dim, pref_dim, hidden1_dim, hidden2_dim, hidden3_dim, alfa, beta1, beta2, name='Q_net', chkpt_dir = './Train/Networks'):
        super(Q_Network, self).__init__()
        self.obs_dim = obs_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim
        self.actions_dim = actions_dim
        self.pref_dim = pref_dim

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = self.checkpoint_dir + '/' + self.name
        
        self.hidden1 = nn.Linear(self.obs_dim + self.actions_dim + self.pref_dim, self.hidden1_dim, dtype=data_type)
        self.hidden2 = nn.Linear(self.hidden1_dim, self.hidden2_dim, dtype=data_type)
        self.hidden3 = nn.Linear(self.hidden2_dim, self.hidden3_dim, dtype=data_type)
        self.q = nn.Linear(self.hidden3_dim, 1, dtype=data_type)

        self.optimizer = optim.Adam(self.parameters(), lr=alfa, betas=(beta1, beta2))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print("Device for " + self.name + ": ", self.device)

        self.to(self.device)
    
    def forward(self, state, action, pref):
        Q_value = self.hidden1(torch.cat([state, action, pref], dim=1))
        Q_value = F.relu(Q_value)
        
        Q_value = self.hidden2(Q_value)
        Q_value = F.relu(Q_value)
        
        Q_value = self.hidden3(Q_value)
        Q_value = F.relu(Q_value)

        Q_value = self.q(Q_value)

        return Q_value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class P_Network(nn.Module):
    def __init__(self, obs_dim, actions_dim, pref_dim, hidden1_dim, hidden2_dim, alfa, beta1, beta2, name='P_net', chkpt_dir = './Train/Networks'):
        super(P_Network, self).__init__()
        self.obs_dim = obs_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.actions_dim = actions_dim
        self.pref_dim = pref_dim

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = self.checkpoint_dir + '/' + self.name

        self.reparam_noise = 1e-9
        
        self.hidden1 = nn.Linear(self.obs_dim + self.pref_dim, self.hidden1_dim, dtype=data_type)
        self.hidden2 = nn.Linear(self.hidden1_dim, self.hidden2_dim, dtype=data_type)
        self.mu = nn.Linear(self.hidden2_dim, self.actions_dim, dtype=data_type)
        self.sigma = nn.Linear(self.hidden2_dim, self.actions_dim, dtype=data_type)

        self.optimizer = optim.Adam(self.parameters(), lr=alfa, betas=(beta1, beta2))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print("Device for " + self.name + ": ", self.device)

        self.to(self.device)
    
    def forward(self, state, pref):
        aux = self.hidden1(torch.cat([state, pref], dim=1))
        aux = F.relu(aux)
        
        aux = self.hidden2(aux)
        aux = F.relu(aux)
        
        mu = self.mu(aux)

        sigma = self.sigma(aux)
        sigma = torch.clamp(torch.exp(sigma), min=self.reparam_noise, max=10)

        return mu, sigma

    def sample_normal(self, state, pref, reparameterize = True):
        mu, sigma = self.forward(state, pref)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()   #Allows backpropagation using reparam trick
        else:
            actions = probabilities.sample()

        actions_restricted = torch.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-actions_restricted.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        sigma = sigma.mean(1, keepdim=True)

        return actions_restricted, log_probs, sigma

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
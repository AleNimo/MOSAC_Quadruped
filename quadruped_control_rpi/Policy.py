import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

data_type = torch.float64


class P_Network(nn.Module):
    def __init__(self, obs_dim, actions_dim, pref_dim, hidden1_dim, hidden2_dim, chkpt_dir = './P_net'):
        super(P_Network, self).__init__()
        self.obs_dim = obs_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.actions_dim = actions_dim
        self.pref_dim = pref_dim

        self.checkpoint_dir = chkpt_dir

        self.reparam_noise = 1e-9
        
        self.hidden1 = nn.Linear(self.obs_dim + self.pref_dim, self.hidden1_dim, dtype=data_type)
        self.hidden2 = nn.Linear(self.hidden1_dim, self.hidden2_dim, dtype=data_type)
        self.mu = nn.Linear(self.hidden2_dim, self.actions_dim, dtype=data_type)
        self.sigma = nn.Linear(self.hidden2_dim, self.actions_dim, dtype=data_type)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print("Device for policy network: ", self.device)

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
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_dir, map_location=self.device))
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ReplayBuffer import ReplayBuffer
from Networks import Q_Network, P_Network

import copy

class SAC_Agent():
    def __init__(self, name, obs_dim, actions_dim, replay_buffer_size):

        self.agent_name = name

        #Default values
        self.discount_factor = 0.99
        self.update_factor = 0.005
        self.replay_batch_size = 1000

        self.update_Q = 1
        self.update_P = 1

        self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_dim, actions_dim)

        self.P_net = P_Network(obs_dim, actions_dim, hidden1_dim=64, hidden2_dim=32,
                               alfa=0.0003, beta1=0.9, beta2=0.999)

        self.P_loss = torch.tensor(0, dtype=torch.float).to(self.P_net.device)

        self.Q1_net = Q_Network(obs_dim, actions_dim, hidden1_dim=128, hidden2_dim=64, hidden3_dim=32,
                                  alfa=0.0003, beta1=0.9, beta2=0.999, name='Q1_net')

        self.Q2_net = Q_Network(obs_dim, actions_dim, hidden1_dim=128, hidden2_dim=64, hidden3_dim=32,
                                  alfa=0.0003, beta1=0.9, beta2=0.999, name='Q2_net')

        self.Q_loss = torch.tensor(0, dtype=torch.float).to(self.Q1_net.device)

        # Create target networks with different names and directories

        self.Q1_target_net = copy.deepcopy(self.Q1_net)
        self.Q1_target_net.name = 'Q1_target_net'
        self.Q1_target_net.checkpoint_file = self.Q1_target_net.checkpoint_dir + '/' + self.Q1_target_net.name

        self.Q2_target_net = copy.deepcopy(self.Q2_net)
        self.Q2_target_net.name = 'Q2_target_net'
        self.Q2_target_net.checkpoint_file = self.Q2_target_net.checkpoint_dir + '/' + self.Q2_target_net.name

        self.target_entropy = -actions_dim

        self.entropy = torch.tensor(0, dtype=torch.float).to(self.P_net.device)

        # Create entropy temperature coefficient 
        self.alpha = torch.tensor(0.01, dtype=torch.float64).to(self.P_net.device)
        self.alpha.requires_grad = True

        self.alpha_optimizer = optim.Adam([self.alpha], lr=0.0003, betas=(0.9, 0.999))

        # Create the required directories if necessary
        if not os.path.isdir("./{0:s}".format(self.agent_name)):
            if os.path.isfile("./{0:s}".format(self.agent_name)):
                input("File './{0:s}' needs to be deleted. Press enter to continue.".format(self.agent_name))
                os.remove("./{0:s}".format(self.agent_name))
            os.mkdir("./{0:s}".format(self.agent_name))
            os.chdir("./{0:s}".format(self.agent_name))
            os.mkdir("./Train")
            os.mkdir("./Train/Networks")
            with open('./Train/Progress.txt', 'w') as file: np.savetxt(file, np.array((0, )), fmt='%d')
        else:
            os.chdir("./{0:s}".format(self.agent_name))

    def choose_action(self, observations, random = True):
        state = torch.tensor([observations]).to(self.P_net.device)

        if random:
            actions,_ = self.P_net.sample_normal(state, reparameterize=False)
        else:
            actions,_ = self.P_net(state)

        return actions.detach().cpu().numpy()

    def minimal_Q(self, state, action):
        Q1 = self.Q1_net(state, action)
        Q2 = self.Q2_net(state, action)

        return torch.min(Q1, Q2)

    def minimal_Q_target(self, state, action):
        Q1 = self.Q1_target_net(state, action)
        Q2 = self.Q2_target_net(state, action)

        return torch.min(Q1, Q2)

    def remember(self, state, action, reward, next_state, done_flag):
        self.replay_buffer.store(state, action, reward, next_state, done_flag)

    def update_target_net_parameters(self):
        target_Q1_state_dict = dict(self.Q1_target_net.named_parameters())
        Q1_state_dict = dict(self.Q1_net.named_parameters())

        target_Q2_state_dict = dict(self.Q2_target_net.named_parameters())
        Q2_state_dict = dict(self.Q2_net.named_parameters())

        for name in Q1_state_dict:
            Q1_state_dict[name] = self.update_factor * Q1_state_dict[name].clone() + (1-self.update_factor) * target_Q1_state_dict[name].clone()
            Q2_state_dict[name] = self.update_factor * Q2_state_dict[name].clone() + (1-self.update_factor) * target_Q2_state_dict[name].clone()

        self.Q1_target_net.load_state_dict(Q1_state_dict)
        self.Q2_target_net.load_state_dict(Q2_state_dict)

    def save_models(self):
        self.P_net.save_checkpoint()
        self.Q1_net.save_checkpoint()
        self.Q2_net.save_checkpoint()
        self.Q1_target_net.save_checkpoint()
        self.Q2_target_net.save_checkpoint()
        torch.save(self.alpha, './Train/Networks/alpha_tensor.pt')

    def load_models(self):
        self.P_net.load_checkpoint()
        self.Q1_net.load_checkpoint()
        self.Q2_net.load_checkpoint()
        self.Q1_target_net.load_checkpoint()
        self.Q2_target_net.load_checkpoint()
        self.alpha = torch.load('./Train/Networks/alpha_tensor.pt')
        self.alpha_optimizer = optim.Adam([self.alpha], lr=0.001, betas=(0.9, 0.999))

    def learn(self, episode):
        if self.replay_buffer.mem_counter < self.replay_batch_size:
            return
        state, action, reward, next_state, done_flag = self.replay_buffer.sample(self.replay_batch_size)

        #Convert np.arrays to tensors in GPU
        state = torch.tensor(state, dtype=torch.float64).to(self.P_net.device)
        action = torch.tensor(action, dtype=torch.float64).to(self.P_net.device)
        reward = torch.tensor(reward, dtype=torch.float64).to(self.P_net.device)
        next_state = torch.tensor(next_state, dtype=torch.float64).to(self.P_net.device)
        done_flag = torch.tensor(done_flag, dtype=torch.float64).to(self.P_net.device)

        if episode % self.update_Q == 0:
            #Update Q networks
            with torch.no_grad():
                next_action, log_prob = self.P_net.sample_normal(next_state, reparameterize=False)
                next_Q = self.minimal_Q_target(next_state, next_action)
                Q_hat = reward + self.discount_factor * (1-done_flag) * (next_Q.view(-1) - self.alpha * log_prob.view(-1))

            Q = self.minimal_Q(state, action).view(-1)

            self.Q_loss = F.mse_loss(Q, Q_hat, reduction='mean')

            self.Q1_net.optimizer.zero_grad()
            self.Q2_net.optimizer.zero_grad()
            self.Q_loss.backward()
            self.Q1_net.optimizer.step()
            self.Q2_net.optimizer.step()

        if episode % self.update_P == 0:
            #Update P networks
            action, log_prob = self.P_net.sample_normal(state, reparameterize=True)

            self.entropy = torch.mean(-log_prob)

            Q = self.minimal_Q(state, action).view(-1)

            self.P_loss = torch.mean(self.alpha * log_prob.view(-1) - Q)

            self.P_net.optimizer.zero_grad()
            self.P_loss.backward()
            self.P_net.optimizer.step()

            #Update Alpha

            self.alpha_optimizer.zero_grad()

            Alpha_loss = self.alpha * torch.mean((-log_prob - self.target_entropy).detach())
            Alpha_loss.backward()

            self.alpha_optimizer.step()

            self.update_target_net_parameters()
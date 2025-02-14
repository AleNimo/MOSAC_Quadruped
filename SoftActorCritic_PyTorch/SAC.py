import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ReplayBuffer import ReplayBuffer
from Networks import Q_Network, P_Network

import copy

class SAC_Agent():
    def __init__(self, name, environment, pref_max_vector, pref_min_vector, replay_buffer_size):

        self.agent_name = name
        self.environment = environment

        #Default values
        self.discount_factor = 0.95
        self.update_factor = 0.005
        self.replay_batch_size = 1000

        self.update_Q = 1
        self.update_P = 1

        if np.array_equal(pref_max_vector, pref_min_vector):    #Without multi-objective approach
            print("Training without Multi-Objective approach")
            pref_dim = 0
        else:
            pref_dim = pref_max_vector.size

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.environment.obs_dim, self.environment.act_dim, self.environment.rwd_dim, pref_max_vector, pref_min_vector)

        self.P_net = P_Network(self.environment.obs_dim, self.environment.act_dim, pref_dim, hidden1_dim=64, hidden2_dim=32,
                               alfa=0.0003, beta1=0.9, beta2=0.999)

        self.P_loss = torch.tensor(0, dtype=torch.float64).to(self.P_net.device)

        self.Q1_net = Q_Network(self.environment.obs_dim, self.environment.act_dim, pref_dim, hidden1_dim=128, hidden2_dim=64, hidden3_dim=32,
                                  alfa=0.0003, beta1=0.9, beta2=0.999, name='Q1_net')

        self.Q2_net = Q_Network(self.environment.obs_dim, self.environment.act_dim, pref_dim, hidden1_dim=128, hidden2_dim=64, hidden3_dim=32,
                                  alfa=0.0003, beta1=0.9, beta2=0.999, name='Q2_net')

        self.Q_loss = torch.tensor(0, dtype=torch.float64).to(self.Q1_net.device)

        # Create target networks with different names and directories

        self.Q1_target_net = copy.deepcopy(self.Q1_net)
        self.Q1_target_net.name = 'Q1_target_net'
        self.Q1_target_net.checkpoint_file = self.Q1_target_net.checkpoint_dir + '/' + self.Q1_target_net.name

        self.Q2_target_net = copy.deepcopy(self.Q2_net)
        self.Q2_target_net.name = 'Q2_target_net'
        self.Q2_target_net.checkpoint_file = self.Q2_target_net.checkpoint_dir + '/' + self.Q2_target_net.name

        self.target_entropy = -self.environment.act_dim

        self.entropy = torch.tensor(0, dtype=torch.float64).to(self.P_net.device)

        self.std = torch.tensor(0, dtype=torch.float64).to(self.P_net.device)

        # Create entropy temperature coefficient, we store log_alpha and use log_alpha.exp() when needed to force alpha to be always positive
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float64).to(self.P_net.device)   #Initial alpha of 0.1
        self.log_alpha.requires_grad = True

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003, betas=(0.9, 0.999))

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

    def choose_action(self, state_numpy, pref_numpy, random = True, tensor = False):

        state = torch.tensor(np.expand_dims(state_numpy, axis=0)).to(self.P_net.device)
        pref = torch.tensor(pref_numpy).to(self.P_net.device)

        if random:
            actions,_,_ = self.P_net.sample_normal(state, pref, reparameterize=False)
        else:
            actions,_ = self.P_net(state, pref)
            actions = torch.tanh(actions) # Restrict the actions to (-1;1) when sample_normal is not used

        #Return as a np.array when acting on the environment
        if tensor == False: return actions.detach().cpu().numpy()
        #Return as a tensor only for computing the real return and the agent doesn't complete the episode (falling or reaching target) under 200 steps
        else:               return actions

    def minimal_Q(self, state, action, pref):
        Q1 = self.Q1_net(state, action, pref)
        Q2 = self.Q2_net(state, action, pref)

        return torch.min(Q1, Q2)

    def minimal_Q_target(self, state, action, pref):
        Q1 = self.Q1_target_net(state, action, pref)
        Q2 = self.Q2_target_net(state, action, pref)

        return torch.min(Q1, Q2)

    def remember(self, state, action, reward, next_state, done_flag):   #The reward is not saved because it is recalculated every time with a different preference
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

    def save_models(self, episode):
        self.P_net.save_checkpoint(episode)
        self.Q1_net.save_checkpoint(episode)
        self.Q2_net.save_checkpoint(episode)
        self.Q1_target_net.save_checkpoint(episode)
        self.Q2_target_net.save_checkpoint(episode)
        torch.save(self.log_alpha, './Train/Networks/alpha_episode_{0:07d}.pt'.format(episode))

    def load_models(self):
        if not os.path.isfile('./Train/Progress.txt'):
            print('Progress.txt could not be found')
            exit
        with open('./Train/Progress.txt', 'r') as file: episode = int(np.loadtxt(file))

        self.P_net.load_checkpoint(episode)
        self.Q1_net.load_checkpoint(episode)
        self.Q2_net.load_checkpoint(episode)
        self.Q1_target_net.load_checkpoint(episode)
        self.Q2_target_net.load_checkpoint(episode)
        self.log_alpha = torch.load('./Train/Networks/alpha_episode_{0:07d}.pt'.format(episode))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003, betas=(0.9, 0.999))

    def learn(self, step):
        if self.replay_buffer.mem_counter < self.replay_batch_size:
            return
        state, action, partial_rewards, pref, next_state, done_flag = self.replay_buffer.sample(self.replay_batch_size)

        #Convert np.arrays to tensors in GPU
        state = torch.tensor(state, dtype=torch.float64).to(self.P_net.device)
        action = torch.tensor(action, dtype=torch.float64).to(self.P_net.device)
        pref = torch.tensor(pref, dtype=torch.float64).to(self.P_net.device)
        next_state = torch.tensor(next_state, dtype=torch.float64).to(self.P_net.device)
        partial_rewards = torch.tensor(partial_rewards, dtype=torch.float64).to(self.P_net.device)
        done_flag = torch.tensor(done_flag, dtype=torch.float64).to(self.P_net.device)

        #Recompute the total reward based on the random preferences (linear combination)
        #the "paws", "ommited step" and "not flipping reward" are not weighted by a preference so they are added afterwards
        reward = torch.sum(pref * partial_rewards[:,:-2], axis=1) + partial_rewards[:,-2] + partial_rewards[:,-1]

        if step % self.update_Q == 0:
            #Update Q networks
            with torch.no_grad():
                next_action, log_prob, _ = self.P_net.sample_normal(next_state, pref, reparameterize=False)
                next_Q = self.minimal_Q_target(next_state, next_action, pref)
                Q_hat = reward.view(-1) + self.discount_factor * (1-done_flag) * (next_Q.view(-1) - self.log_alpha.exp() * log_prob.view(-1))
                #The view(-1) is to ensure that the computation is with vectors and not matrices (so Q_hat.shape = batch_size and not (batch_size,batch_size))

            Q = self.minimal_Q(state, action, pref).view(-1)

            self.Q_loss = F.mse_loss(Q, Q_hat, reduction='mean')

            self.Q1_net.optimizer.zero_grad()
            self.Q2_net.optimizer.zero_grad()
            self.Q_loss.backward()
            self.Q1_net.optimizer.step()
            self.Q2_net.optimizer.step()

            #Update Q target networks
            self.update_target_net_parameters()

        if step % self.update_P == 0:
            #Update P network
            action, log_prob, sigma = self.P_net.sample_normal(state, pref, reparameterize=True)

            self.std = torch.mean(sigma)

            self.entropy = torch.mean(-log_prob)

            Q = self.minimal_Q(state, action, pref).view(-1)

            self.P_loss = torch.mean(self.log_alpha.exp() * log_prob.view(-1) - Q)

            self.P_net.optimizer.zero_grad()
            self.P_loss.backward()
            self.P_net.optimizer.step()

            #Update Alpha

            self.alpha_optimizer.zero_grad()

            Alpha_loss = self.log_alpha.exp() * torch.mean((-log_prob - self.target_entropy).detach())
            Alpha_loss.backward()

            self.alpha_optimizer.step()
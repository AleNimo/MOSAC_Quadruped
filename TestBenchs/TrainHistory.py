import numpy as np
import os

data_type = np.float64

class TrainHistory():
    def __init__(self, max_episodes):
        self.episode = 0
        self.ep_ret = np.zeros((max_episodes, 3), dtype=data_type)      # Returns for each episode (real, expected and RMSE)
        self.ep_loss = np.zeros((max_episodes, 2), dtype=data_type)     # Training loss for each episode (Q and P)
        self.ep_alpha = np.zeros((max_episodes,), dtype=data_type)      # Alpha for each episode
        self.ep_entropy = np.zeros((max_episodes,), dtype=data_type)    # Entropy of the policy for each episode
        self.ep_time = np.zeros((max_episodes,), dtype=data_type)       # Training elapsed time in hours for each episode to use as x_axis instead of steps if needed

    def save(self):
        filename = './Train/Train_History_episode_{0:07d}'.format(self.episode)
        np.savez_compressed(filename, returns = self.ep_ret[0:self.episode+1], loss = self.ep_loss[0:self.episode+1], alpha = self.ep_alpha[0:self.episode+1], entropy = self.ep_entropy[0:self.episode+1], time = self.ep_time[0:self.episode+1])

    def load(self):
        # Check the last episode saved in Progress.txt
        if not os.path.isfile('./Train/Progress.txt'):
            print('Progress.txt could not be found')
            exit
        with open('./Train/Progress.txt', 'r') as file: last_episode = int(np.loadtxt(file))

        filename = './Train/Train_History_episode_{0:07d}.npz'.format(last_episode)
        loaded_arrays = np.load(filename)

        self.ep_ret[0:last_episode+1] = loaded_arrays['returns']
        self.ep_loss[0:last_episode+1] = loaded_arrays['loss']
        self.ep_alpha[0:last_episode+1] = loaded_arrays['alpha']
        self.ep_entropy[0:last_episode+1] = loaded_arrays['entropy']
        self.ep_time[0:last_episode+1] = loaded_arrays['time']
        self.episode = last_episode

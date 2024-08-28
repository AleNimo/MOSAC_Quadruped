import numpy as np
import os

data_type = np.float64

class TrainHistory():
    def __init__(self, max_steps):
        self.global_step = 0

        self.ep_loss = np.zeros((max_steps, 2), dtype=data_type)     # Training loss for each step (Q and P)
        self.ep_alpha = np.zeros((max_steps,), dtype=data_type)      # Alpha for each step
        self.ep_entropy = np.zeros((max_steps,), dtype=data_type)    # Entropy of the policy for each step
        self.ep_time = np.zeros((max_steps,), dtype=data_type)       # Training elapsed time in hours for each step to use as x_axis instead of steps if needed

    def save(self):
        filename = './Train/Train_History_step_{0:07d}'.format(self.global_step)
        np.savez_compressed(filename, loss = self.ep_loss[0:self.global_step+1], alpha = self.ep_alpha[0:self.global_step+1], entropy = self.ep_entropy[0:self.global_step+1], time = self.ep_time[0:self.global_step+1])

    def load(self):
        # Check the last global_step saved in Progress.txt
        if not os.path.isfile('./Train/Progress.txt'):
            print('Progress.txt could not be found')
            exit
        with open('./Train/Progress.txt', 'r') as file: last_global_step = int(np.loadtxt(file))

        filename = './Train/Train_History_step_{0:07d}.npz'.format(last_global_step)
        loaded_arrays = np.load(filename)
        
        self.ep_loss[0:last_global_step+1] = loaded_arrays['loss']
        self.ep_alpha[0:last_global_step+1] = loaded_arrays['alpha']
        self.ep_entropy[0:last_global_step+1] = loaded_arrays['entropy']
        self.ep_time[0:last_global_step+1] = loaded_arrays['time']
        self.global_step = last_global_step

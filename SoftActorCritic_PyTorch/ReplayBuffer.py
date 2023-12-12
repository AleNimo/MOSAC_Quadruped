import numpy as np
import os

class ReplayBuffer():
    def __init__(self, max_size, obs_dim, actions_dim):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, obs_dim))
        self.action_memory = np.zeros((self.mem_size, actions_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, obs_dim))
        self.done_flag_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store(self, state, action, reward, next_state, done_flag):
        if self.mem_counter == self.mem_size:
            print("Replay Buffer max. size reached, overwriting buffer (circular)\n")

        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_flag_memory[index] = done_flag

        self.mem_counter += 1
    
    def sample(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch_index = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch_index]
        actions = self.action_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        next_states = self.next_state_memory[batch_index]
        done_flags = self.done_flag_memory[batch_index]

        return states, actions, rewards, next_states, done_flags
    
    def save(self, episode):
        filename = 'Train/Replay_Buffer_episode_{0:07d}'.format(episode)

        np.savez_compressed(filename, size = self.mem_size, counter = self.mem_counter, state = self.state_memory, action = self.action_memory,
                    reward = self.reward_memory, next_state = self.next_state_memory, done = self.done_flag_memory)
        
        # Update progress file
        with open('./Train/Progress.txt', 'w') as file: np.savetxt(file, np.array((episode,)), fmt='%d')
    
    def load(self, episode):
        filename = './Train/Replay_Buffer_episode_{0:07d}.npz'.format(episode)
        loaded_arrays = np.load(filename)

        self.mem_size = loaded_arrays['size']
        self.mem_counter = loaded_arrays['counter']
        self.state_memory = loaded_arrays['state']
        self.action_memory = loaded_arrays['action']
        self.reward_memory = loaded_arrays['reward']
        self.next_state_memory = loaded_arrays['next_state']
        self.done_flag_memory = loaded_arrays['done']


import numpy as np
from CoppeliaSocket import CoppeliaSocket

class Environment:
    def __init__(self, obs_sp_shape, act_sp_shape, dest_pos):
        '''
        Creates a 3D environment using CoppeliaSim, where an agent capable of choosing its joints' angles tries to find
        the requested destination.
        :param obs_sp_shape: Numpy array's shape of the observed state
        :param act_sp_shape: Numpy array's shape of the action
        :param dest_pos: Destination position that the agent should search
        '''
        self.name = "ComplexAgentSAC"
        self.obs_sp_shape = obs_sp_shape                        # Observation space shape
        self.act_sp_shape = act_sp_shape                        # Action space shape
        self.dest_pos = np.array(dest_pos)                      # Agent's destination
        self.pos_idx = tuple(i for i in range(len(dest_pos)))   # Position's indexes in the observed state array
        self.__pos_size = len(dest_pos)                         # Position's size
        self.__end_cond = 0.1                                   # End condition
        self.__obs = np.zeros((1,)+self.obs_sp_shape)           # Observed state
        self.__coppelia = CoppeliaSocket(obs_sp_shape[0])       # Socket to the simulated environment
        
        # self.__step = 0                                         # Number of steps taken in the episode
        # self.__total_reward_mean = 0                            # Mean of all the rewards before adding penalization for moving violently
        # self.__reward_threshold = 1                             # Threshold at which penalization for moving violently begins (mean step of 0.01)
        # self.__K1 = -3                                       # To penalize proportionally to the excess of reward above the threshold if it moves violently
        # self.__K2 = -9                                       # To penalize a maximum amount if the excess of reward above the threshold is high enough

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Generate a new random starting position
        pos = 2 * np.random.rand(self.__pos_size+1) - 1
        while np.sqrt(np.sum(np.square(pos[0:2]))) < self.__end_cond:
            pos = 2 * np.random.random_sample(self.__pos_size+1) - 1

        # Reset the simulation environment and obtain the new state
        self.__step = 0
        self.__obs = self.__coppelia.reset(pos)
        return np.copy(self.__obs)

    def set_pos(self, pos):
        ''' Sets and returns a new observed state for the environment '''
        # Reset the simulation environment and obtain the new state
        self.__obs = self.__coppelia.reset(pos.reshape(-1))
        return np.copy(self.__obs)

    def get_pos(self):
        ''' Returns the current position of the agent in the environment '''
        # Return the position
        return self.__obs[0:self.__pos_size]

    def act(self, act):
        ''' Simulates the agent's action in the environment, computes and returns the environment's next state, the
        obtained reward and the termination condition status '''
        # Take the requested action in the simulation and obtain the new state
        next_obs = self.__coppelia.act(act.reshape(-1))
        # Compute the reward
        reward, end = self.__compute_reward_and_end(self.__obs.reshape(1,-1), next_obs.reshape(1,-1))
        # Update the observed state
        self.__obs = np.copy(next_obs)
        # Return the environment's next state, the obtained reward and the termination condition status
        return next_obs, reward, end

    def compute_reward(self, obs):
        reward, _ = self.__compute_reward_and_end(obs[0:-1], obs[1:])
        return reward

    def __compute_reward_and_end(self, obs, next_obs):
        #Compute initial and final distance to the target for every individual state

        dist_ini = np.sqrt(np.sum(np.square(obs[:,0:self.__pos_size]), axis=1, keepdims=True))
        dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:self.__pos_size]), axis=1, keepdims=True))
        
        reward, end = np.zeros((dist_ini.shape[0], 1)), np.zeros((dist_ini.shape[0], 1))
        
        for i in range(dist_fin.shape[0]):
            
            #First Compute the reward based only on distance traveled to the target
            
            if dist_fin[i] <= self.__end_cond:
                reward[i], end[i] = 100*(dist_ini[i]), True
            else:
                reward[i], end[i] = 100*(dist_ini[i]-dist_fin[i]), False if dist_fin[i] <= 1.5 else True
            
            # #increase number of steps taken
            # self.__step += 1

            # #compute reward mean of all steps taken
            # self.__total_reward_mean = (self.__total_reward_mean * (self.__step - 1) + reward[i]) / self.__step
            
            # #If the mean of the last 4 rewards based only on distance traveled to the target is:
            #     # <= the reward threshold, the agent is not penalized for moving violently
            #     # > the reward threshold, the agent is penalized for moving violently proportionally to the the excess of reward above the threshold (K1 * (r-thr))
            #     # >> the reward threshold, the agent is penalized for moving violently with a maximum constant multiplied (K2).
            
            # if self.__total_reward_mean >= self.__reward_threshold:
            if reward[i] > 0:
                #Get the 2 angles (x and y axes) of the back
                for j in range(3, 5):
                    back_angle = np.abs(next_obs[i, j])    #angle of the back with respect to 0° (horizontal position)
                    
                    #if the angle is 0° the reward increases 50%, if it is 30° or more it is not increased (The mean of the X,Y angles is computed)
                    if back_angle < 1/6:
                        reward[i] += (1/6-back_angle)*3 * reward[i] * 1/2
    
                #Get angular movement of the 12 joints
                # for joint in range(6, 18):
                #     delta_joint_angle = np.abs(obs[i, joint] - next_obs[i, joint])    #angle of every joint with respect to the previous one
    
                #     if delta_joint_angle > 1/9: #if it exceeds 10°
                #         reward[i] += max(self.__K1*(self.__total_reward_mean-self.__reward_threshold), self.__K2) * delta_joint_angle * 1/12    #Mean multiplied by negative constant
            
            #If the robot flips downwards the episode ends (absolute value of X or Y angle greater than 90°)
            if abs(next_obs[i, 3]) >= 0.5 or abs(next_obs[i, 4]) >= 0.5:
                end[i] = True

        return reward, end

    def max_ret(self, obs):
        ''' Computes and returns the maximum return for the state '''
        return 100*np.sqrt(np.sum(np.square(obs.reshape(1,-1)[:,0:self.__pos_size]), axis=1, keepdims=True))


if __name__ == '__main__':
    from SoftActorCritic import SoftActorCritic

    # Get the environment
    env = Environment(obs_sp_shape=(18,), act_sp_shape=(12,), dest_pos=(0,0))

    # Create the model
#    model = SoftActorCritic("Tetrapod", env, (13, 5), (11, 7, 3), replay_buffer_size=1000000)
    model = SoftActorCritic("Tetrapod", env, (64, 32), (128, 64, 32), replay_buffer_size=1000000)
#    model = SoftActorCritic.load("Tetrapod", env)

    # Set training hyper-parameters
    model.discount_factor = 0.95
    model.update_factor = 0.005
    model.replay_batch_size = 1000 #2
    model.entropy = env.act_sp_shape[0]
    model.initial_alpha = 0.01
    model.H_adam_alpha = 0.001
    model.P_adam_alpha, model.P_train_frequency = 0.001, 3
    model.Q_adam_alpha, model.Q_train_frequency = 0.001, 3
    model.plot_resolution = 10

    # Start training
    model.train(episodes=60000, ep_steps=100, save_period=1000, plot_period=50)

    model.test(ep_steps=100)
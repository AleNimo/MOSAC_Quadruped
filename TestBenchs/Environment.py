import numpy as np
from CoppeliaSocket import CoppeliaSocket

class Environment:
    def __init__(self, sim_measurement, obs_dim, act_dim, rwd_dim):
        '''
        Creates a 3D environment using CoppeliaSim, where an agent capable of choosing its joints' angles tries to find
        the requested destination.
        :param obs_dim: Numpy array's shape of the observed state
        :param act_dim: Numpy array's shape of the action
        :param dest_pos: Destination position that the agent should search
        '''
        self.name = "ComplexAgentSAC"
        self.sim_measurement = sim_measurement                      # Simulation only measurements
        self.obs_dim = obs_dim                                      # Observation dimension
        self.act_dim = act_dim                                      # Action dimension
        self.rwd_dim = rwd_dim                                      # Reward dimension
        self.__end_cond = 11                                        # End condition (radius in meters)
        self.__obs = np.zeros((1,self.obs_dim))                     # Observed state
        self.__coppelia = CoppeliaSocket(obs_dim+sim_measurement)   # Socket to the simulated environment

        #Parameters for forward velocity reward
        self.forward_velocity_reward = 0
        self.__target_velocity = 0.3 # m/s (In the future it could be a changing velocity)
        self.__vmax = 2
        self.__delta_vel = 2 * self.__target_velocity
        self.__vmin = -6

        self.__curvature_forward_vel = - 2* self.__vmax / (self.__delta_vel * self.__vmin)

        #Parameters for forward acceleration penalization
        self.forward_acc_penalty = 0
        self.__max_acc = 4.5 #m/s^2  (Acceleration at which the penalization is -1)

        #Parameters for lateral velocity penalization
        self.lateral_velocity_penalty = 0
        self.__vmin_lat = -2
        self.__curvature_lateral = 3
        
        #Parameters for orientation reward
        self.rotation_penalty = 0
        self.__vmin_rotation = -4
        self.__curvature_rotation = 4
        
        #Parameters for flat back reward
        self.flat_back_penalty = np.zeros(2)
        self.__vmin_back = -2
        self.__curvature_back = 3

        #Reward for not flipping over
        self.__not_flipping_reward = 0.5

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Start position in (0,0) and orientation (in z axis)
        pos = np.zeros(2)
        # z_ang = 2*np.random.rand(1) - 1 #vector of 1 rand between -1 and 1, later multiplied by pi
        z_ang = np.array([0])
        
        # Join position and angle in one vector
        pos_angle = np.concatenate((pos,z_ang))

        # Reset the simulation environment and obtain the new state
        self.__obs = self.__coppelia.reset(pos_angle)
        return np.copy(self.__obs)

    def act(self, act):
        ''' Simulates the agent's action in the environment, computes and returns the environment's next state, the
        obtained reward and the termination condition status '''
        # Take the requested action in the simulation and obtain the new state
        next_obs = self.__coppelia.act(act.reshape(-1))
        # Compute the reward
        reward, end = self.compute_reward_and_end(self.__obs.reshape(1,-1), next_obs.reshape(1,-1))
        # Update the observed state
        self.__obs[:] = next_obs
        # Return the environment's next state, the obtained reward and the termination condition status
        return next_obs, reward, end

    def compute_reward_and_end(self, obs, next_obs):
        # Compute reward for every individual transition (state -> next_state)

            # Final distance to evaluate end condition
        dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:2]), axis=1, keepdims=True))

            # Velocity vector from every state observed
        forward_velocity = next_obs[:,3]
        lateral_velocity = next_obs[:,4]

        max_forward_acceleration = next_obs[:,5]

            # Empty vectors to store reward and end flags for every transition
        reward, end = np.zeros((obs.shape[0], self.rwd_dim)), np.zeros((obs.shape[0], 1))

        for i in range(obs.shape[0]):

            '''Reward for forward velocity reaching target velocity'''
            if forward_velocity[i] > 0:
                self.forward_velocity_reward = (self.__vmax - self.__vmin)/(self.__curvature_forward_vel * np.abs(self.__target_velocity - forward_velocity[i]) + 1) + self.__vmin
            else:
                self.forward_velocity_reward = self.__vmax / self.__target_velocity * forward_velocity[i]

            reward[i,0] = self.forward_velocity_reward

            '''Penalization for peak abs forward acceleration'''
            if max_forward_acceleration[i] < self.__max_acc:
                self.forward_acc_penalty = -max_forward_acceleration[i]/self.__max_acc
            else:
                self.forward_acc_penalty = -np.power(max_forward_acceleration[i], 4)/np.power(self.__max_acc, 4)

            reward[i,1] = self.forward_acc_penalty

            '''Penalization for Lateral velocity'''
            self.lateral_velocity_penalty = -self.__vmin_lat/(self.__curvature_lateral * np.abs(lateral_velocity[i]) + 1) + self.__vmin_lat

            reward[i,2] = self.lateral_velocity_penalty

            '''Penalization for deviating from target step rotation'''
            # compute rotation made in one step:
            agent_rotation = next_obs[i,6] - obs[i,6]
            #correct if there are any discontinuities:
            if (next_obs[i,6] * obs[i,6] < 0) and (np.abs(next_obs[i,6]) > 100*np.pi/180):
                
                if next_obs[i,6] < 0: agent_rotation += 2*np.pi
                else: agent_rotation -= 2*np.pi

            target_rotation = next_obs[i,7] * np.pi

            rotation_error = np.abs(target_rotation - agent_rotation)

            self.rotation_penalty = (-self.__vmin_rotation/(self.__curvature_rotation * rotation_error + 1) + self.__vmin_rotation)

            reward[i,3] = self.rotation_penalty

            '''Flat Back relative reward: pitch and roll close to 0°'''
            for j in range(8, 10):
                back_angle = np.abs(next_obs[i, j])*np.pi    #angle (in rad) of the back with respect to 0° (horizontal position)

                self.flat_back_penalty[j-8] = (-self.__vmin_back/(self.__curvature_back * back_angle + 1) + self.__vmin_back)
                
                reward[i,4] += self.flat_back_penalty[j-8]
                
            '''Reward for avoiding critical failure (flipping over)'''
            reward[i,5] = self.__not_flipping_reward

            #If the absolute value of X or Y angle is greater than 50° there is a penalization and the episode ends
            if abs(next_obs[i, 8])*180 >= 50 or abs(next_obs[i, 9])*180 >= 50:
                reward[i] -= self.__not_flipping_reward
                end[i] = True

            elif dist_fin[i] >= self.__end_cond:
                end[i] = True
            else:
                end[i] = False

        return reward, end
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
        self.__end_cond = 2.5                                   # End condition
        self.__obs = np.zeros((1,)+self.obs_sp_shape)           # Observed state
        self.__coppelia = CoppeliaSocket(obs_sp_shape[0]-2)     # Socket to the simulated environment   (-2 because coppelia doesn't send the target direction)

        self.__target_velocity = 0.3 # m/s (In the future it could be a changing velocity)
        self.__velocity_reward_normalization = 100/(1 - 1/(self.__target_velocity + 1))

        self.__maxBackAngle = 10    #deg

        self.__maxRelativeIncreaseBack = 0.5    #50%

        #Parameters for orientation reward
        self.__maxRelativeIncreaseOrientation = 0.5
        self.__maxRelativeDecreaseOrientation = -0.5
<<<<<<< HEAD
        self.__maxDisorientation = 90 * np.pi / 180
=======
        self.__maxDisorientation = 45 * np.pi / 180
>>>>>>> 9424406d208f30f62f2c3f93d85b958748b3b9cf

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Start position in (0,0) and random orientation (in z axis)
        pos = np.zeros(2)
        z_ang = 2*np.random.rand(1) - 1 #vector of 1 rand between -1 and 1, later multiplied by pi
        
        # Join position and angle in one vector
        pos_angle = np.concatenate((pos,z_ang))

        # Define the random direction for the episode (versor)
        self.target_direction = 2 * np.random.rand(2) - 1
        self.target_direction = self.target_direction / np.sqrt(np.sum(np.square(self.target_direction)))
        
        # Reset the simulation environment and obtain the new state
        self.__step = 0
        obs_coppelia = self.__coppelia.reset(pos_angle)
        self.__obs = np.concatenate((obs_coppelia, self.target_direction))  # We add the target direction to the observed state
        self.__next_obs = np.copy(self.__obs)
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
        self.__next_obs[:self.obs_sp_shape[0]-2] = self.__coppelia.act(act.reshape(-1))   #Replace first 22 values, maintaining the last 2 corresponding to the target direction
        # Compute the reward
        reward, end = self.__compute_reward_and_end(self.__obs.reshape(1,-1), self.__next_obs.reshape(1,-1))
        # Update the observed state
        self.__obs[:] = self.__next_obs
        # Return the environment's next state, the obtained reward and the termination condition status
        return self.__next_obs, reward, end

    def compute_reward(self, obs):
        reward, _ = self.__compute_reward_and_end(obs[0:-1], obs[1:])
        return reward

    def __compute_reward_and_end(self, obs, next_obs):
        # Compute reward for every individual transition (state -> next_state)

            # Final distance to evaluate end condition
        dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:self.__pos_size]), axis=1, keepdims=True))

            # Velocity vector from every state observed
        velocity_vector = next_obs[:,19:21]

            # Empty vectors to store reward and end flags for every transition
        reward, end = np.zeros((obs.shape[0], 1)), np.zeros((obs.shape[0], 1))

<<<<<<< HEAD
=======
        velocity_vector = next_obs[:,19:22]

        #mod_velocity = np.sqrt(np.sum(np.square(velocity_vector), axis=1))

        
>>>>>>> 9424406d208f30f62f2c3f93d85b958748b3b9cf
        for i in range(obs.shape[0]):

            if dist_fin[i] >= self.__end_cond:
                end[i] = True
            else:
                end[i] = False

<<<<<<< HEAD
            '''Velocity reward -> base_reward'''
            velocity_in_target_direction = np.dot(velocity_vector[i], self.target_direction)

            velocity_reward = ( 1/(np.abs(velocity_in_target_direction-self.__target_velocity) + 1) - 1/(self.__target_velocity + 1) ) * self.__velocity_reward_normalization

            reward[i] = velocity_reward
=======
            '''Velocity reward (vel. of the body)'''

            dotproduct_velocidad = np.dot(velocity_vector[i,:-1],self.target_direction)

            vel_reward = ( 1/(np.abs(dotproduct_velocidad-self.target_velocity) + 1) - 1/(self.target_velocity + 1) ) * self.velocity_reward_normalization
           
            base_reward = vel_reward
            reward[i] = base_reward
>>>>>>> 9424406d208f30f62f2c3f93d85b958748b3b9cf

            '''Orientation reward'''
            # Compute angle between agents orientation and target direction:
            agent_orientation = np.array([next_obs[i,5], next_obs[i,6]])

            dot_product = np.dot(agent_orientation, self.target_direction)
            
            # Due to rounding errors we have to check that the dot product doesn't exceed the domain of the arccosine [-1;1]
            if dot_product > 1: angle_agent2target = 0
            elif dot_product < -1: angle_agent2target = np.pi
            else:
                angle_agent2target = np.arccos(dot_product) #Angle between Agent and Center
            
            # Compute reward based on angle:
            if angle_agent2target < self.__maxDisorientation: 
                orientation_reward = self.__maxRelativeIncreaseOrientation * np.cos(angle_agent2target * np.pi/(2*self.__maxDisorientation))
            else:
                orientation_reward = self.__maxRelativeDecreaseOrientation * np.cos((angle_agent2target - np.pi) * np.pi/(2*(np.pi-self.__maxDisorientation) ) )

<<<<<<< HEAD
            # If both rewards are negative, the product will be positive but the reward must decrease
            if velocity_reward < 0 and orientation_reward < 0:
                reward[i] -= orientation_reward * velocity_reward
            else:   
                reward[i] += orientation_reward * velocity_reward
=======
            '''Define base reward'''
            if base_reward < 0 and orientation_reward < 0:    
                reward[i]  -=  orientation_reward * base_reward * 0.5
            
            else:   
                reward[i] += orientation_reward * base_reward * 0.5

>>>>>>> 9424406d208f30f62f2c3f93d85b958748b3b9cf
        
            '''Extra Reward for the 2 angles (x and y axes) of the back close to 0°'''
            for j in range(3, 5):
                back_angle = np.abs(next_obs[i, j])*180    #angle (in deg) of the back with respect to 0° (horizontal position)

                #if the angle is 0° the reward increases a maximumRelativeValue of the base reward
                #if it is __maxBackAngle° or more, base reward is decreased (The mean of the X,Y angles is computed)
                if velocity_reward < 0 and back_angle > self.__maxBackAngle:
                    reward[i] -= (self.__maxBackAngle-back_angle)* self.__maxRelativeIncreaseBack/self.__maxBackAngle * velocity_reward * 1/2
                else:
                    reward[i] += (self.__maxBackAngle-back_angle)* self.__maxRelativeIncreaseBack/self.__maxBackAngle * velocity_reward * 1/2

            #If the robot flips downwards the episode ends (absolute value of X or Y angle greater than 50°)
            if abs(next_obs[i, 3]) >= 0.278 or abs(next_obs[i, 4]) >= 0.278:
                end[i] = True

        return reward, end

    def max_ret(self, obs):
        ''' Computes and returns the maximum return for the state '''
        return 100*np.sqrt(np.sum(np.square(obs.reshape(1,-1)[:,0:self.__pos_size]), axis=1, keepdims=True))
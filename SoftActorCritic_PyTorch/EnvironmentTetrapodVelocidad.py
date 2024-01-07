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

        #Parameters for velocity in target direction reward
        self.__target_velocity = 0.3 # m/s (In the future it could be a changing velocity)
        self.__velocity_reward_normalization = 1/(1 - 1/(self.__target_velocity + 1))

        #Parameters for velocity direction reward
        self.__maxIncreaseVelocityDirection = 0.5
        self.__curvaturePositiveReward_velocity = 30

        self.__neutralAngleVelocity = 5 * np.pi / 180

        self.__maxDecreaseVelocityDirection = -0.5
        self.__curvatureNegativeReward_velocity = 0.5

            #Auxiliary parameters to simplify equation
        self.__b1_vel = -np.exp(-self.__curvaturePositiveReward_velocity*self.__neutralAngleVelocity)
        self.__k1_vel = self.__maxIncreaseVelocityDirection/(1+self.__b1_vel)

        self.__b2_vel = -np.exp(-self.__curvatureNegativeReward_velocity*self.__neutralAngleVelocity)
        self.__k2_vel = self.__maxDecreaseVelocityDirection/(np.exp(-self.__curvatureNegativeReward_velocity*np.pi)+self.__b2_vel)

        #Parameters for orientation reward
        self.__maxIncreaseOrientation = 0.5
        self.__curvaturePositiveReward_orientation = 30

        self.__neutralAngleOrientation = 5 * np.pi / 180

        self.__maxDecreaseOrientation = -0.5
        self.__curvatureNegativeReward_orientation = 0.5

            #Auxiliary parameters to simplify equation
        self.__b1_ori = -np.exp(-self.__curvaturePositiveReward_orientation*self.__neutralAngleOrientation)
        self.__k1_ori = self.__maxIncreaseOrientation/(1+self.__b1_ori)

        self.__b2_ori = -np.exp(-self.__curvatureNegativeReward_orientation*self.__neutralAngleOrientation)
        self.__k2_ori = self.__maxDecreaseOrientation/(np.exp(-self.__curvatureNegativeReward_orientation*np.pi)+self.__b2_ori)
        
        #Parameters for flat back reward
        self.__neutralAngleBack = 10    #deg
        self.__maxIncreaseBack = 0.5    #50%

        #Parameters for flipping penalization
        self.__flipping_penalization = -5

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Start position in (0,0) and random orientation (in z axis)
        pos = np.zeros(2)
        z_ang = 2*np.random.rand(1) - 1 #vector of 1 rand between -1 and 1, later multiplied by pi
        
        # Join position and angle in one vector
        pos_angle = np.concatenate((pos,z_ang))

        # Define the random direction for the episode (versor)
        self.target_direction = 2 * np.random.rand(2) - 1   #(In the future it could be a changing direction, instead of a constant in the episode)
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

        for i in range(obs.shape[0]):

            if dist_fin[i] >= self.__end_cond:
                end[i] = True
            else:
                end[i] = False

            '''Velocity in target direction reward -> base_reward'''
            velocity_in_target_direction = np.dot(velocity_vector[i], self.target_direction)

            velocity_reward = ( 1/(np.abs(velocity_in_target_direction-self.__target_velocity) + 1) - 1/(self.__target_velocity + 1) ) * self.__velocity_reward_normalization

            reward[i] = velocity_reward


            '''Velocity direction reward'''
            # Compute angle between direction of velocity and target direction:
            abs_velocity = np.square(np.sum(np.square(velocity_vector[i])))
            # If the agent isn't moving, we can't divide by the abs_velocity (and there is no direction of velocity to compute)
            # We decide to slightly penalize in this case
            if abs_velocity == 0:
                velocity_direction_reward = self.__maxDecreaseVelocityDirection/2
            else:
                cos_angle_velocity2target = velocity_in_target_direction/abs_velocity
                # Due to rounding errors we have to check that cos_angle_velocity2target doesn't exceed the domain of the arccosine [-1;1]
                if cos_angle_velocity2target > 1: angle_velocity2target = 0
                elif cos_angle_velocity2target < -1: angle_velocity2target = np.pi
                else:
                    angle_velocity2target = np.arccos(cos_angle_velocity2target) #Angle between Agent and Center

                # Compute reward based on angle:
                if angle_velocity2target < self.__neutralAngleVelocity:
                    velocity_direction_reward = self.__k1_vel*(np.exp(-self.__curvaturePositiveReward_velocity * angle_velocity2target) + self.__b1_vel)
                else:
                    velocity_direction_reward = self.__k2_vel*(np.exp(-self.__curvatureNegativeReward_velocity * angle_velocity2target) + self.__b2_vel)


            if velocity_reward < 0 and velocity_direction_reward < 0:
                reward[i] -= velocity_direction_reward * velocity_reward
            else:
                reward[i] += velocity_direction_reward * velocity_reward

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
            if angle_agent2target < self.__neutralAngleOrientation:
                orientation_reward = self.__k1_ori*(np.exp(-self.__curvaturePositiveReward_orientation * angle_agent2target) + self.__b1_ori)
            else:
                orientation_reward = self.__k2_ori*(np.exp(-self.__curvatureNegativeReward_orientation * angle_agent2target) + self.__b2_ori)

            # If both rewards are negative, the product will be positive but the reward must decrease
            if velocity_reward < 0 and orientation_reward < 0:
                reward[i] -= orientation_reward * velocity_direction_reward
            else:   
                reward[i] += orientation_reward * velocity_direction_reward
        
            '''Flat Back reward: extra reward for the 2 angles (x and y axes) of the back close to 0°'''
            for j in range(3, 5):
                back_angle = np.abs(next_obs[i, j])*180    #angle (in deg) of the back with respect to 0° (horizontal position)

                #if the angle is 0° the reward increases a __maxIncreaseBack of the base reward
                #if it is __neutralAngleBack or more, base reward is decreased (The mean of the X,Y angles is computed)
                if velocity_reward < 0 and back_angle > self.__neutralAngleBack:
                    reward[i] -= (self.__neutralAngleBack-back_angle)* self.__maxIncreaseBack/self.__neutralAngleBack * velocity_reward * 1/2
                else:
                    reward[i] += (self.__neutralAngleBack-back_angle)* self.__maxIncreaseBack/self.__neutralAngleBack * velocity_reward * 1/2

            '''Penalization for flipping downwards'''
            #If the absolute value of X or Y angle is greater than 50° there is a penalization and the episode ends
            if abs(next_obs[i, 3]) >= 0.278 or abs(next_obs[i, 4]) >= 0.278:
                reward[i] += self.__flipping_penalization
                end[i] = True

        return reward, end

    def max_ret(self, obs):
        ''' Computes and returns the maximum return for the state '''
        return 100*np.sqrt(np.sum(np.square(obs.reshape(1,-1)[:,0:self.__pos_size]), axis=1, keepdims=True))
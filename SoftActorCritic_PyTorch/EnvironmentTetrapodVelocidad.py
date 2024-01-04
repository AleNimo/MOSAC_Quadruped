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
        self.__coppelia = CoppeliaSocket(obs_sp_shape[0])       # Socket to the simulated environment

        self.target_velocity = 0.3 # m/s (In the future it could be a changing velocity)
        self.velocity_reward_normalization = 100/(1 - 1/(self.target_velocity + 1))

        self.__maxBackAngle = 10    #deg
        
        self.__maxRelativeIncreaseBack = 0.5    #50%
        
        #Parameters for orientation reward
        self.__maxRelativeIncreaseOrientation = 1
        self.__maxRelativeDecreaseOrientation = -0.5
        self.__maxDisorientation = 45 * np.pi / 180

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Start position in (0,0) and random orientation (in z axis)
        pos = np.zeros(2)
        z_ang = 2*np.random.rand(1) - 1 #vector of 1 rand between -1 and 1, later multiplied by pi
        
        # Join position and angle in one vector
        pos_angle = np.concatenate((pos,z_ang))

        # Define the random direction for the episode
        self.target_direction = 2 * np.random.rand(2) - 1
        self.target_direction = self.target_direction / np.sqrt(np.sum(np.square(self.target_direction)))
        
        # Reset the simulation environment and obtain the new state
        self.__step = 0
        self.__obs = self.__coppelia.reset(pos_angle)
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
        
        # Final distance to evaluate end condition
        dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:self.__pos_size]), axis=1, keepdims=True))

        # Compute reward for every individual state observed

        reward, end = np.zeros((obs.shape[0], 1)), np.zeros((obs.shape[0], 1))

        velocity_vector = next_obs[:,19:22]

        #mod_velocity = np.sqrt(np.sum(np.square(velocity_vector), axis=1))

        
        for i in range(obs.shape[0]):
            
            if dist_fin[i] >= self.__end_cond:
                end[i] = True
            else:
                end[i] = False

            '''Velocity reward (vel. of the body)'''

            dotproduct_velocidad = np.dot(velocity_vector[i,:-1],self.target_direction)

            vel_reward = ( 1/(np.abs(dotproduct_velocidad-self.target_velocity) + 1) - 1/(self.target_velocity + 1) ) * self.velocity_reward_normalization
           
            base_reward = vel_reward
            reward[i] = base_reward

            '''Orientation reward'''
            # Agent's orientation vector:
            Agent = np.array([next_obs[i,5], next_obs[i,6]])

            dotproduct = np.dot(Agent,self.target_direction)
            
            # Due to rounding errors we have to check that the dotproduct doesn't exceed the domain of the arcosine [-1;1]
            if dotproduct > 1: angle_agent2target = 0
            elif dotproduct < -1: angle_agent2target = np.pi
            else:
                angle_agent2target = np.arccos(dotproduct) #Angle between Agent and Center
            
            if angle_agent2target < self.__maxDisorientation: 
                orientation_reward = self.__maxRelativeIncreaseOrientation * np.cos(angle_agent2target * np.pi/(2*self.__maxDisorientation))
            else:
                orientation_reward = self.__maxRelativeDecreaseOrientation * np.cos((angle_agent2target - np.pi) * np.pi/(2*(np.pi-self.__maxDisorientation) ) )

            '''Define base reward'''
            if base_reward < 0 and orientation_reward < 0:    
                reward[i]  -=  orientation_reward * base_reward * 0.5
            
            else:   
                reward[i] += orientation_reward * base_reward * 0.5

        
            '''Extra Reward for the 2 angles (x and y axes) of the back close to 0°'''
            for j in range(3, 5):
                back_angle = np.abs(next_obs[i, j])*180    #angle (in deg) of the back with respect to 0° (horizontal position)

                #if the angle is 0° the reward increases a maximumRelativeValue of the base reward
                #if it is __maxBackAngle° or more, base reward is decreased (The mean of the X,Y angles is computed)
                if base_reward < 0 and back_angle > self.__maxBackAngle:
                    reward[i] -= (self.__maxBackAngle-back_angle)* self.__maxRelativeIncreaseBack/self.__maxBackAngle * base_reward * 1/2
                else:
                    reward[i] += (self.__maxBackAngle-back_angle)* self.__maxRelativeIncreaseBack/self.__maxBackAngle * base_reward * 1/2


            #If the robot flips downwards the episode ends (absolute value of X or Y angle greater than 50°)
            if abs(next_obs[i, 3]) >= 0.278 or abs(next_obs[i, 4]) >= 0.278:
                end[i] = True

        return reward, end

    def max_ret(self, obs):
        ''' Computes and returns the maximum return for the state '''
        return 100*np.sqrt(np.sum(np.square(obs.reshape(1,-1)[:,0:self.__pos_size]), axis=1, keepdims=True))
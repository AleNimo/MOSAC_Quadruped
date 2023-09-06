import numpy as np
from CoppeliaSocket import CoppeliaSocket

#Cuadruped absolute joint limits:
# bod_llim, bod_ulim = -15,  60
# leg_llim, leg_ulim = -40, 130
# paw_llim, paw_ulim = -90,  30

#Cuadruped preferred joint limits for training:
bod_min, bod_max = -15, 20
leg_min, leg_max = 0, 60
paw_min, paw_max = -60, -30

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
        
        self.__maxBackAngle = 10    #deg
        #For Tetrapod:
        #self.__joint_midpoint = 0
        #self.__joint_range = 45 #+/-
        
        #For cuadruped (angles in deg):
        self.__joint_midpoint = np.array([(bod_max + bod_min)/2, (leg_max + leg_min)/2, (paw_max + paw_min)/2])
        self.__joint_range =    np.array([(bod_max - bod_min)/2, (leg_max - leg_min)/2, (paw_max - paw_min)/2]) #(+/-)
        
        #self.__maxJointAngle = 40
        
        self.__maxRelativeIncreaseBack = 1
        self.__maxRelativeIncreaseJoint = 0.5
        
        
        #Calculation of parameters for orientation reward
        self.maxRelativeIncreaseOrientation = 1.3
        self.max_disorientation = 90    #CAUTION: if the angle is near 0, the function punishes very hard the agent
        
        self.A = self.maxRelativeIncreaseOrientation/(1-np.cos(self.max_disorientation*np.pi/180))
        self.B = self.A-self.maxRelativeIncreaseOrientation
        
        #Parameters for not jumping reward
        self.__maxRelativeDecreaseJumping = 1
        self.__maxRelativeIncrease_notJumping = 0.3

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Generate a new random starting position (x and y) and orientation (in z axis)
        pos = 4 * np.random.rand(self.__pos_size) - 2   #vector of 2 rand between -2 and 2
        z_ang = 2*np.random.rand(1) - 1 #vector of 1 rand between -1 and 1, later multiplied by pi
        
        #Make sure the start position is outside the center of the map (end condition)
        while np.sqrt(np.sum(np.square(pos))) < self.__end_cond:
            pos = 4 * np.random.random_sample(self.__pos_size) - 2

        # Join position and angle in one vector
        pos_angle = np.concatenate((pos,z_ang))
        
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
        #Compute initial and final distance to the target for every individual state

        dist_ini = np.sqrt(np.sum(np.square(obs[:,0:self.__pos_size]), axis=1, keepdims=True))
        dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:self.__pos_size]), axis=1, keepdims=True))
        
        reward, base_reward, end = np.zeros((dist_ini.shape[0], 1)), np.zeros((dist_ini.shape[0], 1)), np.zeros((dist_ini.shape[0], 1))
        
        for i in range(dist_fin.shape[0]):
            
            '''First Compute the reward based only on distance traveled to the target ("Base reward")'''
            
            if dist_fin[i] <= self.__end_cond:
                base_reward[i], end[i] = 100*(dist_ini[i]), True
            else:
                base_reward[i], end[i] = 100*(dist_ini[i]-dist_fin[i]), False if dist_fin[i] <= 3 else True
            
            reward[i] = base_reward[i]
            
            '''Extra Reward for the 2 angles (x and y axes) of the back close to 0°'''
#            print("reward before back analysis: ", reward[i])
            for j in range(3, 5):
                back_angle = np.abs(next_obs[i, j])*180    #angle (in deg) of the back with respect to 0° (horizontal position)
#                print(f"back_angle {j-2}: {back_angle}")
                #if the angle is 0° the reward increases a maximumRelativeValue of the base reward
                #if it is __maxBackAngle° or more, base reward is decreased (The mean of the X,Y angles is computed)
                if base_reward[i] < 0 and back_angle > self.__maxBackAngle:
                    reward[i] -= (self.__maxBackAngle-back_angle)* self.__maxRelativeIncreaseBack/self.__maxBackAngle * base_reward[i] * 1/2
                else:
                    reward[i] += (self.__maxBackAngle-back_angle)* self.__maxRelativeIncreaseBack/self.__maxBackAngle * base_reward[i] * 1/2
#            print("reward after back analysis: ", reward[i])
            
            '''Extra Reward analizing the biggest joint movement of the agent'''
            # biggest_joint = 0
            #Find the biggest joint movement of the agent
#            print("Separación")
            # for joint in range(7, 19):
                #Convert coppelia output angle from -1 to 1, to value in deg
                # joint_angle = np.abs(next_obs[i, joint]*self.__joint_range[(joint-7)%3] + self.__joint_midpoint[(joint-7)%3]) #abs angle of every joint in deg
#                print(f"Joint {joint-6}: {joint_angle}")
                # if joint_angle > biggest_joint:
                #     biggest_joint = joint_angle
            
            #if the joint movement is lower than maxJointAngle, the reward increases up to maximumRelativeValue of the base reward
            #else it is decreased
            
#            print("reward before biggest joint analysis: ", reward[i])
            # if base_reward[i] < 0 and biggest_joint > self.__maxJointAngle:
            #     reward[i] -= (self.__maxJointAngle-biggest_joint)* self.__maxRelativeIncreaseJoint/self.__maxJointAngle * base_reward[i]
            # else:
            #     reward[i] += (self.__maxJointAngle-biggest_joint)* self.__maxRelativeIncreaseJoint/self.__maxJointAngle * base_reward[i]
#            print("reward after biggest joint analysis: ", reward[i])

            '''Extra Reward for the agent looking to the center'''

            #Agent's orientation vector:
            Agent = np.array([next_obs[i,5], next_obs[i,6]])
            
            #Vector to the center:
            x = next_obs[i,0]
            y = next_obs[i,1]
            Center = -np.array([x,y])/np.sqrt(x**2 + y**2)
            
            dot_product = np.dot(Agent, Center) #Equal to cosine of angle between Agent and Center
  
            K = (self.A*dot_product-self.B)
            
            if base_reward[i] < 0 and K < 0:    
                reward[i] -= K * base_reward[i]
            
            else:   
                reward[i] += K * base_reward[i]

            '''Extra Reward for the agent moving the legs in different directions (to avoid moving jumping forward)'''
            #Read movement of the front leg joints:
            delta_front_right = (next_obs[i,8] - obs[i,8]) * self.__joint_range[1]
            delta_front_left = (next_obs[i,11] - obs[i,11]) * self.__joint_range[1]
            
            mod_delta_front_right = np.abs(delta_front_right)
            mod_delta_front_left = np.abs(delta_front_left)
            mod_greater_delta_front = np.maximum(mod_delta_front_right, mod_delta_front_left)
            
            #Read movement of the back leg joints:
            delta_back_right = (next_obs[i,14] - obs[i,14]) * self.__joint_range[1]
            delta_back_left = (next_obs[i,17] - obs[i,17]) * self.__joint_range[1]
            
            mod_delta_back_right = np.abs(delta_back_right)
            mod_delta_back_left = np.abs(delta_back_left)
            mod_greater_delta_back = np.maximum(mod_delta_back_right, mod_delta_back_left)

            if mod_greater_delta_front != 0:
                
                if delta_front_left * delta_front_right > 0:  #Same direction of movement, punishment for jumping
                    
                    punishment_front = np.abs(delta_front_right - delta_front_left) / mod_greater_delta_front - 1
                    
                    if base_reward[i] < 0:
                        reward[i] -= self.__maxRelativeDecreaseJumping * punishment_front * base_reward[i]
                    else:
                        reward[i] += self.__maxRelativeDecreaseJumping * punishment_front * base_reward[i]
                    
                else:   #Diferent direction of movement, reward for not jumping and moving simetrically with respect to 0° (less important)
                    
                    reward_front = 1 - np.abs(mod_delta_front_right - mod_delta_front_left) / mod_greater_delta_front
                    reward[i] += self.__maxRelativeIncrease_notJumping * reward_front * base_reward[i]

                
            #Apply punishment or reward to back legs
            if mod_greater_delta_back != 0:
                
                if delta_back_left * delta_back_right > 0:  #Same direction of movement, punishment for jumping
                    
                    punishment_back = np.abs(delta_back_right - delta_back_left) / mod_greater_delta_back - 1
                    
                    if base_reward[i] < 0:
                        reward[i] -= self.__maxRelativeDecreaseJumping * punishment_back * base_reward[i]
                    else:
                        reward[i] += self.__maxRelativeDecreaseJumping * punishment_back * base_reward[i]
                    
                else:   #Diferent direction of movement, reward for not jumping and moving simetrically with respect to 0° (less important)
                    
                    reward_back = 1 - np.abs(mod_delta_back_right - mod_delta_back_left) / mod_greater_delta_back
                    reward[i] += self.__maxRelativeIncrease_notJumping * reward_back * base_reward[i]
                
            #If the robot flips downwards the episode ends (absolute value of X or Y angle greater than 50°)
            if abs(next_obs[i, 3]) >= 0.278 or abs(next_obs[i, 4]) >= 0.278:
                end[i] = True

        return reward, end

    def max_ret(self, obs):
        ''' Computes and returns the maximum return for the state '''
        return 100*np.sqrt(np.sum(np.square(obs.reshape(1,-1)[:,0:self.__pos_size]), axis=1, keepdims=True))


if __name__ == '__main__':
    from SoftActorCritic import SoftActorCritic

    # Get the environment
    env = Environment(obs_sp_shape=(19,), act_sp_shape=(12,), dest_pos=(0,0))

    # Create the model
#    model = SoftActorCritic("Cuadruped", env, (13, 5), (11, 7, 3), replay_buffer_size=1000000)
#    model = SoftActorCritic("Cuadruped", env, (64, 32), (128, 64, 32), replay_buffer_size=1000000)
    model = SoftActorCritic.load("Cuadruped", env, emptyReplayBuffer = True)

    # Set training hyper-parameters
    model.discount_factor = 0.94
    model.update_factor = 0.005
    model.replay_batch_size = 1000 #2
    model.entropy = env.act_sp_shape[0]
    model.initial_alpha = 0.01
    model.H_adam_alpha = 0.001
    model.P_adam_alpha, model.P_train_frequency = 0.001, 3
    model.Q_adam_alpha, model.Q_train_frequency = 0.001, 3
    model.plot_resolution = 10

    # Start training
    model.train(episodes=100000, ep_steps=200, save_period=1000, plot_period=50)

    model.test(ep_steps=100)
import numpy as np
from CoppeliaSocket import CoppeliaSocket

#Cuadruped absolute joint limits:
# bod_llim, bod_ulim = -15,  60
# leg_llim, leg_ulim = -40, 130
# paw_llim, paw_ulim = -90,  30

#Cuadruped preferred joint limits for training:
bod_min, bod_max = -10, 15 #inicial 0
leg_min, leg_max = -10, 40 #inicial 0
paw_min, paw_max = -15, 5  #inicial 0

# piñe puso:                    inicial
# bod_min, bod_max = -15, 10    (0)
# leg_min, leg_max = -40, -10  (-10)
# paw_min, paw_max = 10, 20     (20)

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
        
        self.__maxRelativeIncreaseBack = 1.5
        #self.__maxRelativeIncreaseJoint = 0.5
        
        
        #Calculation of parameter for orientation reward
        self.__maxRelativeIncreaseOrientation = 1
        self.__maxRelativeDecreaseOrientation = -2
        self.__maxDisorientation = 10 * np.pi / 180
        
        #Parameters for not jumping reward
        self.__maxRelativeIncrease_notJumping = 0.5
        self.__maxRelativeDecrease_Jumping = -1
        
        self.__maxRelativeIncrease_crossWalking = 0.5
        self.__maxRelativeDecrease_notCrossWalking = -1

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
            pos = np.array([x, y])
            Center = -pos/np.sqrt(np.sum(np.square(pos)))
            
            dotproduct = np.dot(Agent,Center)
            
            #Due to rounding errors we have to check that the dotproduct doesn't exceed the domain of the arcosine [-1;1]
            if dotproduct > 1: angle_agent_center = 0
            elif dotproduct < -1: angle_agent_center = np.pi
            else:
                angle_agent_center = np.arccos(dotproduct) #Angle between Agent and Center
            
            if angle_agent_center < self.__maxDisorientation: 
                K = self.__maxRelativeIncreaseOrientation * np.cos(angle_agent_center * np.pi/(2*self.__maxDisorientation))
            else:
                K = self.__maxRelativeDecreaseOrientation * np.cos((angle_agent_center - np.pi) * np.pi/(2*(np.pi-self.__maxDisorientation) ) )
            
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
                    
                    punishment_front = 1 - np.abs(delta_front_right - delta_front_left) / mod_greater_delta_front
                    
                    if base_reward[i] < 0:
                        reward[i] -= self.__maxRelativeDecrease_Jumping * punishment_front * base_reward[i]
                    else:
                        reward[i] += self.__maxRelativeDecrease_Jumping * punishment_front * base_reward[i]
                    
                else:   #Different direction of movement, reward for not jumping and moving simetrically with respect to 0° (less important)
                    
                    reward_front = 1 - np.abs(mod_delta_front_right - mod_delta_front_left) / mod_greater_delta_front
                    reward[i] += self.__maxRelativeIncrease_notJumping * reward_front * base_reward[i]

                
            #Apply punishment or reward to back legs
            if mod_greater_delta_back != 0:
                
                if delta_back_left * delta_back_right > 0:  #Same direction of movement, punishment for jumping
                    
                    punishment_back = 1 - np.abs(delta_back_right - delta_back_left) / mod_greater_delta_back
                    
                    if base_reward[i] < 0:
                        reward[i] -= self.__maxRelativeDecrease_Jumping * punishment_back * base_reward[i]
                    else:
                        reward[i] += self.__maxRelativeDecrease_Jumping * punishment_back * base_reward[i]
                    
                else:   #Different direction of movement, reward for not jumping and moving simetrically with respect to 0° (less important)
                    
                    reward_back = 1 - np.abs(mod_delta_back_right - mod_delta_back_left) / mod_greater_delta_back
                    reward[i] += self.__maxRelativeIncrease_notJumping * reward_back * base_reward[i]
                
            '''Extra Reward for the agent moving the crossed legs and paws in the same way'''
            #version 2 con deltas
            mod_greater_delta_crossed_1 = np.maximum(mod_delta_front_right, mod_delta_back_left)
            mod_greater_delta_crossed_2 = np.maximum(mod_delta_front_left, mod_delta_back_right)
            
            if mod_greater_delta_crossed_1 != 0:
                if delta_front_right * delta_back_left < 0: #different direction of crossed legs 1, punishment
                
                    punishment_crossed_1 = 1 - np.abs(mod_delta_front_right - mod_delta_back_left)/mod_greater_delta_crossed_1
                    
                    if base_reward[i] < 0:
                        reward[i] -= self.__maxRelativeDecrease_notCrossWalking * punishment_crossed_1 * base_reward[i]
                    else:
                        reward[i] += self.__maxRelativeDecrease_notCrossWalking * punishment_crossed_1 * base_reward[i]
                
                else:   #Same direction of crossed legs 1, reward
                    
                    reward_crossed_1 = 1 - np.abs(delta_front_right - delta_back_left)/mod_greater_delta_crossed_1
                    reward[i] += self.__maxRelativeIncrease_crossWalking * reward_crossed_1 * base_reward[i]
            
            if mod_greater_delta_crossed_2 != 0:
                if delta_front_left * delta_back_right < 0: #different direction of crossed legs 2, punishment
                
                    punishment_crossed_2 = 1 - np.abs(mod_delta_front_left - mod_delta_back_right)/mod_greater_delta_crossed_2
                    
                    if base_reward[i] < 0:
                        reward[i] -= self.__maxRelativeDecrease_notCrossWalking * punishment_crossed_2 * base_reward[i]
                    else:
                        reward[i] += self.__maxRelativeDecrease_notCrossWalking * punishment_crossed_2 * base_reward[i]
                
                else:   #Same direction of crossed legs 2, reward
                    
                    reward_crossed_2 = 1 - np.abs(delta_front_left - delta_back_right)/mod_greater_delta_crossed_2
                    reward[i] += self.__maxRelativeIncrease_crossWalking * reward_crossed_2 * base_reward[i]
            #Read leg angles:
            # front_right_leg = (next_obs[i,8] - self.__joint_midpoint[1]) * self.__joint_range[1]
            # front_left_leg = (next_obs[i,11] - self.__joint_midpoint[1]) * self.__joint_range[1]
            # back_right_leg = (next_obs[i,14] - self.__joint_midpoint[1]) * self.__joint_range[1]
            # back_left_leg  = (next_obs[i,17] - self.__joint_midpoint[1]) * self.__joint_range[1]
            
            # #greater absolute angle bewteen crossed legs
            # mod_greater_cross_legs_1 = np.maximum(np.abs(front_right_leg), np.abs(back_left_leg))
            # mod_greater_cross_legs_2 = np.maximum(np.abs(front_left_leg), np.abs(back_right_leg))
            
            # #Read paw angles:
            # front_right_paw = (next_obs[i,9] - self.__joint_midpoint[2]) * self.__joint_range[2]
            # front_left_paw = (next_obs[i,12] - self.__joint_midpoint[2]) * self.__joint_range[2]
            # back_right_paw = (next_obs[i,15] - self.__joint_midpoint[2]) * self.__joint_range[2]
            # back_left_paw  = (next_obs[i,18] - self.__joint_midpoint[2]) * self.__joint_range[2]
            
            # #greater absolute angle bewteen crossed paws
            # mod_greater_cross_paws_1 = np.maximum(np.abs(front_right_paw), np.abs(back_left_paw))
            # mod_greater_cross_paws_2 = np.maximum(np.abs(front_left_paw), np.abs(back_right_paw))
                    
            # #Reward if cross side pair of legs move identically
            # if mod_greater_cross_legs_1 !=0:

            #     reward_cross_legs_1 = 1-np.abs(front_right_leg-back_left_leg)/(2*mod_greater_cross_legs_1)
            #     reward[i] += self.__maxRelativeIncrease_crossWalking * reward_cross_legs_1 * base_reward[i]


            # if mod_greater_cross_legs_2 !=0:

            #     reward_cross_legs_2 = 1-np.abs(front_left_leg-back_right_leg)/(2*mod_greater_cross_legs_2)
            #     reward[i] += self.__maxRelativeIncrease_crossWalking * reward_cross_legs_2 * base_reward[i]
                
            # #Reward if cross side pair of paws move identically
            # if mod_greater_cross_paws_1 !=0:

            #     reward_cross_paws_1 = 1-np.abs(front_right_paw-back_left_paw)/(2*mod_greater_cross_paws_1)
            #     reward[i] += self.__maxRelativeIncrease_crossWalking * reward_cross_paws_1 * base_reward[i]


            # if mod_greater_cross_paws_2 !=0:

            #     reward_cross_paws_2 = 1-np.abs(front_left_paw-back_right_paw)/(2*mod_greater_cross_paws_2)
            #     reward[i] += self.__maxRelativeIncrease_crossWalking * reward_cross_paws_2 * base_reward[i]
                
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
    model = SoftActorCritic("Cuadruped", env, (64, 32), (128, 64, 32), replay_buffer_size=1000000)
#    model = SoftActorCritic.load("Cuadruped", env, emptyReplayBuffer = False)

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
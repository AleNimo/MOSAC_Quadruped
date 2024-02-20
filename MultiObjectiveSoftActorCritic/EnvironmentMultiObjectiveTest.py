import numpy as np

class Environment:
    def __init__(self):
        ''' Creates a 2D environment, where an agent capable of choosing its movement angle tries to find the origin '''
        self.name = "SimpleMultiObjectiveEnvironment"
        self.obs_sp_shape = (7,)                        # Observation space shape [cos(a_dst), sin(a_dst), d_dst, cos(a_vst), sin(a_vst), d_vst]
        self.act_sp_shape = (2,)                        # Action space shape [cos(a_act), sin(a_act)]
        self.rwd_sp_shape = (1,)                        # Reward space shape [empty]
        self.prf_sp_shape = (3,)                        # Preference space shape [u_dst, u_vst]
        self.dst_pos = np.array([0, 0])                 # Agent's final destination
        self.int_pos = np.array([0, 0])                 # Agent's intermediate destination
        self.bgn_pos = np.array([0, 0])                 # Agent's beginning position
        self.__end_cond = 0.1                           # End condition
        self.__obs = np.zeros((1,)+self.obs_sp_shape)   # Observed state
        self.__pos_size = 2

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        d = 0
        while (d <= self.__end_cond) or (d >= 1.5):
            pos = 3 * np.random.random_sample((1, self.__pos_size)) - 1.5
            d = np.sqrt(np.sum(np.square(pos)))
        d = 0
        while (d <= self.__end_cond) or (d >= 1.5):
            self.int_pos = 3 * np.random.random_sample(self.__pos_size) - 1.5
            d = np.sqrt(np.sum(np.square(self.int_pos)))
        return self.set_pos(pos)

    def set_pos(self, pos):
        ''' Sets and returns a new observed state for the environment '''
        # Reshape the position array
        pos = np.array(pos).reshape((-1, self.__pos_size))
        obs = np.zeros((pos.shape[0],)+self.obs_sp_shape)
        # Positional variables wrt the objective
        delta = pos - self.dst_pos
        obs[:, 0] = np.sqrt(np.sum(np.square(delta), axis=1))
        obs[:, 1:1+self.__pos_size] = np.divide(delta, obs[:, 0:1])
        # Positional variables wrt the sub-objective
        delta = pos - self.int_pos
        obs[:, 1+self.__pos_size] = np.sqrt(np.sum(np.square(delta), axis=1))
        obs[:, 2+self.__pos_size:2+2*self.__pos_size] = np.divide(delta, obs[:, 1+self.__pos_size:2+self.__pos_size])
        # Special condition reward
        obs[:, 2+2*self.__pos_size] = 1
        # Copy the first entry as the environment's observed state
        self.__obs = obs[0:1, :]
        return obs

    def get_pos(self, obs=None):
        ''' Returns the positions of the agent in the environment corresponding to the observed states (current observed state if none is passed) '''
        if obs is None: obs = self.__obs
        return self.dst_pos + np.multiply(obs[:, 0:1], obs[:, 1:1+self.__pos_size])

    def act(self, act):
        ''' Simulates the agent's action in the environment, computes and returns the environment's next state, the obtained reward and termination condition status '''
        act = np.copy(act).reshape((1,)+self.act_sp_shape) / np.sqrt(np.sum(np.square(act)))
        # Distance to objective
        delta = np.multiply(self.__obs[:, 0], self.__obs[:, 1:1+self.__pos_size]) + 0.1 * act
        self.__obs[:, 0] = np.sqrt(np.sum(np.square(delta), axis=1))
        self.__obs[:, 1:1+self.__pos_size] = delta/self.__obs[0, 0]
        end_condition = (self.__obs[:, 0] <= self.__end_cond) or (self.__obs[:, 0] > 1.5)
        # Return if sub-objective has been completed
        if self.__obs[:, 1+self.__pos_size] <= self.__end_cond:
            self.__obs[:, 1+self.__pos_size:2+2*self.__pos_size] = np.zeros(1+self.__pos_size)
            return np.copy(self.__obs), 0, end_condition
        # Distance to sub-objective
        delta = np.multiply(self.__obs[:, 1+self.__pos_size], self.__obs[:, 2+self.__pos_size:2+2*self.__pos_size]) + 0.1 * act
        self.__obs[:, 1+self.__pos_size] = np.sqrt(np.sum(np.square(delta), axis=1))
        self.__obs[:, 2+self.__pos_size:2+2*self.__pos_size] = delta/self.__obs[:, 1+self.__pos_size]
        return np.copy(self.__obs), 0, end_condition

    def compute_observed_state(self, obs, act, next_obs, prf):
        act = np.copy(act).reshape((-1,) + self.act_sp_shape)
        act = np.divide(act, np.sqrt(np.sum(np.square(act), axis=1, keepdims=True)))
        obs = np.copy(obs).reshape((-1,) + self.obs_sp_shape)
        next_obs = np.copy(next_obs).reshape((-1,) + self.obs_sp_shape)
        prf= np.copy(prf).reshape((-1,) + self.prf_sp_shape)
        obs[:, 2+2*self.__pos_size] = np.random.random_sample(obs.shape[0]).reshape((-1,))
        next_obs[:, 2+2*self.__pos_size] = np.min(np.concatenate((obs[:, 2+2*self.__pos_size].reshape(-1,1), np.exp(-np.square(np.min(np.abs(act), axis=1, keepdims=True)[:,0] / prf[:, 2])).reshape(-1,1)), axis=1), axis=1)
        return obs, next_obs

    def compute_rwd(self, obs, act, next_obs, rwd, prf):
        '''
        :param obs: list of initial observed states
        :param act: list of taken actions
        :param next_obs: list of final observed states
        :param rwd: list of obtained rewards
        :param prf: list of preferences
        :return: A tuple containing the total reward obtained from the state transition under the selected set of preferences
        '''
        obs = np.copy(obs).reshape((-1,) + self.obs_sp_shape)
        act = np.copy(act).reshape((-1,) + self.act_sp_shape)
        act = np.divide(act, np.sqrt(np.sum(np.square(act), axis=1, keepdims=True)))
        next_obs = np.copy(next_obs).reshape((-1,) + self.obs_sp_shape)
        rwd = np.copy(rwd).reshape((-1,) + self.rwd_sp_shape)
        prf= np.copy(prf).reshape((-1,) + self.prf_sp_shape)
        reward = ((np.multiply(prf[:, 0], (obs[:, 0] - np.multiply(next_obs[:, 0], next_obs[:, 0] > self.__end_cond))) + np.multiply(1-prf[:, 0], (obs[:, 1+self.__pos_size] - next_obs[:, 1+self.__pos_size])))
#        prf[:, 0:2] = np.exp(-3*prf[:, 0:2])
#        prf[:, 1] = np.multiply(1-prf[:,0], prf[:,1])
#        reward = ((np.multiply(prf[:, 0], (obs[:, 0] - np.multiply(next_obs[:, 0], next_obs[:, 0] > self.__end_cond))) + np.multiply(prf[:, 1], (obs[:, 1+self.__pos_size] - next_obs[:, 1+self.__pos_size]))) + ((1-prf[:,0]-prf[:,1]) * 0.1 * np.exp(-np.square(np.min(np.abs(act), axis=1, keepdims=True)[:,0] / prf[:, 2]))))
#        reward = (((1+np.multiply(prf[:, 0], (obs[:, 0] - np.multiply(next_obs[:, 0], next_obs[:, 0] > self.__end_cond)))) * (1+np.multiply(prf[:, 1], (obs[:, 1+self.__pos_size] - next_obs[:, 1+self.__pos_size])))) + (1+((1-prf[:,0]-prf[:,1]) * 0.1 * (np.min(np.abs(act), axis=1, keepdims=True)[:,0] <= prf[:, 2]))))
#        reward = ((np.multiply(prf[:, 0], (obs[:, 0] - np.multiply(next_obs[:, 0], next_obs[:, 0] > self.__end_cond))) + np.multiply(prf[:, 1], (obs[:, 1+self.__pos_size] - next_obs[:, 1+self.__pos_size]))) + ((1-prf[:,0]-prf[:,1]) * 4 * np.multiply(next_obs[:, 2+2*self.__pos_size], next_obs[:, 0] <= self.__end_cond)))
        return reward.reshape((-1,1))



if __name__ == '__main__':
    from MultiObjectiveSoftActorCritic import SoftActorCritic

    # Get the environment
    env = Environment()

    # Create the model
    model = SoftActorCritic("TestMultiObj", env, (64, 32), (128, 64, 32), replay_buffer_size=10000, seed=123)
#    model = SoftActorCritic.load("TestMultiObj", env)

    # Set training hyper-parameters
    model.discount_factor = 0.95
    model.update_factor = 0.005
    model.replay_batch_size = 1000
    model.entropy = env.act_sp_shape[0]
    model.initial_alpha = 0.1
    model.H_adam_alpha = 0.001
    model.P_adam_alpha, model.P_train_frequency = 0.002, 1
    model.Q_adam_alpha, model.Q_train_frequency = 0.001, 1
    model.plot_resolution = 30

    # Start training
    model.train(episodes=500, ep_steps=100, save_period=1000, plot_period=50)

    model.test(ep_steps=100, preferences=(0.5, 0.95, 0.01))



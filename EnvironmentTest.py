import numpy as np

class Environment:
    def __init__(self):
        ''' Creates a 2D environment, where an agent capable of choosing its movement angle tries to find the origin '''
        self.name = "SimpleEnvironment"
        self.obs_sp_shape = (2,)                        # Observation space shape
        self.act_sp_shape = (1,)                        # Action space shape
        self.dest_pos = np.array([0, 0])                # Agent's destination
        self.pos_idx = (0, 1)                           # Position's indexes in the observed state array
        self.__end_cond = 0.1                           # End condition
        self.__obs = np.zeros((1,)+self.obs_sp_shape)   # Observed state
        self.__pos_size = 2

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        obs = 2 * np.random.random_sample((1,)+self.obs_sp_shape) - 1
        while np.sqrt(np.sum(np.square(obs[0]))) < self.__end_cond:
            obs = 2 * np.random.random_sample((1,) + self.obs_sp_shape) - 1
        self.__obs = np.copy(obs)
        return obs

    def set_pos(self, pos):
        ''' Sets and returns a new observed state for the environment '''
        self.__obs[0, 0:self.__pos_size] = pos
        return np.copy(self.__obs)

    def get_pos(self):
        ''' Returns the current position of the agent in the environment '''
        return self.__obs[0, 0:self.__pos_size]

    def act(self, act):
        ''' Simulates the agent's action in the environment, computes and returns the environment's next state, the obtained reward and termination condition status '''
        act = np.copy(act).reshape((1,)+self.act_sp_shape)
        obs = np.copy(self.__obs)
        dist1 = np.sqrt(np.sum(np.square(obs[:]), axis=1))[0]
        obs[:, 0] = obs[:, 0] + 0.1 * np.cos(act[0,0]*np.pi).reshape(-1)
        obs[:, 1] = obs[:, 1] + 0.1 * np.sin(act[0,0]*np.pi).reshape(-1)
        dist2 = np.sqrt(np.sum(np.square(obs[:]), axis=1))[0]
        if dist2 <= self.__end_cond:    reward, end = 10*dist1, True
        else:                           reward, end = 10*(dist1-dist2), False if dist2 <= 1.5 else True
        self.__obs = np.copy(obs)
        return obs, reward, end

    def best_act(self):
        ''' Computes and returns the best possible action the agent can take '''
        obs = np.copy(self.__obs)
        return np.array([[np.arctan2(obs[:,1],obs[:,0])/np.pi],])

    def max_ret(self, obs):
        ''' Computes and returns the maximum return for the state '''
        return 10*np.sqrt(np.sum(np.square(obs.reshape(1,-1)[:,:]), axis=1))[0]



if __name__ == '__main__':
    from SoftActorCritic import SoftActorCritic

    # Get the environment
    env = Environment()

    # Create the model
    model = SoftActorCritic("Test", env, (64, 32), (128, 64, 32), replay_buffer_size=10000, seed=123)
#    model = SoftActorCritic.load("Test", env)

    # Set training hyper-parameters
    model.discount_factor = 0.95
    model.update_factor = 0.005
    model.replay_batch_size = 1000
    model.entropy = env.act_sp_shape[0]
    model.initial_alpha = 0.1
    model.H_adam_alpha = 0.001
    model.P_adam_alpha, model.P_train_frequency = 0.02, 1
    model.Q_adam_alpha, model.Q_train_frequency = 0.001, 1
    model.plot_resolution = 30

    # Start training
    model.train(episodes=3000, ep_steps=100, save_period=1000, plot_period=50)

    model.test(ep_steps=100)



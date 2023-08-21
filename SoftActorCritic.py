import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

data_type = np.float64  # Data type used in internal arrays

class SoftActorCritic:
    def __init__(self, name, environment, P_shape, Q_shape, replay_buffer_size, seed=None):
        '''
        Creates a Soft Actor-Critic learning context object
        :param name: The string that will be used to name the root folder and save files of the model
        :param environment: A reference to an environment object with obs_sp_shape, act_sp_shape, dest_pos and pos_idx attributes and reset, act, set_pos and get_pos methods. E.g.: "EnvironmentTest.py"
        :param P_shape: An array with the number of neurons in the internal layers of the policy network preceding the mean and standard deviation layers
        :param Q_shape: An array with the number of neurons in the internal layers of the state-action value function network
        :param replay_buffer_size: The number of entries in the replay buffer
        :param seed: The initial seed. If None, a random seed is generated
        :return: The Soft Actor-Critic learning context object
        '''
        self.name = name

        # Initial random seed
        if seed is None: seed = np.random.randint(9999999, size=1)
        self.seed = int(seed)
        np.random.seed(self.seed)

        # Environment
        self.environment = environment

        # Default hyper-parameters
        self.discount_factor = 0.95
        self.update_factor = 0.005      #To update the target parameters
        self.replay_batch_size = 1000   
        self.initial_alpha = 0.01
        self.entropy = environment.act_sp_shape[0]
        self.P_train_frequency, self.Q_train_frequency = 1, 1
        self.P_adam_alpha, self.P_adam_beta1, self.P_adam_beta2 = 0.005, 0.9, 0.999
        self.Q_adam_alpha, self.Q_adam_beta1, self.Q_adam_beta2 = 0.001, 0.9, 0.999
        self.H_adam_alpha, self.H_adam_beta1, self.H_adam_beta2 = 0.001, 0.9, 0.999
        self.P_reg_L1, self.P_reg_L2 = 0.0, 0.0
        self.Q_reg_L1, self.Q_reg_L2 = 0.0, 0.0
        self.plot_resolution = 30
        self.__alpha = 0.01
        self.__episode = 0
        self.__ep_ret = None
        self.__ep_loss = None

        # Policy estimators
        u, s = 0.0, 0.001
        self.__P_shape = P_shape
        self.__P = [np.random.normal(u, s, size=(environment.obs_sp_shape[-1]+1, P_shape[0])).astype(data_type), ]  #We always add 1 to the first layer of the weight matrix to consider the bias
        self.__P = self.__P + [np.random.normal(u, s, size=(P_shape[i]+1, P_shape[i+1])).astype(data_type) for i in range(len(P_shape)-1)]
        self.__P = self.__P + [np.random.normal(u, s, size=(P_shape[-1]+1, environment.act_sp_shape[-1])).astype(data_type), ]  #One for the mean
        self.__P = self.__P + [np.random.normal(u, s, size=(P_shape[-1]+1, environment.act_sp_shape[-1])).astype(data_type), ]  #Another for the standard deviation
        for w in self.__P:
            w[-1:, :] = np.zeros(w[-1:, :].shape, dtype=data_type)     #Bias in zero

        # Action-Value function estimator
        self.__Q_shape = Q_shape
        self.__Q = [np.random.normal(u, s, size=(2, environment.obs_sp_shape[-1]+environment.act_sp_shape[-1]+1, Q_shape[0])).astype(data_type), ]
        self.__Q = self.__Q + [np.random.normal(u, s, size=(2, Q_shape[i]+1, Q_shape[i+1])).astype(data_type) for i in range(len(Q_shape)-1)]
        self.__Q = self.__Q + [np.random.normal(u, s, size=(2, Q_shape[-1]+1, 1)).astype(data_type), ]
        for w in self.__Q:
            w[:, -1:, :] = np.zeros(w[:, -1:, :].shape, dtype=data_type)

        # Target networks
        self.__PT = [np.copy(weights) for weights in self.__P]
        self.__QT = [np.copy(weights) for weights in self.__Q]

        # Adaptive moment gradient descent variables
        self.__P_adam_aux = [[1.0, 1.0, np.zeros(w.shape, dtype=data_type), np.zeros(w.shape, dtype=data_type)] for w in self.__P]
        self.__Q_adam_aux = [[1.0, 1.0, np.zeros(w.shape, dtype=data_type), np.zeros(w.shape, dtype=data_type)] for w in self.__Q]
        self.__H_adam_aux = [1.0, 1.0, np.zeros((1,), dtype=data_type), np.zeros((1,), dtype=data_type)]

        # Replay buffer
        self.__rb_max_size = replay_buffer_size                                         # Replay buffer maximum size
        self.__rb_entries = 0                                                           # Replay buffer occupied entries
        self.__rb_obs = np.zeros((replay_buffer_size,) + environment.obs_sp_shape, dtype=data_type)      # Observed state buffer
        self.__rb_act = np.zeros((replay_buffer_size,) + environment.act_sp_shape, dtype=data_type)      # Action buffer
        self.__rb_nobs = np.zeros((replay_buffer_size,) + environment.obs_sp_shape, dtype=data_type)     # Next observed state buffer
        self.__rb_rwd = np.zeros((replay_buffer_size, 1), dtype=data_type)                               # Reward buffer
        self.__rb_end = np.zeros((replay_buffer_size, 1), dtype=data_type)                               # End buffer

        # Plotting
        self.__fig = plt.figure(figsize=(16,9))
        self.__fig.canvas.set_window_title(name)
        self.__colormap = mpl.colors.ListedColormap(mpl.cm.winter(np.linspace(0, 1, 1000))[0:1000, :-1]*0.9)
        self.__axis_1D, self.__mx, self.__my, self.__axis_2D = None, None, None, None
        self.__axs = [
            self.__fig.add_subplot(40, 100, ( 604, 1724)),                  # Trajectory
            self.__fig.add_subplot(40, 100, ( 628, 1748), projection='3d'), # Policy
            self.__fig.add_subplot(40, 100, ( 652, 1772), projection='3d'), # State-action value function
            self.__fig.add_subplot(40, 100, ( 676, 1796), projection='3d'), # State value function
            self.__fig.add_subplot(40, 100, (2304, 3424)),                  # Policy loss
            self.__fig.add_subplot(40, 100, (2328, 3448)),                  # State-action value function loss
            self.__fig.add_subplot(40, 100, (2352, 3472)),                  # Expected return vs Obtained return
            self.__fig.add_subplot(40, 100, (2376, 3496)),                  # Return Root Mean Squared Error
        ]
        self.__plot_num = 0
        self.__start_pos = None

        # Create the required directories if necessary
        if not os.path.isdir("./{0:s}".format(name)):
            if os.path.isfile("./{0:s}".format(name)):
                input("File './{0:s}' needs to be deleted. Press enter to continue.".format(name))
                os.remove("./{0:s}".format(name))
            os.mkdir("./{0:s}".format(name))
            os.chdir("./{0:s}".format(name))
            os.mkdir("./Train")
            with open('./Train/Progress.txt', 'w') as file: np.savetxt(file, np.array((0, )), fmt='%d')
        else:
            os.chdir("./{0:s}".format(name))

    def __sample_replay_buffer(self, replay_size):
        '''
        Randomly selects and returns replay_size entries from the replay buffer
        :param replay_size: Number of entries to return (if greater than the current number of entries, all entries are returned)
        :return: Numpy arrays for the observed state, action, next observed state, reward and termination condition
        '''
        # Obtain indexes for replay_batch_size samples and return the corresponding entries (without duplicates)
        idx = list(np.random.choice(range(0, self.__rb_entries), size=min(replay_size, self.__rb_entries), replace=False))
        return self.__rb_obs[idx], self.__rb_act[idx], self.__rb_nobs[idx], self.__rb_rwd[idx], self.__rb_end[idx]
        # print("Index = ",idx," - Entries = ",self.__rb_entries, " / ",self.__rb_max_size," - Data = ",
        # (self.__rb_obs[idx], self.__rb_act[idx], self.__rb_nobs[idx], self.__rb_rwd[idx], self.__rb_end[idx]))

    def __store_to_replay_buffer(self, obs, act, next_obs, reward, end):
        '''
        Stores a new entry in the replay buffer (If the buffer is full overwrites a random entry)
        :param obs: Numpy array representing the observed state
        :param act: Numpy array representing the action
        :param next_obs: Numpy array representing the next observed state
        :param reward: Float representing the reward
        :param end: Boolean representing the termination condition
        :return:
        '''
        # Compute the next index and update the corresponding entry
        idx = self.__rb_entries if self.__rb_entries < self.__rb_max_size else np.random.randint(0, self.__rb_max_size)
        self.__rb_entries = min(self.__rb_entries + 1, self.__rb_max_size)
        self.__rb_obs[idx] = obs
        self.__rb_act[idx] = act
        self.__rb_nobs[idx] = next_obs
        self.__rb_rwd[idx] = reward
        self.__rb_end[idx] = end
        #print("Index = ",idx," - Entries = ",self.__rb_entries, " / ",self.__rb_max_size," - Data = ",
        # (obs, act, next_obs, reward, end))

    def __opt_adaptive_moment_gradient_descent(self, alpha, beta1, beta2, w, dl_dw, adam_aux):
        '''
        Performs the adaptive moment gradient descent update
        :param alpha: Float representing the learning rate
        :param beta1: Float representing the first moment decay rate
        :param beta2: Float representing the second moment decay rate
        :param w: Numpy array representing the values to update
        :param dl_dw: Numpy array representing the difference to update (shape=w.shape)
        :param adam_aux: List composed of two Numpy arrays (shapes=w.shape) and two floats representing internal variables
        :return:
        '''
        adam_aux[0] *= beta1        # Power of the first moment decay to the number of optimization passes
        adam_aux[1] *= beta2        # Power of the second moment decay to the number of optimization passes

        np.add(beta1 * adam_aux[2], (1-beta1) * dl_dw, out=adam_aux[2])
        np.add(beta2 * adam_aux[3], (1-beta2) * np.square(dl_dw), out=adam_aux[3])
        np.subtract(w, (alpha/(1-adam_aux[0])) * np.divide(adam_aux[2], np.sqrt(adam_aux[3]/(1-adam_aux[1]))+1E-8), out=w)

    def policy(self, obs):
        '''
        Computes the agent's deterministic policy
        :param obs: Numpy array representing observed states
        :return: Numpy array representing the next action for each state
        '''
        x = obs
        # Compute the relu layers
        for w in self.__P[0:-2]:
            x = np.add(np.matmul(x, w[0:-1, :]), w[-1:, :])
            x = np.multiply(x, x > 0)
        # Compute the tanh layer (Flattened Gaussian mean)
        x = np.add(np.matmul(x, self.__P[-2][0:-1, :]), self.__P[-2][-1:, :])
        # Compute the tanh layer (Flattened Gaussian without sampling)
        return np.tanh(x)

    def __Hcompute(self, obs):
        '''
        Computes the policy's entropy
        :param obs: Numpy array representing M observed states
        :return: Numpy array representing the entropy for each state
        '''
        a = obs
        # Compute the relu layers
        for w in self.__P[0:-2]:
            z = np.add(np.matmul(a, w[0:-1, :]), w[-1:, :])
            a = np.multiply(z, z > 0)
        # Compute the tanh layer (Flattened Gaussian mean)
        z = np.add(np.matmul(a, self.__P[-2][0:-1, :]), self.__P[-2][-1:, :])
        u = np.tanh(z)
        # Compute the exponential layer (Flattened Gaussian standard deviation)
        z = np.add(np.matmul(a, self.__P[-1][0:-1, :]), self.__P[-1][-1:, :])
        s = np.clip(np.exp(z), 1E-9, 10)
        # Sample the Flattened Gaussian distribution
        z = np.random.normal(size=(300,)+s.shape).astype(data_type)
        a = u + np.multiply(s, z)
        return np.mean(0.5*np.square(z) + np.log(np.sqrt(2*np.pi)*s) - 2*(a+np.log(0.5+0.5*np.exp(-2*a))))

    def __Pcompute(self, obs):
        '''
        Computes the agent's stochastic policy
        :param obs: Numpy array representing M observed states
        :return: Numpy array representing the next action for each state
        '''
        a = obs.reshape((-1,)+self.environment.obs_sp_shape)
        # Compute the relu layers
        for w in self.__P[0:-2]:
            z = np.add(np.matmul(a, w[0:-1, :]), w[-1:, :])
            a = np.multiply(z, z > 0)
        # Compute the tanh layer (Flattened Gaussian mean)
        z = np.add(np.matmul(a, self.__P[-2][0:-1, :]), self.__P[-2][-1:, :])
        u = z
        # Compute the exponential layer (Flattened Gaussian standard deviation)
        z = np.add(np.matmul(a, self.__P[-1][0:-1, :]), self.__P[-1][-1:, :])
        s = np.clip(np.exp(z), 1E-9, 10)
        # Sample the Flattened Gaussian distribution
        z = np.random.normal(size=s.shape).astype(data_type)
        a = np.tanh(u + np.multiply(s, z))
        return a

    def __Qcompute(self, obs, act):
        '''
        Computes the state-action value function
        :param obs: Numpy array representing M observed states
        :param act: Numpy array representing M taken actions
        :return: Numpy array representing the value of each state-action pair
        '''
        obs = obs.reshape((-1,)+self.environment.obs_sp_shape)
        act = act.reshape((-1,)+self.environment.act_sp_shape)
        a = np.concatenate([obs, act], axis=-1)
        a = a.reshape((1,) + a.shape)   #DIFF(ADD)
        # Compute the relu layers
        for w in self.__Q[0:-1]:
            z = np.add(np.matmul(a, w[:, 0:-1, :]), w[:, -1:, :])
            a = np.multiply(z, z > 0)
        # Compute the linear layers
        a = np.add(np.matmul(a, self.__Q[-1][:, 0:-1, :]), self.__Q[-1][:, -1:, :])
        # Compute the minimum layer
        a = np.min(a, axis=0).reshape(-1)   #DIFF
        return a

    def __VTcompute(self, obs):
        '''
        Computes the state value function
        :param obs: Numpy array representing M observed states
        :return: Numpy array representing the value of each state
        '''
        a = obs.reshape((-1,)+self.environment.obs_sp_shape)
        ## Policy
        # Compute the relu layers
        for w in self.__PT[0:-2]:
            z = np.add(np.matmul(a, w[0:-1, :]), w[-1:, :])
            a = np.multiply(z, z > 0)
        # Compute the tanh layer (Flattened Gaussian mean)
        z = np.add(np.matmul(a, self.__PT[-2][0:-1, :]), self.__PT[-2][-1:, :])
        u = z
        # Compute the exponential layer (Flattened Gaussian standard deviation)
        z = np.add(np.matmul(a, self.__PT[-1][0:-1, :]), self.__PT[-1][-1:, :])
        s = np.clip(np.exp(z), 1E-9, 10)
        # Sample the Flattened Gaussian distribution
        z = np.random.normal(size=s.shape).astype(data_type)
        a = u + np.multiply(s, z)
        
        #Posible solution to exponential's overflow in h calculation
        if np.min(a) <= -350:   #If any of the elements is lower than -350, we go through each element (slower)
            h = np.zeros((a.shape[0],1))
            
            for row in range(0,a.shape[0]):
                for column in range(0,a.shape[1]):
                    if a[row,column] <= -350: #If a is lower than -350, we aproximate the log(0.5 + 0.5*exp(-2*a)) to -2*a - log(2)
                        print("aproximacion usada")
                        h[row] += 0.5*np.square(z[row,column]) + np.log(np.sqrt(2*np.pi)*s[row,column]) + 2*(a[row,column]+np.log(2))
                    else:
                        h[row] += 0.5*np.square(z[row,column]) + np.log(np.sqrt(2*np.pi)*s[row,column]) - 2*(a[row,column]+np.log(0.5+0.5*np.exp(-2*a[row,column])))
        else:
            h = np.sum(0.5*np.square(z) + np.log(np.sqrt(2*np.pi)*s) - 2*(a+np.log(0.5+0.5*np.exp(-2*a))), axis=-1, keepdims=True)
        a = np.tanh(a)

        ## State-action value function
        a = np.concatenate([obs, a], axis=-1)
        a = a.reshape((1,) + a.shape)   #DIFF(ADD)
        # Compute the relu layers
        for w in self.__QT[0:-1]:
            z = np.add(np.matmul(a, w[:, 0:-1, :]), w[:, -1:, :])
            a = np.multiply(z, z > 0)
        # Compute the linear layers
        a = np.add(np.matmul(a, self.__QT[-1][:, 0:-1, :]), self.__QT[-1][:, -1:, :])
        # Compute the minimum layer
        a = np.min(a, axis=0)

        return a + self.__alpha * h

    def __Ptrain(self, x_ref, epochs, n_batches=1):
        '''
        Updates the policy network's weights
        :param x_ref: Numpy array representing M observed states
        :param epochs: The number of times that the whole x_ref should be used to train
        :param n_batches: The number of batches in which x_ref should be separated
        :return: Numpy array representing the loss in each training epoch
        '''
        n_batches = x_ref.shape[0] if n_batches > x_ref.shape[0] else n_batches
        loss_evolution = np.zeros((epochs, n_batches), dtype=data_type)
        batch_idxs = np.floor(np.linspace(0, x_ref.shape[0], n_batches+1)).astype(int)
        opt_alpha, opt_beta1, opt_beta2 = self.P_adam_alpha, self.P_adam_beta1, self.P_adam_beta2
        reg_L1, reg_L2 = None, None # self.P_reg_L1, self.P_reg_L2

        # For each epoch
        batch_x = x_ref
        for epoch in range(epochs):
            # Shuffle the inputs and form batches when appropriate
#            epoch_x = x_ref[np.random.permutation(x_ref.shape[0])]

            # Train n batches
            for batch in range(n_batches):
                # Get the batch input and output
#                batch_x = epoch_x[batch_idxs[batch]:batch_idxs[batch+1]]
#                print("rand: ", np.random.normal(0.0, 1.0, 1))

                ## Forward pass
                a, k = batch_x.reshape((-1,)+self.environment.obs_sp_shape), []
                # Compute the relu layers
                for w in self.__P[0:-2]:
                    x = a
                    z = np.add(np.matmul(x, w[0:-1, :]), w[-1:, :])
                    a = np.multiply(z, z > 0)
                    k.append([x, z])
                # Compute the tanh layer (Flattened Gaussian mean)
                x, w = a, self.__P[-2]
                z = np.add(np.matmul(x, w[0:-1, :]), w[-1:, :])
                u = z
                k.append([x, u])
                # Compute the exponential layer (Flattened Gaussian standard deviation)
                x, w = a, self.__P[-1]
                z = np.add(np.matmul(x, w[0:-1, :]), w[-1:, :])
                s = np.clip(np.exp(z), 1E-9, 10)
                k.append([x, s])
                # Sample the Flattened Gaussian distribution
                z = np.random.normal(size=s.shape).astype(data_type)
                a = u + np.multiply(s, z)
                h = np.sum(0.5*np.square(z) + np.log(np.sqrt(2*np.pi)*s) - 2*(a+np.log(0.5+0.5*np.exp(-2*a))), axis=-1, keepdims=True)
                a = np.tanh(a)
                k.append([z, a])

                ## Compute the loss function and the derivative dL/dA
                # Compute the state-action value
                a = np.concatenate([batch_x, a], axis=-1)
                a = a.reshape((1,)+a.shape)
                for w in self.__Q[0:-1]:
                    z = np.add(np.matmul(a, w[:, 0:-1, :]), w[:, -1:, :])
                    a = np.multiply(z, z > 0)
                    k.append(z)
                x = np.add(np.matmul(a, self.__Q[-1][:, 0:-1, :]), self.__Q[-1][:, -1:, :])
                a = np.min(x, axis=0, keepdims=True)
                # Compute the loss function and the derivative wrt the last layer's output
                loss_evolution[epoch, batch] = np.mean(-(a + self.__alpha * h))
                dl_da = - np.ones(a.shape)
                # Obtain the state-action value function derivative wrt the input action
                dl_da = np.multiply(np.array(x == a), dl_da)
                dl_da = np.matmul(dl_da, np.transpose(self.__Q[-1][:, 0:-1, :], (0, 2, 1)))
                for w in reversed(self.__Q[0:-1]):
                    dl_da = np.matmul(np.multiply(dl_da, k.pop() > 0), np.transpose(w[:, 0:-1, :], (0, 2, 1)))
                dl_da = np.sum(dl_da[..., x_ref.shape[-1]:], axis=0)

                ## Backward pass
                # Backpropagate the Flattened Gaussian distribution
                z, a = k.pop()
                dl_du = np.multiply(1 - np.square(a), dl_da)
                dl_ds = np.multiply(np.multiply(1 - np.square(a), z), dl_da)
                # Backpropagate the entropy term
                da_du = -2 + np.divide(4, 1 + np.exp(2 * (u + np.multiply(s, z))))
                dl_dz = -self.__alpha * np.ones((batch_x.shape[0], self.environment.act_sp_shape[0]))
                dl_du += np.multiply(dl_dz, da_du)
                dl_ds += np.multiply(dl_dz, np.divide(1, s) + np.multiply(z, da_du))
                # Backpropagate the exponential layer (Flattened Gaussian standard deviation)
                x, s = k.pop()
                w = self.__P[-1]
                dl_dz = np.multiply(dl_ds, s)
                dl_dw = np.concatenate([np.divide(np.matmul(x.T, dl_dz), batch_x.shape[0]), np.mean(dl_dz, axis=0, keepdims=True)], axis=0)
                dl_da = np.matmul(dl_dz, w[0:-1,:].T)
                if reg_L1: np.add(dl_dw, (reg_L1/batch_x.shape[0]) * np.sign(dl_dw), out=dl_dw)
                if reg_L2: np.add(dl_dw, (reg_L2/batch_x.shape[0]) * w[0:-1, :], out=dl_dw)
                self.__opt_adaptive_moment_gradient_descent(opt_alpha, opt_beta1, opt_beta2, self.__P[-1], dl_dw, self.__P_adam_aux[-1])
                # Backpropagate the tanh layer (Flattened Gaussian mean)
                x, u = k.pop()
                w = self.__P[-2]
                dl_dz = dl_du
                dl_dw = np.concatenate([np.divide(np.matmul(x.T, dl_dz), batch_x.shape[0]), np.mean(dl_dz, axis=0, keepdims=True)], axis=0)
                dl_da += np.matmul(dl_dz, w[0:-1, :].T)
                if reg_L1: np.add(dl_dw, (reg_L1/batch_x.shape[0]) * np.sign(dl_dw), out=dl_dw)
                if reg_L2: np.add(dl_dw, (reg_L2/batch_x.shape[0]) * w[0:-1, :], out=dl_dw)
                self.__opt_adaptive_moment_gradient_descent(opt_alpha, opt_beta1, opt_beta2, self.__P[-2], dl_dw, self.__P_adam_aux[-2])
                # Backpropagate the relu layers
                for w, opt_aux in zip(reversed(self.__P[0:-2]), reversed(self.__P_adam_aux[0:-2])):
                    x, z = k.pop()
                    dl_dz = np.multiply(dl_da, z > 0)
                    dl_dw = np.concatenate([np.divide(np.matmul(x.T, dl_dz), batch_x.shape[0]), np.mean(dl_dz, axis=0, keepdims=True)], axis=0)
                    dl_da = np.matmul(dl_dz, w[0:-1, :].T)
                    if reg_L1: np.add(dl_dw, (reg_L1/batch_x.shape[0]) * np.sign(dl_dw), out=dl_dw)
                    if reg_L2: np.add(dl_dw, (reg_L2/batch_x.shape[0]) * w[0:-1, :], out=dl_dw)
                    self.__opt_adaptive_moment_gradient_descent(opt_alpha, opt_beta1, opt_beta2, w, dl_dw, opt_aux)

        # Update alpha
        self.__opt_adaptive_moment_gradient_descent(self.H_adam_alpha, self.H_adam_beta1, self.H_adam_beta2, self.__alpha, np.mean(h) + self.entropy, self.__H_adam_aux)
        return loss_evolution

    def __Qtrain(self, x_ref, y_ref, epochs, n_batches=1):
        '''
        Updates the state-action value function network's weights
        :param x_ref: Numpy array representing M observed states and actions
        :param y_ref: Numpy array representing the expected state-action value for each state-action pair
        :param epochs: The number of times that the whole x_ref and y_ref should be used to train
        :param n_batches: The number of batches in which x_ref and y_ref should be separated
        :return: Numpy array representing the loss in each training epoch
        '''
        x_ref = np.concatenate(x_ref, axis=-1).reshape((-1, self.environment.obs_sp_shape[0]+self.environment.act_sp_shape[0]))
        n_batches = x_ref.shape[0] if n_batches > x_ref.shape[0] else n_batches
        loss_evolution = np.zeros((epochs, n_batches), dtype=data_type)
        batch_idxs = np.floor(np.linspace(0, x_ref.shape[0], n_batches+1)).astype(int)
        opt_alpha, opt_beta1, opt_beta2 = self.Q_adam_alpha, self.Q_adam_beta1, self.Q_adam_beta2
        reg_L1, reg_L2 = None, None # self.Q_reg_L1, self.Q_reg_L2

        # For each epoch
        batch_x = x_ref
        batch_y = y_ref
        for epoch in range(epochs):
            # Shuffle the inputs and form batches when appropriate
#            idx_shuffle = np.random.permutation(x_ref.shape[0])
#            epoch_x, epoch_y = x_ref[idx_shuffle], y_ref[idx_shuffle]

            # Train n batches
            for batch in range(n_batches):
                # Get the batch input and output
#                batch_x = epoch_x[batch_idxs[batch]:batch_idxs[batch+1]]
#                batch_y = epoch_y[batch_idxs[batch]:batch_idxs[batch+1]]
#                print("rand: ", np.random.normal(0.0, 1.0, 1))

                ## Forward pass
                a, k = batch_x.reshape((1,)+batch_x.shape), []
                # Compute the relu layers
                for w in self.__Q[0:-1]:
                    x = a
                    z = np.add(np.matmul(x, w[:, 0:-1, :]), w[:, -1:, :])
                    a = np.multiply(z, z > 0)
                    k.append([x, z])
#                    print("F - Q dense: a:", a)
                # Compute the linear layers
                x, w = a, self.__Q[-1]
                a = np.add(np.matmul(x, w[:, 0:-1, :]), w[:, -1:, :])
#                print("F - Q linear: a:", a)
                k.append(x)
                # Compute the minimum layer
                x = a
                a = np.min(x, axis=0, keepdims=True)
#                print("F - Q min: a:", a)

                ## Compute the loss function and the derivative dL/dA
                loss_evolution[epoch, batch] = np.mean(np.square(batch_y - a))
                dl_da = 2*(a - batch_y)

                ## Backward pass
                # Backpropagate the min layer
                dl_da = np.multiply(np.array(x == a), dl_da)
#                print("Q min: dl/da:", dl_da)
                # Backpropagate the linear layers
                x, w = k.pop(), self.__Q[-1]
                dl_dz = dl_da
                dl_dw = np.concatenate([np.divide(np.matmul(np.transpose(x, (0, 2, 1)), dl_dz), batch_x.shape[0]), np.mean(dl_dz, axis=1, keepdims=True)], axis=1)
                dl_da = np.matmul(dl_dz, np.transpose(w[:, 0:-1, :], (0, 2, 1)))
                if reg_L1: np.add(dl_dw, (reg_L1 / batch_x.shape[0]) * np.sign(dl_dw), out=dl_dw)
                if reg_L2: np.add(dl_dw, (reg_L2 / batch_x.shape[0]) * w[:, 0:-1, :], out=dl_dw)
#                print("Q linear: w:", w,"\ndl/dw:", dl_dw)
                self.__opt_adaptive_moment_gradient_descent(opt_alpha, opt_beta1, opt_beta2, w, dl_dw, self.__Q_adam_aux[-1])
                # Backpropagate the relu layers
                for w, opt_aux in zip(reversed(self.__Q[0:-1]), reversed(self.__Q_adam_aux[0:-1])):
                    x, z = k.pop()
                    dl_dz = np.multiply(dl_da, z > 0)
                    dl_dw = np.concatenate([np.divide(np.matmul(np.transpose(x, (0, 2, 1)), dl_dz), batch_x.shape[0]), np.mean(dl_dz, axis=1, keepdims=True)], axis=1)
                    dl_da = np.matmul(dl_dz, np.transpose(w[:, 0:-1, :], (0, 2, 1)))
                    if reg_L1: np.add(dl_dw, (reg_L1/batch_x.shape[0]) * np.sign(dl_dw), out=dl_dw)
                    if reg_L2: np.add(dl_dw, (reg_L2/batch_x.shape[0]) * w[:, 0:-1, :], out=dl_dw)
#                    print("Q dense w:", w,"\ndl/dw:", dl_dw)
                    self.__opt_adaptive_moment_gradient_descent(opt_alpha, opt_beta1, opt_beta2, w, dl_dw, opt_aux)

        return loss_evolution

    def train(self, episodes, ep_steps, save_period=0, plot_period=0, tplot_period=5):
        '''
        Trains the model with the specified parameters
        :param episodes: The number of episodes to run
        :param ep_steps: The maximum number of steps in each episode
        :param save_period: Number of episodes between two save operations of the algorithm's context
        :param plot_period: Number of episodes between two full plot operations
        :param tplot_period: Number of episodes between two trajectory plot operations
        :return:
        '''
        # Adjust the function's arguments
        if self.__episode > episodes:
            print("The loaded episode is greater than the requested number of episodes")
            return
        episode = self.__episode
        save_period = save_period if save_period else int(np.ceil(episodes/10))
        plot_period = plot_period if plot_period else episodes+1

        # Retrieve the algorithm's environment and hyper-parameters
        env = self.environment
        discount_factor, update_factor = self.discount_factor, self.update_factor
        replay_batch_size = self.replay_batch_size
        P_freq, Q_freq = self.P_train_frequency, self.Q_train_frequency

        # Initialize algorithm variables
        ep_obs = np.zeros((ep_steps+1,) + env.obs_sp_shape, dtype=data_type)     # Episode's observed states
        ep_act = np.zeros((ep_steps,) + env.act_sp_shape, dtype=data_type)       # Episode's actions
        ep_rwd = np.zeros((ep_steps, 1), dtype=data_type)                        # Episode's rewards
        ep_ret = np.zeros((episodes, 3), dtype=data_type)                        # Returns for each episode (real, expected and RMSE)
        ep_loss = np.zeros((episodes, 2), dtype=data_type)                       # Training loss for each episode (Q and P)

        # Retrieve data from the previous execution if necessary
        if self.__episode == 0:
            self.__alpha = np.array(self.initial_alpha).reshape((1,))
            self.__ep_ret, self.__ep_loss = ep_ret, ep_loss
            with open('./Train/Branch.txt', 'a') as file:
                np.savetxt(file, ("Started training with seed {0:d}.".format(self.seed),), fmt='%s')
            self.__save(0)
        else:
            ep_ret[0:episode], ep_loss[0:episode] = self.__ep_ret, self.__ep_loss
            self.__ep_ret, self.__ep_loss = ep_ret, ep_loss

        # Initialize plot related variables
        self.__axis_1D = np.linspace(-1, 1, num=self.plot_resolution)
        self.__mx, self.__my = np.meshgrid(self.__axis_1D, self.__axis_1D)
        self.__axis_2D = np.concatenate((np.array(self.__mx).reshape(-1,1), np.array(self.__my).reshape(-1,1)), axis=1)

        # For all episodes
        self.__plot(episode, ep_obs[0:2, env.pos_idx])
#        print("Q:", self.__Q)
#        print("P:", self.__P)
        while episode < episodes:
#            print("seed: ", self.seed)
            # Initialization
            ep_obs[0], s, end = env.reset(), 0, False

            # Do N steps following the policy estimator
            for s in range(ep_steps):
                # Decide the next action
                ep_act[s] = self.__Pcompute(ep_obs[s])
                # Act in the environment
                ep_obs[s+1], ep_rwd[s], end = env.act(ep_act[s])
                # Store in replay buffer
                self.__store_to_replay_buffer(ep_obs[s], ep_act[s], ep_obs[s+1], ep_rwd[s], end)
                # End on terminal state
                if end: break
            ep_len = s + 1

            # Compute the real and expected returns and the root mean square error
            if not end: ep_rwd[s] += discount_factor * self.__Qcompute(ep_obs[s+1], self.__Pcompute(ep_obs[s+1]))
            for i in range(ep_len-2, -1, -1): ep_rwd[i] = ep_rwd[i] + discount_factor * ep_rwd[i + 1]
            ep_ret[episode, 0] = ep_rwd[0]
            ep_ret[episode, 1] = self.__Qcompute(ep_obs[0], ep_act[0])
            ep_ret[episode, 2] = np.sqrt(np.square(ep_ret[episode, 0] - ep_ret[episode, 1]))
#            ep_ret[episode, :] = ep_ret[episode, :] / env.max_ret(ep_obs[0])

            for i in range(ep_len):
                # Sample the replay buffer
                if (i % Q_freq == 0) or (i % P_freq == 0):

                    tr_obs, tr_act, tr_next_obs, tr_reward, tr_end = self.__sample_replay_buffer(replay_batch_size)
                    tr_reward += (1 - tr_end) * discount_factor * self.__VTcompute(tr_next_obs)

                    if i % Q_freq == 0:
                        # Train the state-action value function estimator
                        ep_loss[episode, 0] = np.mean(self.__Qtrain([tr_obs, tr_act], tr_reward, 1, 1))
    #                   print("Q:", self.__Q)

                    if i % P_freq == 0:
                        # Train policy estimator
                        ep_loss[episode, 1] = np.mean(self.__Ptrain(tr_obs, 1, 1))
    #                   print("P:", self.__P)

                        # Update target model's weights
                        for w, w_target in zip(self.__P, self.__PT): np.add((1-update_factor) * w_target, update_factor * w, out=w_target)
                        for w, w_target in zip(self.__Q, self.__QT): np.add((1-update_factor) * w_target, update_factor * w, out=w_target)
#version vieja comentada por las dudas
#             for i in range(int(np.ceil(ep_len/Q_freq))):
#                 # Sample the replay buffer
#                 tr_obs, tr_act, tr_next_obs, tr_reward, tr_end = self.__sample_replay_buffer(replay_batch_size)
#                 tr_reward += (1 - tr_end) * discount_factor * self.__VTcompute(tr_next_obs)

#                 # Train the state-action value function estimator
#                 ep_loss[episode, 0] = np.mean(self.__Qtrain([tr_obs, tr_act], tr_reward, 1, 1))
# #                print("Q:", self.__Q)

#                 if i % P_freq == 0:
#                     # Train policy estimator
#                     ep_loss[episode, 1] = np.mean(self.__Ptrain(tr_obs, 1, 1))
# #                    print("P:", self.__P)

#                     # Update target model's weights
#                     for w, w_target in zip(self.__P, self.__PT): np.add((1-update_factor) * w_target, update_factor * w, out=w_target)
#                     for w, w_target in zip(self.__Q, self.__QT): np.add((1-update_factor) * w_target, update_factor * w, out=w_target)

            # Increase the episode number
            episode += 1

            # Save the algorithm's context and update the tracker file
            if (episode % save_period) == 0: self.__save(episode)
            if (episode % plot_period) == 0: self.__plot(episode, ep_obs[:ep_len+1, env.pos_idx])
            elif (episode % tplot_period) == 0: self.__plot(episode, ep_obs[:ep_len+1, env.pos_idx], trajectory_only=True)
            print("Progress: ", episode, "/", episodes, "Alpha: ", self.__alpha, "Loss (Q,P):", ep_loss[episode-1, 0:2])

        # Update the episode number
        self.__episode = episode

    def __save(self, episode):
        '''
        Saves the model's variables to a set of files
        :param episode: The episode number to use in the filename
        :return:
        '''
        # Get the file name
        filename = 'Train/episode_{0:07d}'.format(episode)
        self.__episode = episode

        # Save the algorithm's context
        with open('{0:s}.rlcd'.format(filename), 'wb') as file:
            # Store class information
            np.save(file, self.__P_shape)
            np.save(file, self.__Q_shape)
            np.save(file, self.__rb_max_size)
            np.save(file, self.seed)

            # Store the hyper-parameters
            np.save(file, self.discount_factor)
            np.save(file, self.update_factor)
            np.save(file, self.replay_batch_size)
            np.save(file, self.initial_alpha)
            np.save(file, self.entropy)
            np.save(file, self.P_train_frequency)
            np.save(file, self.Q_train_frequency)
            np.save(file, self.P_adam_alpha)
            np.save(file, self.P_adam_beta1)
            np.save(file, self.P_adam_beta2)
            np.save(file, self.Q_adam_alpha)
            np.save(file, self.Q_adam_beta1)
            np.save(file, self.Q_adam_beta2)
            np.save(file, self.H_adam_alpha)
            np.save(file, self.H_adam_beta1)
            np.save(file, self.H_adam_beta2)
            np.save(file, self.P_reg_L1)
            np.save(file, self.P_reg_L2)
            np.save(file, self.Q_reg_L1)
            np.save(file, self.Q_reg_L2)

            # Store the execution data
            np.save(file, self.__episode)
            np.save(file, self.__alpha)
            for w in self.__P:  np.save(file, w)
            for w in self.__Q:  np.save(file, w)
            for w in self.__PT:  np.save(file, w)
            for w in self.__QT:  np.save(file, w)
            for w in self.__P_adam_aux:
                for e in w: np.save(file, e)
            for w in self.__Q_adam_aux:
                for e in w: np.save(file, e)
            for e in self.__H_adam_aux: np.save(file, e)
            np.save(file, self.__ep_ret[0:episode])
            np.save(file, self.__ep_loss[0:episode])
            np.save(file, self.__rb_entries)
            np.save(file, self.__rb_obs[0:self.__rb_entries])
            np.save(file, self.__rb_act[0:self.__rb_entries])
            np.save(file, self.__rb_nobs[0:self.__rb_entries])
            np.save(file, self.__rb_rwd[0:self.__rb_entries])
            np.save(file, self.__rb_end[0:self.__rb_entries])

            # Generate, set and store a random seed
#            print("rand save: ", np.random.normal(0.0, 1.0, 1))
            seed = int(np.random.randint(9999999, size=1))
            np.save(file, seed)
            np.random.seed(seed)

        # Update the progress file
        with open('./Train/Progress.txt', 'w') as file: np.savetxt(file, np.array((episode,)), fmt='%d')

    @classmethod
    def load(cls, name, environment, filename="", seed=None):
        '''
        Restores a Soft Actor-Critic learning context object from a save file
        :param name: The name of the root folder in which the model is saved
        :param environment: A reference to an environment object with obs_sp_shape, act_sp_shape, dest_pos and pos_idx attributes and reset, act, set_pos and get_pos methods. E.g.: "EnvironmentTest.py"
        :param filename: The name of the save file to be loaded. If None or empty, training is restarted from the last save.
        :param seed: The initial seed value. If None, then the last saved seed is used
        :return: The Soft Actor-Critic learning context object
        '''
        # If no filename was specified load the last training configuration
        if (filename is None) or (filename == ""):
            # Access the requested directory
            if not os.path.isdir('./{0:s}'.format(name)):
                print('Project {0:s} could not be found'.format(name))
                return None
            # Determine the appropriate load file from the progress file
            if not os.path.isfile('./{0:s}/Train/Progress.txt'.format(name)):
                print('Progress file could not be found'.format(filename))
                return None
            with open('./{0:s}/Train/Progress.txt'.format(name), 'r') as file: episode = int(np.loadtxt(file))
            filename = '{0:s}/Train/episode_{1:07d}.rlcd'.format(name, episode)

        # Verify the existence of the requested file
        if not os.path.isfile('./{0:s}'.format(filename)):
            file = filename
            filename = '{0:s}/Train/{1:s}'.format(name, filename)
            if not os.path.isfile('./{0:s}'.format(filename)):
                print('File {0:s} could not be found'.format(file))
                return None

        # Load the reinforcement learning algorithm's context data
        with open('./{0:s}'.format(filename), 'rb') as file:
            # Run the default constructor
            Pshape = np.load(file)
            Qshape = np.load(file)
            replay_buffer_size = np.load(file)
            if seed is None:
                seed = np.load(file)
            else:
                _ = np.load(file)   # Discard the stored seed
                if os.path.isfile('./{0:s}/Train/Progress.txt'.format(name)):
                    episode = int(np.loadtxt('./{0:s}/Train/Progress.txt'.format(name)))
                    with open('./{0:s}/Train/Branch.txt'.format(name), 'a') as file2:
                        np.savetxt(file2, ("Trained up to episode {0:d}.".format(episode),), fmt='%s')
                with open('./{0:s}/Train/Branch.txt'.format(name), 'a') as file2:
                    episode = int(filename[-12:-5])
                    np.savetxt(file2, ("Branched on episode {0:d} with seed {1:d}.".format(episode, seed),), fmt='%s')
            learning_process = cls(name, environment, Pshape, Qshape, replay_buffer_size, seed)

            # Load the hyper-parameters
            learning_process.discount_factor = np.load(file)
            learning_process.update_factor = np.load(file)
            learning_process.replay_batch_size = np.load(file)
            learning_process.initial_alpha = np.load(file)
            learning_process.entropy = np.load(file)
            learning_process.P_train_frequency = np.load(file)
            learning_process.Q_train_frequency = np.load(file)
            learning_process.P_adam_alpha = np.load(file)
            learning_process.P_adam_beta1 = np.load(file)
            learning_process.P_adam_beta2 = np.load(file)
            learning_process.Q_adam_alpha = np.load(file)
            learning_process.Q_adam_beta1 = np.load(file)
            learning_process.Q_adam_beta2 = np.load(file)
            learning_process.H_adam_alpha = np.load(file)
            learning_process.H_adam_beta1 = np.load(file)
            learning_process.H_adam_beta2 = np.load(file)
            learning_process.P_reg_L1 = np.load(file)
            learning_process.P_reg_L2 = np.load(file)
            learning_process.Q_reg_L1 = np.load(file)
            learning_process.Q_reg_L2 = np.load(file)

            # Recover the execution context
            learning_process.__episode = np.load(file)
            learning_process.__alpha = np.load(file)
            for i in range(len(learning_process.__P)): learning_process.__P[i] = np.load(file)
            for i in range(len(learning_process.__Q)): learning_process.__Q[i] = np.load(file)
            for i in range(len(learning_process.__PT)): learning_process.__PT[i] = np.load(file)
            for i in range(len(learning_process.__QT)): learning_process.__QT[i] = np.load(file)
            for i in range(len(learning_process.__P_adam_aux)):
                for j in range(4): learning_process.__P_adam_aux[i][j] = np.load(file)
            for i in range(len(learning_process.__Q_adam_aux)):
                for j in range(4): learning_process.__Q_adam_aux[i][j] = np.load(file)
            for i in range(4): learning_process.__H_adam_aux[i] = np.load(file)
            learning_process.__ep_ret = np.load(file)
            learning_process.__ep_loss = np.load(file)
            learning_process.__rb_entries = np.load(file)
            learning_process.__rb_obs[0:learning_process.__rb_entries] = np.load(file)
            learning_process.__rb_act[0:learning_process.__rb_entries] = np.load(file)
            learning_process.__rb_nobs[0:learning_process.__rb_entries] = np.load(file)
            learning_process.__rb_rwd[0:learning_process.__rb_entries] = np.load(file)
            learning_process.__rb_end[0:learning_process.__rb_entries] = np.load(file)

            # Recover the random seed
            np.random.seed(np.load(file))

        return learning_process

    def _trajectory_click_CB(self, event):
        ''' Callback for click events in the trajectory plot '''
        T_ax = self.__axs[0]
        if event.inaxes == T_ax:
            rel_pos = T_ax.transAxes.inverted().transform((event.x, event.y))
            xlim = T_ax.get_xlim()
            ylim = T_ax.get_ylim()
            self.__start_pos = np.array([rel_pos[0]*(xlim[1]-xlim[0])+xlim[0], rel_pos[1]*(ylim[1]-ylim[0])+ylim[0]])

    def test(self, ep_steps):
        '''
        Tests the model using the graphic user interface, starting trajectories in the mouse click's position
        :param ep_steps: Maximum number of steps allowed in each trajectory
        :return:
        '''
        # Initialize algorithm variables
        env = self.environment
        ep_obs = np.zeros((ep_steps+1,) + env.obs_sp_shape)   # Episode's observed states

        # Initialize plot related variables
        self.__axis_1D = np.linspace(-1, 1, num=self.plot_resolution)
        self.__mx, self.__my = np.meshgrid(self.__axis_1D, self.__axis_1D)
        self.__axis_2D = np.concatenate((np.array(self.__mx).reshape(-1,1), np.array(self.__my).reshape(-1,1)), axis=1)

        # Set the initial position click callback
        self.__fig.canvas.mpl_connect("button_press_event", self._trajectory_click_CB)
        self.__plot(self.__episode, ep_obs[0:2, env.pos_idx])

        while True:
            # Obtain the starting position
            while self.__start_pos is None: plt.pause(0.001)
            pos = self.__start_pos.reshape((1, 2))
            self.__start_pos = None
            print("Selected position: ", pos)
            ep_obs[0], s = env.set_pos(pos), 0

            # Act
            for s in range(ep_steps):
                act = self.policy(ep_obs[s].reshape((1,) + env.obs_sp_shape))   # Decide the next action
                ep_obs[s+1], _, end = env.act(act)                              # Act in the environment
                if end: break                                                   # End the episode on terminal state

            # Plot
            self.__plot(self.__episode, ep_obs[:s+2, env.pos_idx], trajectory_only=True)

    def __plot(self, episode, trajectory, trajectory_only=False):
        '''
        Plots the last trajectory, the policy, the state-action value function, the target state value function, the training losses and the difference between expected and obtained return
        :param episode: The episode number
        :param trajectory: Numpy array containing the agent's positions in the trajectory
        :param trajectory_only: Boolean representing if all plots should be updated or only the trajectory plot
        :return:
        '''
        percentiles = [0, 5, 10, 25, 75, 90, 95, 100]
        T_ax, P_ax, Q_ax, V_ax, Pl_ax, Ql_ax, R_ax, Re_ax = self.__axs
        resolution = self.plot_resolution
        env = self.environment
        self.__plot_num = self.__plot_num + 1 if self.__plot_num < env.act_sp_shape[0]-1 else 0

        # Reset plot
        self.__fig.suptitle("Episode {0:d}/{1:d}".format(episode, self.__ep_ret.shape[0]))
        if trajectory_only:
            T_ax.clear()
        else:
            for ax in self.__axs:
                ax.clear()
                P_ax.set_xlabel('Xo')
                P_ax.set_ylabel('Yo')

        # Plot trajectory
        T_ax.set_title('Last Trajectory')
        T_ax.set_xlim(-1.5, 1.5)
        T_ax.set_ylim(-1.5, 1.5)
        T_ax.set_xticks(np.linspace(-1.5, 1.5, num=7, endpoint=True))
        T_ax.set_yticks(np.linspace(-1.5, 1.5, num=7, endpoint=True))
        T_ax.add_patch(plt.Circle((0, 0), 0.1, color='orange', fill=False))
        T_ax.add_patch(plt.Circle((0, 0), 1.5, color='orange', fill=False))
        if (len(env.dest_pos) <= 1) or (len(trajectory) <= 1): return;
        T_ax.scatter(env.dest_pos[0], env.dest_pos[1], color='blue', alpha=0.8, linewidth=2, linestyle='solid')
        T_ax.scatter(trajectory[0, 0], trajectory[0, 1], color='black', alpha=0.8, linewidth=2, linestyle='solid')
        T_ax.plot(trajectory[:, 0], trajectory[:, 1], color='black', alpha=0.8, linewidth=2, linestyle='solid')

        if trajectory_only: plt.draw(); plt.pause(0.001); return

        # Plot policy
        obs = np.zeros((len(self.__axis_2D),)+env.obs_sp_shape)
        obs[:, 0:2] = self.__axis_2D[:, 0:2]
        P_ax.invert_yaxis()
        P_ax.set_title('Action[{0:d}]'.format(self.__plot_num))
        P_ax.plot_surface(self.__mx, self.__my, self.__Pcompute(obs).reshape((resolution, resolution, -1))[:, :, self.__plot_num], cmap=self.__colormap)

        # Plot state-action value function
        act = np.zeros((len(self.__axis_2D),)+env.act_sp_shape)
        Q_ax.invert_yaxis()
        Q_ax.set_title('State-action value function')
        Q_ax.plot_surface(self.__mx, self.__my, self.__Qcompute(obs, act).reshape((resolution, resolution)), cmap=self.__colormap)

        # Plot target state value function
        V_ax.invert_yaxis()
        V_ax.set_title('Target state value function')
        V_ax.plot_surface(self.__mx, self.__my, self.__VTcompute(obs).reshape((resolution, resolution)), cmap=self.__colormap)

        if(episode > resolution):
            # Policy training loss
            episode = int(episode/resolution)*resolution
            episodes = np.linspace(0, episode, num=resolution, endpoint=True)
            Pl_ax.set_title('Policy\nTraining Loss')
            data, color = self.__ep_loss[:episode, 1].reshape(resolution, -1), 'black'
            data_mean = np.mean(data, axis=1)
            data_perc = np.percentile(data, percentiles, axis=1)
            Pl_ax.plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
            for i in range(int(len(percentiles)/2) - 2):
                Pl_ax.fill_between(episodes, data_perc[i+1], data_perc[-i-1], color=color, alpha=0.1)
            Pl_ax.plot(episodes, data_perc[-1], color=color, alpha=0.3, linewidth=1, linestyle='--')
            Pl_ax.plot(episodes, data_perc[0], color=color, alpha=0.3, linewidth=1, linestyle='--')
            digits = int(np.floor(np.log10(np.max(np.abs([data_perc[0], data_perc[-1]])))) - 1)
            Pl_ax.set_yticks(np.linspace(round(np.min(data_perc[0])-0.5*pow(10,digits),-digits), round(np.max(data_perc[-1])+0.5*pow(10,digits),-digits), num=6, endpoint=True))

            # State-action value function training loss
            Ql_ax.set_title('State-Action Value Function\nTraining Loss')
            data, color = self.__ep_loss[:episode, 0].reshape(resolution, -1), 'black'
            data_mean = np.mean(data, axis=1)
            data_perc = np.percentile(data, percentiles, axis=1)
            Ql_ax.plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
            for i in range(int(len(percentiles)/2) - 2):
                Ql_ax.fill_between(episodes, data_perc[i+1], data_perc[-i-1], color=color, alpha=0.1)
            Ql_ax.plot(episodes, data_perc[-1], color=color, alpha=0.3, linewidth=1, linestyle='--')
            Ql_ax.plot(episodes, data_perc[0], color=color, alpha=0.3, linewidth=1, linestyle='--')
            digits = int(np.floor(np.log10(np.max(np.abs([data_perc[0], data_perc[-1]])))) - 1)
            Ql_ax.set_yticks(np.linspace(round(np.min(data_perc[0])-0.5*pow(10,digits),-digits), round(np.max(data_perc[-1])+0.5*pow(10,digits),-digits), num=6, endpoint=True))

            # Return comparison
            R_ax.set_title('Infinite-Horizon Discounted Return\nReal(red) vs. Predicted(blue)')
            for i in [0, 1]:
                data, color = self.__ep_ret[:episode, i].reshape(resolution, -1), ['red', 'blue'][i]
                data_mean = np.mean(data, axis=1)
                data_perc = np.percentile(data, percentiles, axis=1)
                R_ax.plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
                for i in range(int(len(percentiles)/2) - 2):
                    R_ax.fill_between(episodes, data_perc[i+1], data_perc[-i-1], color=color, alpha=0.1)
                R_ax.plot(episodes, data_perc[-1], color=color, alpha=0.3, linewidth=1, linestyle='--')
                R_ax.plot(episodes, data_perc[0], color=color, alpha=0.3, linewidth=1, linestyle='--')
            y_min, y_max = np.min(self.__ep_ret[:episode,0:2]), np.max(self.__ep_ret[:episode,0:2])
            digits = int(np.floor(np.log10(np.max(np.abs([y_min, y_max])))) - 1)
            R_ax.set_yticks(np.linspace(round(y_min-0.5*pow(10,digits),-digits), round(y_max+0.5*pow(10,digits),-digits), num=6, endpoint=True))

            # Return error
            Re_ax.set_title('Infinite-Horizon Discounted Return\n Root Mean Squared Error')
#            Re_ax.set_yscale("log")
            data, color = self.__ep_ret[:episode, 2].reshape(resolution, -1), 'purple'
            data_mean = np.mean(data, axis=1)
            data_perc = np.percentile(data, percentiles, axis=1)
            Re_ax.plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
            for i in range(int(len(percentiles)/2) - 2):
                Re_ax.fill_between(episodes, data_perc[i+1], data_perc[-i-1], color=color, alpha=0.1)
            Re_ax.plot(episodes, data_perc[-1], color=color, alpha=0.3, linewidth=1, linestyle='--')
            Re_ax.plot(episodes, data_perc[0], color=color, alpha=0.3, linewidth=1, linestyle='--')
            digits = int(np.floor(np.log10(np.max(np.abs([data_perc[0], data_perc[-1]])))) - 1)
            #Re_ax.set_yticks(np.logspace(round(np.min(data_perc[0])-0.5*pow(10,digits),-digits), round(np.max(data_perc[-1])+0.5*pow(10,digits),-digits), num=6, endpoint=True))
            Re_ax.set_yticks(np.linspace(round(np.min(data_perc[0])-0.5*pow(10,digits),-digits), round(np.max(data_perc[-1])+0.5*pow(10,digits),-digits), num=6, endpoint=True))

        # Update plot
        plt.draw()
        plt.pause(0.001)

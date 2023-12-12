import os
import numpy as np
import torch
from SAC import SAC_Agent
from EnvironmentTetrapod import Environment

import multiprocessing
import pyqtgraph as pg
import sys
from PyQt5.QtWidgets import QApplication
import time

data_type = np.float64

def SAC_Agent_Training(q):
    env = Environment(obs_sp_shape=(19,), act_sp_shape=(12,), dest_pos=(0,0))

    load_agent = True
    load_replay_buffer = True
    load_train_history = True
    episodes = 10000
    episode = 0
    episode_steps = 200 #Maximum steps allowed per episode
    save_period = 1000
    plot_period = 50

    agent = SAC_Agent('Cuadruped', env.obs_sp_shape[0], env.act_sp_shape[0], replay_buffer_size=1000000)

    agent.discount_factor = 0.95
    agent.update_factor = 0.005
    agent.replay_batch_size = 1000

    if load_agent:
        agent.load_models()

    ep_obs = np.zeros((episode_steps+1,) + env.obs_sp_shape, dtype=data_type)     # Episode's observed states
    ep_act = np.zeros((episode_steps,) + env.act_sp_shape, dtype=data_type)       # Episode's actions
    ep_rwd = np.zeros((episode_steps, 1), dtype=data_type)                          # Episode's rewards
    ep_ret = np.zeros((episodes, 3), dtype=data_type)                       # Returns for each episode (real, expected and RMSE)
    ep_loss = np.zeros((episodes, 2), dtype=data_type)                       # Training loss for each episode (Q and P)
    ep_alpha = np.zeros((episode_steps, 1), dtype=data_type)                # Alpha for each episode

    if load_train_history:
        # Check the last episode saved in Progress.txt
        if not os.path.isfile('./Train/Progress.txt'):
            print('Progress.txt could not be found')
            exit
        with open('./Train/Progress.txt', 'r') as file: last_episode = int(np.loadtxt(file))

        filename = './Train/Train_History_episode_{0:07d}.npz'.format(last_episode)
        loaded_arrays = np.load(filename)

        ep_ret[0:last_episode] = loaded_arrays['returns']
        ep_loss[0:last_episode] = loaded_arrays['loss']
        ep_alpha[0:last_episode] = loaded_arrays['alpha']

        if load_replay_buffer:
            agent.replay_buffer.load(last_episode)

    while episode < episodes:
        
        ep_obs[0], done_flag = env.reset(), False

        for step in range(episode_steps):
            # Decide action based on present observed state
            ep_act[step] = agent.choose_action(ep_obs[step])
            
            # Act in the environment
            ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])
            
            # Store in replay buffer
            agent.remember(ep_obs[step], ep_act[step], ep_rwd[step], ep_obs[step+1], done_flag)

            # End episode on termination condition
            if done_flag: break
        
        ep_len = step + 1


        # Compute the real and expected returns and the root mean square error:
        # Real return: If the episode ended because the agent reached the maximum steps allowed, the rest of the return is estimated with the Q function
        last_state = torch.tensor([ep_obs[step+1]], dtype=torch.float64).to(agent.P_net.device).view(-1)
        last_state = torch.unsqueeze(last_state, 0)
        
        last_action = agent.choose_action(ep_obs[step+1])
        last_action = torch.tensor([last_action], dtype=torch.float64).to(agent.P_net.device).view(-1)
        last_action = torch.unsqueeze(last_action, 0)
        
        if not done_flag: ep_rwd[step] += agent.discount_factor * agent.minimal_Q(last_state, last_action).detach().cpu().numpy().reshape(-1)
        for i in range(ep_len-2, -1, -1): ep_rwd[i] = ep_rwd[i] + agent.discount_factor * ep_rwd[i+1]
        ep_ret[episode, 0] = ep_rwd[0]

        # Expected return at the start of the episode
        initial_state = torch.tensor([ep_obs[0]], dtype=torch.float64).to(agent.P_net.device)
        initial_action = torch.tensor([ep_act[0]], dtype=torch.float64).to(agent.P_net.device)
        ep_ret[episode, 1] = agent.minimal_Q(initial_state, initial_action)

        # Root mean square error
        ep_ret[episode, 2] = np.sqrt(np.square(ep_ret[episode,0] - ep_ret[episode, 1]))

        for i in range(ep_len):
            ep_loss[episode, 0], ep_loss[episode, 1], ep_alpha[episode] = agent.learn()
        
        print("Episode: ", episode)
        print("Q_loss: ", ep_loss[episode, 0])
        print("P_loss: ", ep_loss[episode, 1])
        print("Alpha: ", ep_alpha[episode])

        q.put((episode, ep_ret[episode], ep_loss[episode], ep_alpha[episode]))
        
        episode += 1
        
        if episode % save_period == 0:
            agent.save_models()
            agent.replay_buffer.save(episode)
            
            filename = './Train/Train_History_episode_{0:07d}'.format(episode)
            np.savez_compressed(filename, returns = ep_ret[0:episode], loss = ep_loss[0:episode], alpha = ep_alpha)
    

# def updateplot(q):
#     while True:

#         results=q.get()  #Blocked until it gets a tuple



            
if __name__ == '__main__':
    app = QApplication(sys.argv)

    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    SAC_process=multiprocessing.Process(None,SAC_Agent_Training,args=(q,))
    SAC_process.start()

    # Create layout to hold multiple subplots
    pg_layout = pg.GraphicsLayoutWidget()

    # Create plot widgets (canvas)
    canvas_P_Loss = pg.PlotWidget(title="Policy Loss")
    canvas_Q_Loss = pg.PlotWidget(title="State-Value Loss")
    canvas_Returns = pg.PlotWidget(title="Real return vs Estimated return")
    canvas_Return_Error = pg.PlotWidget(title="RMSD of Real and Estimated")
   
    # Add subplots
    pg_layout.add_subplot(canvas_P_Loss, row=0, col=0)
    pg_layout.addWidget(canvas_Q_Loss, row=0, col=1)
    pg_layout.addWidget(canvas_Returns, row=1, col=0)
    pg_layout.addWidget(canvas_Return_Error, row=1, col=1)

    # Create plot of canvas
    P_Loss_plot = canvas_P_Loss.plot()
    Q_Loss_plot = canvas_Q_Loss.plot()
    Returns_plot = canvas_Returns.plot()
    Return_Error_plot = canvas_Return_Error.plot()

    canvas_P_Loss.showGrid(x=True, y=True, alpha=0.5)
    canvas_Q_Loss.showGrid(x=True, y=True, alpha=0.5)

    # Show our layout holding multiple subplots
    pg_layout.show()
    #Call a function to update the plot when there is new data
    # updateplot(q)

    # status = app.exec_()
    # sys.exit(status)
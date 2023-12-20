import os
import numpy as np
import torch
from SAC import SAC_Agent
from EnvironmentTetrapod import Environment

import multiprocessing
import queue
import pyqtgraph as pg

import sys

from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5 import QtCore

data_type = np.float64

def SAC_Agent_Training(q):
    global obs_plot, act_plot
    env = Environment(obs_sp_shape=(19,), act_sp_shape=(12,), dest_pos=(0,0))

    load_agent = True
    load_train_history = True
    load_replay_buffer = True   #(if load_train_history == false, the replay buffer is never loaded)
    
    episodes = 20000
    episode = 0
    episode_steps = 200 #Maximum steps allowed per episode
    save_period = 1000
    training_frequency = 3

    agent = SAC_Agent('Cuadruped', env.obs_sp_shape[0], env.act_sp_shape[0], replay_buffer_size=1000000)

    agent.discount_factor = 0.95
    agent.update_factor = 0.005
    agent.replay_batch_size = 10000

    if load_agent:
        agent.load_models()

    ep_obs = np.zeros((episode_steps+1,) + env.obs_sp_shape, dtype=data_type)   # Episode's observed states
    ep_act = np.zeros((episode_steps,) + env.act_sp_shape, dtype=data_type)     # Episode's actions
    ep_rwd = np.zeros((episode_steps,), dtype=data_type)                      # Episode's rewards
    ep_ret = np.zeros((episodes, 3), dtype=data_type)                           # Returns for each episode (real, expected and RMSE)
    ep_loss = np.zeros((episodes, 2), dtype=data_type)                          # Training loss for each episode (Q and P)
    ep_alpha = np.zeros((episodes,), dtype=data_type)                           # Alpha for each episode
    ep_entropy = np.zeros((episodes,), dtype=data_type)                         # Entropy of the policy for each episode

    if load_train_history:
        # Check the last episode saved in Progress.txt
        if not os.path.isfile('./Train/Progress.txt'):
            print('Progress.txt could not be found')
            exit
        with open('./Train/Progress.txt', 'r') as file: last_episode = int(np.loadtxt(file))

        filename = './Train/Train_History_episode_{0:07d}.npz'.format(last_episode)
        loaded_arrays = np.load(filename)

        ep_ret[0:last_episode+1] = loaded_arrays['returns']
        ep_loss[0:last_episode+1] = loaded_arrays['loss']
        ep_alpha[0:last_episode+1] = loaded_arrays['alpha']
        ep_entropy[0:last_episode+1] = loaded_arrays['entropy']

        if load_replay_buffer:
            agent.replay_buffer.load(last_episode)
        
        episode = last_episode + 1

    while episode <= episodes:
        
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
        
        aux_rwd = ep_rwd

        if not done_flag: aux_rwd[step] += agent.discount_factor * agent.minimal_Q(last_state, last_action).detach().cpu().numpy().reshape(-1)
        for i in range(ep_len-2, -1, -1): aux_rwd[i] = aux_rwd[i] + agent.discount_factor * aux_rwd[i+1]
        ep_ret[episode, 0] = aux_rwd[0]

        # Expected return at the start of the episode
        initial_state = torch.tensor([ep_obs[0]], dtype=torch.float64).to(agent.P_net.device)
        initial_action = torch.tensor([ep_act[0]], dtype=torch.float64).to(agent.P_net.device)
        ep_ret[episode, 1] = agent.minimal_Q(initial_state, initial_action)

        # Root mean square error
        ep_ret[episode, 2] = np.sqrt(np.square(ep_ret[episode,0] - ep_ret[episode, 1]))

        for i in range(ep_len):
            if i % training_frequency == 0:
                agent.learn()
        
        ep_loss[episode, 0] = agent.Q_loss.item()
        ep_loss[episode, 1] = agent.P_loss.item()
        ep_alpha[episode] = agent.alpha.item()
        ep_entropy[episode] = agent.entropy.item()
        
        print("Episode: ", episode)
        print("Replay_Buffer_counter: ", agent.replay_buffer.mem_counter)
        print("Q_loss: ", ep_loss[episode, 0])
        print("P_loss: ", ep_loss[episode, 1])
        print("Alpha: ", ep_alpha[episode])
        print("Policy's Entropy: ", ep_entropy[episode])
        print("------------------------------------------")

        q.put((episode, ep_ret[0:episode+1], ep_loss[0:episode+1], ep_alpha[0:episode+1], ep_obs[0:ep_len+1, 0], ep_obs[0:ep_len+1, 1], ep_entropy[0:episode+1], ep_rwd[0:ep_len+1]))
        
        if episode % save_period == 0:
            agent.save_models()
            agent.replay_buffer.save(episode)
            
            filename = './Train/Train_History_episode_{0:07d}'.format(episode)
            np.savez_compressed(filename, returns = ep_ret[0:episode+1], loss = ep_loss[0:episode+1], alpha = ep_alpha[0:episode+1], entropy = ep_entropy[0:episode+1])
        
        episode += 1

def updatePlot():   
    global q, curve_Trajectory,curve_Trajectory_startPoint, curve_P_Loss,curve_Q_Loss,curve_Real_Return, curve_Predicted_Return,curve_Return_Error,curve_Alpha, curve_Entropy, curve_StepReward
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()
        last_episode = results[0]
        episode_linspace = np.arange(0,last_episode+1,1,dtype=int)
        Real_Return_data = results[1][:,0]
        Predicted_Return_data = results[1][:,1]
        Return_loss_data = results[1][:,2]
        Q_loss_data = results[2][:,0]
        P_loss_data = results[2][:,1]
        Alpha_data = results[3]
        Trajectory_x_data = results[4]
        Trajectory_y_data = results[5]
        Entropy_data = results[6]
        Step_rwd = results[7]
        steps_linspace = np.arange(0,len(Step_rwd), 1, dtype=int)

        curve_Trajectory.setData(Trajectory_x_data,Trajectory_y_data)
        curve_Trajectory_startPoint.setData([Trajectory_x_data[0]], [Trajectory_y_data[0]])
        curve_StepReward.setData(steps_linspace, Step_rwd)
        curve_P_Loss.setData(episode_linspace,P_loss_data)
        curve_Q_Loss.setData(episode_linspace,Q_loss_data)
        curve_Real_Return.setData(episode_linspace,Real_Return_data)
        curve_Predicted_Return.setData(episode_linspace, Predicted_Return_data)
        curve_Return_Error.setData(episode_linspace,Return_loss_data)
        curve_Alpha.setData(episode_linspace,Alpha_data)
        curve_Entropy.setData(episode_linspace, Entropy_data)

    except queue.Empty:
        #print("Empty Queue")
        pass

if __name__ == '__main__':
    global q, curve_Trajectory,curve_Trajectory_startPoint, curve_P_Loss, curve_Q_Loss, curve_Returns, curve_Return_Error, curve_Alpha, curve_Entropy, curve_StepReward
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    app = QApplication(sys.argv)

    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    SAC_process=multiprocessing.Process(None,SAC_Agent_Training,args=(q,))
    SAC_process.start()

    # Create window
    
    grid_layout = pg.GraphicsLayoutWidget(title="Cuadruped - Training information")
    grid_layout.resize(1305,715)
    
    pg.setConfigOptions(antialias=True)

    plot_Trajectory = grid_layout.addPlot(title="Trajectory in Last Episode", row=0, col=0)

    plot_StepReward = grid_layout.addPlot(title="Reward per Step in Last Episode", row=0, col=1)
    plot_StepReward.showGrid(x=True, y=True)

    plot_Returns = grid_layout.addPlot(title="Real Return vs Predicted Return", row=0, col=2)
    plot_Returns.addLegend()
    plot_Returns.showGrid(x=True, y=True)

    plot_Return_Error = grid_layout.addPlot(title="RMSD of Real and Predicted Return", row=0, col=3)
    plot_Return_Error.showGrid(x=True, y=True)
        
    plot_Q_Loss = grid_layout.addPlot(title="State-Action Value Loss", row=1, col=0)
    plot_Q_Loss.showGrid(x=True, y=True)
    
    plot_P_Loss = grid_layout.addPlot(title="Policy Loss", row=1, col=1)
    plot_P_Loss.showGrid(x=True, y=True)

    plot_Alpha = grid_layout.addPlot(title="Alpha (Entropy Regularization Coefficient)", row=1, col=2)
    plot_Alpha.showGrid(x=True, y=True)

    plot_Entropy = grid_layout.addPlot(title="Policy's Entropy", row=1, col=3)
    plot_Entropy.showGrid(x=True, y=True)

    grid_layout.ci.layout.setColumnMinimumWidth(0,310)
    grid_layout.ci.layout.setColumnMinimumWidth(1,310)
    grid_layout.ci.layout.setColumnMinimumWidth(2,310)
    grid_layout.ci.layout.setColumnMinimumWidth(3,310)
    grid_layout.ci.layout.setRowMinimumHeight(0,340)
    grid_layout.ci.layout.setRowMinimumHeight(1,340)
    grid_layout.ci.layout.setSpacing(15)

    # Drawing of the scene in the trajectory plot:

        # Add point in the center of the scene, target of the agent
    plot_Trajectory.plot([0], [0], pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 255, 255, 200))

        # Add circule to delimitate scene of trajectory plot
    circle = QGraphicsEllipseItem(-3, -3, 6, 6)  # x, y, width, height
    circle.setPen(pg.mkPen((255, 255, 255, 255), width=2))
    circle.setBrush(pg.mkBrush(None))

        # The square delimitates the zone where de agent can appear
    square = QGraphicsRectItem(-2, -2, 4, 4)
    square.setPen(pg.mkPen((255,255,255,100), width=1, style=QtCore.Qt.DashLine))
    square.setBrush(pg.mkBrush(None))

    plot_Trajectory.addItem(circle)
    plot_Trajectory.addItem(square)
    plot_Trajectory.setRange(xRange=(-3,3), yRange=(-3,3), padding=None, update=True, disableAutoRange=True)

    #Curves to update them in updatePlot()
    curve_Trajectory = plot_Trajectory.plot()
    curve_Trajectory_startPoint = plot_Trajectory.plot(pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 255, 0, 150))
    curve_StepReward = plot_StepReward.plot(pen=(255,0,0))
    curve_Real_Return = plot_Returns.plot(pen=(255,0,0), name='Real')
    curve_Predicted_Return = plot_Returns.plot(pen=(0,255,0), name='Predicted')
    curve_Return_Error = plot_Return_Error.plot(pen=(182,102,247))
    curve_Q_Loss = plot_Q_Loss.plot(pen=(0,255,0))
    curve_P_Loss = plot_P_Loss.plot(pen=(0,128,255))
    curve_Alpha = plot_Alpha.plot(pen=(255,150,45))
    curve_Entropy = plot_Entropy.plot(pen=(0,255,255))
    
    #Timer to update plots every 1 second (if there is new data) in another thread
    timer = QtCore.QTimer()
    timer.timeout.connect(updatePlot)
    timer.start(1000)
    
    grid_layout.show()

    status = app.exec_()
    sys.exit(status)
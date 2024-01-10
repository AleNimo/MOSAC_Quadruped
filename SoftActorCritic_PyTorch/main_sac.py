import os
import numpy as np
import torch
from SAC import SAC_Agent
from EnvironmentTetrapodVelocidad import Environment

import multiprocessing
import queue
import pyqtgraph as pg

import sys

from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5 import QtCore

data_type = np.float64

def SAC_Agent_Training(q):

    env = Environment(obs_sp_shape=(21,), act_sp_shape=(12,), dest_pos=(0,0))

    #The 23 values from coppelia, in order:
        #---- Not seen by the agent --------
        #position (x,y,z)
        #target_direction (x,y)
        #-----------------------------------
    
        #pitch and roll of the agent's body
        ##############cosine and sine of the signed angle between the agent and the target direction (CURRENTLY NOT)
        #Forward and lateral velocity with the reference frame of the agent
        #the 12 joint angles

    load_agent = False
    test_agent = False
    load_train_history = False   #(if test_agent == True, the train history and the replay buffer are never loaded)
    load_replay_buffer = False   #(if load_train_history == false, the replay buffer is never loaded)
    
    episodes = 20000
    episode = 0
    episode_steps = 200 #Maximum steps allowed per episode
    save_period = 1000

    #The vector received from coppelia contains the x,y,z coordinates and the target direction not used by the agent, only for plotting. Thats why we subtract the 5 values from the vector dimensions
    agent = SAC_Agent('Cuadruped', env.obs_sp_shape[0]-5, env.act_sp_shape[0], replay_buffer_size=1000000)
    
    agent.replay_batch_size = 10000

    agent.update_Q = 1  # The Q function is updated every episode
    agent.update_P = 1  # The policy is updated every 3 episodes

    if load_agent:
        agent.load_models()

    ep_obs = np.zeros((episode_steps+1,) + env.obs_sp_shape, dtype=data_type)   # Episode's observed states
    ep_act = np.zeros((episode_steps,) + env.act_sp_shape, dtype=data_type)     # Episode's actions
    ep_rwd = np.zeros((episode_steps,), dtype=data_type)                        # Episode's rewards
    ep_ret = np.zeros((episodes, 3), dtype=data_type)                           # Returns for each episode (real, expected and RMSE)
    ep_loss = np.zeros((episodes, 2), dtype=data_type)                          # Training loss for each episode (Q and P)
    ep_alpha = np.zeros((episodes,), dtype=data_type)                           # Alpha for each episode
    ep_entropy = np.zeros((episodes,), dtype=data_type)                         # Entropy of the policy for each episode

    if load_train_history and test_agent==False:
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

    # Training
    while episode <= episodes:

        ep_obs[0], done_flag = env.reset(), False
        # Testing
        if test_agent:
            for step in range(episode_steps):
                # Decide action based on present observed state (taking only the mean)
                ep_act[step] = agent.choose_action(ep_obs[step][5:], random=False)  #The agent doesn't receive the position and target direction although it is on the ep_obs vector for plotting reasons

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                if done_flag: break

            ep_len = step + 1

        else:
            for step in range(episode_steps):
                # Decide action based on present observed state (random action with mean and std)
                ep_act[step] = agent.choose_action(ep_obs[step][5:])

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                # Store in replay buffer
                agent.remember(ep_obs[step][5:], ep_act[step], ep_rwd[step], ep_obs[step+1][5:], done_flag)

                # End episode on termination condition
                if done_flag: break

        ep_len = step + 1

        # Compute the real and expected returns and the root mean square error:
        # Real return: If the episode ended because the agent reached the maximum steps allowed, the rest of the return is estimated with the Q function
        last_state = torch.tensor([ep_obs[step+1][5:]], dtype=torch.float64).to(agent.P_net.device).view(-1)
        last_state = torch.unsqueeze(last_state, 0)
        
        last_action = agent.choose_action(ep_obs[step+1][5:], random=not(test_agent))
        last_action = torch.tensor([last_action], dtype=torch.float64).to(agent.P_net.device).view(-1)
        last_action = torch.unsqueeze(last_action, 0)
        
        aux_rwd = np.copy(ep_rwd)

        if not done_flag: aux_rwd[step] += agent.discount_factor * agent.minimal_Q(last_state, last_action).detach().cpu().numpy().reshape(-1)
        for i in range(ep_len-2, -1, -1): aux_rwd[i] = aux_rwd[i] + agent.discount_factor * aux_rwd[i+1]
        ep_ret[episode, 0] = aux_rwd[0]

        # Expected return at the start of the episode
        initial_state = torch.tensor([ep_obs[0][5:]], dtype=torch.float64).to(agent.P_net.device)
        initial_action = torch.tensor([ep_act[0]], dtype=torch.float64).to(agent.P_net.device)
        ep_ret[episode, 1] = agent.minimal_Q(initial_state, initial_action)

        # Root mean square error
        ep_ret[episode, 2] = np.sqrt(np.square(ep_ret[episode,0] - ep_ret[episode, 1]))

        if test_agent == False:
            for step in range(ep_len):
                agent.learn(step)

        ep_loss[episode, 0] = agent.Q_loss.item()
        ep_loss[episode, 1] = agent.P_loss.item()
        ep_alpha[episode] = agent.log_alpha.exp().item()
        ep_entropy[episode] = agent.entropy.item()
        
        print("Episode: ", episode)
        print("Replay_Buffer_counter: ", agent.replay_buffer.mem_counter)
        print("Q_loss: ", ep_loss[episode, 0])
        print("P_loss: ", ep_loss[episode, 1])
        print("Alpha: ", ep_alpha[episode])
        print("Policy's Entropy: ", ep_entropy[episode])
        print("------------------------------------------")

        q.put((episode, ep_obs[0:ep_len+1], ep_rwd[0:ep_len+1], ep_ret[0:episode+1], ep_loss[0:episode+1], ep_alpha[0:episode+1], ep_entropy[0:episode+1]))
        
        if (episode % save_period == 0 or episode == 50) and test_agent == False:
            agent.save_models()
            agent.replay_buffer.save(episode)
            
            filename = './Train/Train_History_episode_{0:07d}'.format(episode)
            np.savez_compressed(filename, returns = ep_ret[0:episode+1], loss = ep_loss[0:episode+1], alpha = ep_alpha[0:episode+1], entropy = ep_entropy[0:episode+1])
        
        episode += 1

def updatePlot():   
    global q, curve_Trajectory, curve_Trajectory_startPoint,curve_Trajectory_target, curve_ForwardVelocity, curve_LateralVelocity, curve_StepReward, curve_P_Loss, curve_Q_Loss, curve_Real_Return, curve_Predicted_Return, curve_Return_Error, curve_Alpha, curve_Entropy
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()
        last_episode = results[0]
        episode_linspace = np.arange(0,last_episode+1,1,dtype=int)
        
        Trajectory_x_data = results[1][:,0]
        Trajectory_y_data = results[1][:,1]

        target_direction = results[1][0,3:5]    #Currently constant through out the episode

        forward_velocity = results[1][:,7]      #For each step of the episode
        lateral_velocity = results[1][:,8]

        Trajectory_x_target = np.array([0, target_direction[0]*3])
        Trajectory_y_target = np.array([0, target_direction[1]*3])

        Step_rwd = results[2]

        rwd_linspace = np.arange(1,len(Step_rwd)+1, 1, dtype=int)   #Because there is no reward in the first state (step 0)
        vel_linspace = np.arange(0,len(forward_velocity), 1, dtype=int)

        Real_Return_data = results[3][:,0]
        Predicted_Return_data = results[3][:,1]
        Return_loss_data = results[3][:,2]

        Q_loss_data = results[4][:,0]
        P_loss_data = results[4][:,1]

        Alpha_data = results[5]
        
        Entropy_data = results[6]

        curve_Trajectory.setData(Trajectory_x_data,Trajectory_y_data)
        curve_Trajectory_startPoint.setData([Trajectory_x_data[0]], [Trajectory_y_data[0]])
        curve_Trajectory_target.setData(Trajectory_x_target, Trajectory_y_target)
        curve_ForwardVelocity.setData(vel_linspace, forward_velocity)
        curve_LateralVelocity.setData(vel_linspace, lateral_velocity)
        curve_StepReward.setData(rwd_linspace, Step_rwd)
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
    global q, curve_Trajectory, curve_Trajectory_startPoint, curve_Trajectory_target, curve_ForwardVelocity, curve_LateralVelocity, curve_StepReward, curve_P_Loss, curve_Q_Loss, curve_Real_Return, curve_Predicted_Return, curve_Return_Error, curve_Alpha, curve_Entropy
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    app = QApplication(sys.argv)

    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    SAC_process=multiprocessing.Process(None,SAC_Agent_Training,args=(q,))
    SAC_process.start()

    # Create window
    
    grid_layout = pg.GraphicsLayoutWidget(title="Cuadruped - Training information")
    grid_layout.resize(1625,715)
    
    pg.setConfigOptions(antialias=True)

    plot_Trajectory = grid_layout.addPlot(title="Trajectory with Target Direction", row=0, col=0)

    plot_StepVelocity = grid_layout.addPlot(title="Forward and lateral velocity per Step", row=0, col=1)
    plot_StepVelocity.addLegend()
    plot_StepVelocity.showGrid(x=True, y=True)

    plot_StepReward = grid_layout.addPlot(title="Reward per Step", row=0, col=2)
    plot_StepReward.showGrid(x=True, y=True)

    plot_P_Loss = grid_layout.addPlot(title="Policy Loss", row=0, col=3, colspan=2)
    plot_P_Loss.showGrid(x=True, y=True)

    plot_Returns = grid_layout.addPlot(title="Real Return vs Predicted Return", row=1, col=0)
    plot_Returns.addLegend()
    plot_Returns.showGrid(x=True, y=True)

    plot_Return_Error = grid_layout.addPlot(title="RMSD of Real and Predicted Return", row=1, col=1)
    plot_Return_Error.showGrid(x=True, y=True)
        
    plot_Q_Loss = grid_layout.addPlot(title="State-Action Value Loss", row=1, col=2)
    plot_Q_Loss.showGrid(x=True, y=True)

    plot_Alpha = grid_layout.addPlot(title="Alpha (Entropy Regularization Coefficient)", row=1, col=3)
    plot_Alpha.showGrid(x=True, y=True)

    plot_Entropy = grid_layout.addPlot(title="Policy's Entropy", row=1, col=4)
    plot_Entropy.showGrid(x=True, y=True)

    grid_layout.ci.layout.setColumnMinimumWidth(0,310)
    grid_layout.ci.layout.setColumnMinimumWidth(1,310)
    grid_layout.ci.layout.setColumnMinimumWidth(2,310)
    grid_layout.ci.layout.setColumnMinimumWidth(3,310)
    grid_layout.ci.layout.setColumnMinimumWidth(4,310)
    grid_layout.ci.layout.setRowMinimumHeight(0,340)
    grid_layout.ci.layout.setRowMinimumHeight(1,340)
    grid_layout.ci.layout.setSpacing(15)

    # Drawing of the scene in the trajectory plot:

        # Point in the center of the scene, target of the agent
    plot_Trajectory.plot([0], [0], pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 255, 255, 200))

        # Circule to delimitate scene of trajectory plot
    scene_limit = QGraphicsEllipseItem(-3, -3, 6, 6)  # x, y, width, height
    scene_limit.setPen(pg.mkPen((255, 255, 255, 255), width=2))
    scene_limit.setBrush(pg.mkBrush(None))

        # Square to delimitate the zone where de agent can appear (Currently not used)
    square = QGraphicsRectItem(-2, -2, 4, 4)
    square.setPen(pg.mkPen((255,255,255,100), width=1, style=QtCore.Qt.DashLine))
    square.setBrush(pg.mkBrush(None))

        # Circule to delimitate the end condition of the episode (the goal to reach)
    end_limit = QGraphicsEllipseItem(-2.5, -2.5, 5, 5)  # x, y, width, height
    end_limit.setPen(pg.mkPen((0, 255, 0, 100), width=1, style=QtCore.Qt.DashLine))
    end_limit.setBrush(pg.mkBrush(None))

    plot_Trajectory.addItem(scene_limit)
    plot_Trajectory.addItem(square)
    plot_Trajectory.addItem(end_limit)

    plot_Trajectory.setRange(xRange=(-3,3), yRange=(-3,3), padding=None, update=True, disableAutoRange=True)

    #Curves to update them in updatePlot()
    curve_Trajectory = plot_Trajectory.plot()
    curve_Trajectory_startPoint = plot_Trajectory.plot(pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 255, 0, 150))
    curve_Trajectory_target = plot_Trajectory.plot(pen=pg.mkPen((255,201,14,255), width=1, style=QtCore.Qt.DashLine))
    curve_ForwardVelocity = plot_StepVelocity.plot(pen=(0,255,0), name='Forward')
    curve_LateralVelocity = plot_StepVelocity.plot(pen=(255,0,0), name='Lateral')
    curve_StepReward = plot_StepReward.plot(pen=(255,201,14))
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
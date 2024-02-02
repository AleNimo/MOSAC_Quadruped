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

    env = Environment(obs_sp_shape=(24,), act_sp_shape=(12,), dest_pos=(0,0))

    #The 24 values from coppelia, in order:
        #---- Not seen by the agent --------
        #position (x,y,z)
        #target_direction (x,y)
        #-----------------------------------
    
        #pitch and roll of the agent's body
        #cosine and sine of the signed angle between the agent and the target direction
        #Mean forward and lateral velocities with the reference frame of the agent
        #Maximum absolute forward acceleration measured during the previous step with the reference frame of the agent
        #the 12 joint angles

    load_agent = False
    test_agent = False
    load_train_history = False   #(if test_agent == True, the train history and the replay buffer are never loaded)
    load_replay_buffer = False   #(if load_train_history == false, the replay buffer is never loaded)
    
    episodes = 20000
    episode = 0
    episode_steps = 200 #Maximum steps allowed per episode
    save_period = 500

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
    ep_ind_rwd = np.zeros((episode_steps, 6), dtype=data_type)                  # Epidose's individual rewards
    ep_ret = np.zeros((episodes, 3), dtype=data_type)                           # Returns for each episode (real, expected and RMSE)
    ep_loss = np.zeros((episodes, 2), dtype=data_type)                          # Training loss for each episode (Q and P)
    ep_alpha = np.zeros((episodes,), dtype=data_type)                           # Alpha for each episode
    ep_entropy = np.zeros((episodes,), dtype=data_type)                         # Entropy of the policy for each episode
    ep_std = np.zeros((episodes,), dtype=data_type)                             # Mean standard deviation of the policy for each episode

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
        ep_std[0:last_episode+1] = loaded_arrays['std']

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

                ep_ind_rwd[step, :] = [env.forward_velocity_reward, env.lateral_velocity_penalty, env.orientation_reward, env.flat_back_reward[0], env.flat_back_reward[1], env.forward_acc_penalty]

                if done_flag: break

            ep_len = step + 1

        else:
            for step in range(episode_steps):
                # Decide action based on present observed state (random action with mean and std)
                ep_act[step] = agent.choose_action(ep_obs[step][5:])

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                ep_ind_rwd[step, :] = [env.forward_velocity_reward, env.lateral_velocity_penalty, env.orientation_reward, env.flat_back_reward[0], env.flat_back_reward[1], env.forward_acc_penalty]

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
        ep_std[episode] = agent.std.item()
        
        print("Episode: ", episode)
        print("Replay_Buffer_counter: ", agent.replay_buffer.mem_counter)
        print("Q_loss: ", ep_loss[episode, 0])
        print("P_loss: ", ep_loss[episode, 1])
        print("Alpha: ", ep_alpha[episode])
        print("Policy's Entropy: ", ep_entropy[episode])
        print("------------------------------------------")

        q.put((episode, ep_obs[0:ep_len+1], ep_rwd[0:ep_len+1], ep_ind_rwd[0:ep_len+1], ep_ret[0:episode+1], ep_loss[0:episode+1], ep_alpha[0:episode+1], ep_entropy[0:episode+1], ep_act[0:ep_len+1], ep_std[0:episode+1]))
        
        if episode % save_period == 0 and episode != 0 and test_agent == False:
            agent.save_models()
            agent.replay_buffer.save(episode)
            
            filename = './Train/Train_History_episode_{0:07d}'.format(episode)
            np.savez_compressed(filename, returns = ep_ret[0:episode+1], loss = ep_loss[0:episode+1], alpha = ep_alpha[0:episode+1], entropy = ep_entropy[0:episode+1], std = ep_std[0:episode+1])
        
        episode += 1

body_min, body_max = -10.0, 15.0
body_mean = (body_min + body_max)/2
body_range = (body_max - body_min)/2

leg_min, leg_max = -10.0, 40.0
# leg_min, leg_max = 0.0, 70.0
leg_mean = (leg_min + leg_max)/2
leg_range = (leg_max - leg_min)/2

paw_min, paw_max = -15.0,  5.0
# paw_min, paw_max = -50.0,  10.0
paw_mean = (paw_min + paw_max)/2
paw_range = (paw_max - paw_min)/2

def updatePlot():   
    global q, curve_Trajectory, curve_Trajectory_startPoint, curve_Trajectory_target, curve_ForwardVelocity, curve_LateralVelocity, curve_ForwardAcc, curve_Pitch, \
        curve_Roll, curve_gamma, curve_Reward, curve_Forward_vel_rwd, curve_Lateral_vel_rwd, curve_Orientation_rwd, curve_Pitch_rwd, curve_Roll_rwd, \
        curve_Acc_rwd, curve_P_Loss, curve_Q_Loss, curve_Real_Return, curve_Predicted_Return, curve_Return_Error, curve_Alpha, curve_Entropy, curve_Std, \
        curve_FrontBody_right_state, curve_FrontBody_left_state, curve_FrontBody_right_action, curve_FrontBody_left_action, \
        curve_BackBody_right_state, curve_BackBody_left_state, curve_BackBody_right_action, curve_BackBody_left_action, \
        curve_FrontLeg_right_state, curve_FrontLeg_left_state, curve_FrontLeg_right_action, curve_FrontLeg_left_action, \
        curve_BackLeg_right_state, curve_BackLeg_left_state, curve_BackLeg_right_action, curve_BackLeg_left_action, \
        curve_FrontPaw_right_state, curve_FrontPaw_left_state, curve_FrontPaw_right_action, curve_FrontPaw_left_action, \
        curve_BackPaw_right_state, curve_BackPaw_left_state, curve_BackPaw_right_action, curve_BackPaw_left_action, \
        body_joints_state, body_joints_action, leg_joints_state, leg_joints_action, paw_joints_state, paw_joints_action
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()

        last_episode = results[0]
        episode_linspace = np.arange(0,last_episode+1,1,dtype=int)

        ####Trajectory update
        Trajectory_x_data = results[1][:,0]
        Trajectory_y_data = results[1][:,1]

        target_direction = results[1][0,3:5]    #Currently constant through out the episode

        target_direction = target_direction / np.sqrt(np.sum(np.square(target_direction)))

        Trajectory_x_target = np.array([0, target_direction[0]*3])
        Trajectory_y_target = np.array([0, target_direction[1]*3])

        curve_Trajectory.setData(Trajectory_x_data,Trajectory_y_data)
        curve_Trajectory_startPoint.setData([Trajectory_x_data[0]], [Trajectory_y_data[0]])
        curve_Trajectory_target.setData(Trajectory_x_target, Trajectory_y_target)

        ####Velocities update
        forward_velocity = results[1][:,9]      #For each step of the episode
        lateral_velocity = results[1][:,10]

        state_linspace = np.arange(0,len(forward_velocity), 1, dtype=int)

        curve_ForwardVelocity.setData(state_linspace, forward_velocity)
        curve_LateralVelocity.setData(state_linspace, lateral_velocity)

        ####Acceleration update
        forward_acc = results[1][:,11]

        curve_ForwardAcc.setData(state_linspace, forward_acc)

        ####Inclination and Orientation update
        pitch = results[1][:,5] * 180 #deg
        roll = results[1][:,6] * 180 #deg
        gamma = np.arctan2(results[1][:,8], results[1][:,7]) * 180/np.pi    #Angle between agent and target direction in deg

        curve_Pitch.setData(state_linspace, pitch)
        curve_Roll.setData(state_linspace, roll)
        curve_gamma.setData(state_linspace, gamma)

        ####Rewards update
        total_rwd = results[2]
        forward_velocity_rwd = results[3][:,0]
        lateral_velocity_rwd = results[3][:,1]
        orientation_rwd = results[3][:,2]
        pitch_rwd = results[3][:,3]
        roll_rwd = results[3][:,4]
        acc_rwd = results[3][:,5]

        rwd_linspace = np.arange(1,len(total_rwd)+1, 1, dtype=int)   #Because there is no reward in the first state (step 0)

        curve_Reward.setData(rwd_linspace, total_rwd)
        curve_Forward_vel_rwd.setData(rwd_linspace, forward_velocity_rwd)
        curve_Lateral_vel_rwd.setData(rwd_linspace, lateral_velocity_rwd)
        curve_Orientation_rwd.setData(rwd_linspace, orientation_rwd)
        curve_Pitch_rwd.setData(rwd_linspace, pitch_rwd)
        curve_Roll_rwd.setData(rwd_linspace, roll_rwd)
        curve_Acc_rwd.setData(rwd_linspace, acc_rwd)

        ####Body joints update
        body_joints_state = []
        body_joints_action = []
        for i in range(4):
            body_joints_state.append(results[1][:,12+i*3] * body_range + body_mean)
            body_joints_action.append(results[8][:,i*3] * body_range + body_mean)

        action_linspace = np.arange(1,len(body_joints_action[0])+1,1, dtype=int)    #Because no action has been made in the step 0

        curve_FrontBody_right_state.setData(state_linspace, body_joints_state[0])
        curve_FrontBody_left_state.setData(state_linspace, body_joints_state[1])
        curve_FrontBody_right_action.setData(action_linspace, body_joints_action[0])
        curve_FrontBody_left_action.setData(action_linspace, body_joints_action[1])

        curve_BackBody_right_state.setData(state_linspace, body_joints_state[2])
        curve_BackBody_left_state.setData(state_linspace, body_joints_state[3])
        curve_BackBody_right_action.setData(action_linspace, body_joints_action[2])
        curve_BackBody_left_action.setData(action_linspace, body_joints_action[3])

        ####Leg joints update
        leg_joints_state = []
        leg_joints_action = []
        for i in range(4):
            leg_joints_state.append(results[1][:,13+i*3] * leg_range + leg_mean)
            leg_joints_action.append(results[8][:,1+i*3] * leg_range + leg_mean)

        curve_FrontLeg_right_state.setData(state_linspace, leg_joints_state[0])
        curve_FrontLeg_left_state.setData(state_linspace, leg_joints_state[1])
        curve_FrontLeg_right_action.setData(action_linspace, leg_joints_action[0])
        curve_FrontLeg_left_action.setData(action_linspace, leg_joints_action[1])

        curve_BackLeg_right_state.setData(state_linspace, leg_joints_state[2])
        curve_BackLeg_left_state.setData(state_linspace, leg_joints_state[3])
        curve_BackLeg_right_action.setData(action_linspace, leg_joints_action[2])
        curve_BackLeg_left_action.setData(action_linspace, leg_joints_action[3])

        ####Paw joints update
        paw_joints_state = []
        paw_joints_action = []
        for i in range(4):
            paw_joints_state.append(results[1][:,14+i*3] * paw_range + paw_mean)
            paw_joints_action.append(results[8][:,2+i*3] * paw_range + paw_mean)

        curve_FrontPaw_right_state.setData(state_linspace, paw_joints_state[0])
        curve_FrontPaw_left_state.setData(state_linspace, paw_joints_state[1])
        curve_FrontPaw_right_action.setData(action_linspace, paw_joints_action[0])
        curve_FrontPaw_left_action.setData(action_linspace, paw_joints_action[1])

        curve_BackPaw_right_state.setData(state_linspace, paw_joints_state[2])
        curve_BackPaw_left_state.setData(state_linspace, paw_joints_state[3])
        curve_BackPaw_right_action.setData(action_linspace, paw_joints_action[2])
        curve_BackPaw_left_action.setData(action_linspace, paw_joints_action[3])

        ####Returns update
        Real_Return_data = results[4][:,0]
        Predicted_Return_data = results[4][:,1]

        curve_Real_Return.setData(episode_linspace,Real_Return_data)
        curve_Predicted_Return.setData(episode_linspace, Predicted_Return_data)

        ####Returns error update
        Return_loss_data = results[4][:,2]

        curve_Return_Error.setData(episode_linspace,Return_loss_data)

        ####Qloss update
        Q_loss_data = results[5][:,0]

        curve_Q_Loss.setData(episode_linspace,Q_loss_data)

        ####Ploss update
        P_loss_data = results[5][:,1]

        curve_P_Loss.setData(episode_linspace,P_loss_data)

        ####Alpha update
        Alpha_data = results[6]
        curve_Alpha.setData(episode_linspace,Alpha_data)
        
        ####Entropy update
        Entropy_data = results[7]

        curve_Entropy.setData(episode_linspace, Entropy_data)

        ####Standard deviation update
        Std_data = results[9]

        curve_Std.setData(episode_linspace, Std_data)

    except queue.Empty:
        #print("Empty Queue")
        pass

if __name__ == '__main__':
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    app = QApplication(sys.argv)

    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    SAC_process=multiprocessing.Process(None,SAC_Agent_Training,args=(q,))
    SAC_process.start()

    # Create window
    
    grid_layout = pg.GraphicsLayoutWidget(title="Cuadruped - Training information")
    grid_layout.resize(1200,800)
    
    pg.setConfigOptions(antialias=True)

############################################### PLOTS #####################################################################

    ####Trajectory plot
    plot_Trajectory = grid_layout.addPlot(title="Trajectory with Target Direction", row=0, col=0)

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


    ####Velocity plot
    plot_Velocity = grid_layout.addPlot(title="Mean forward and lateral velocity (m/s)", row=0, col=1)
    plot_Velocity.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Velocity.showGrid(x=True, y=True)

    curve_ForwardVelocity = plot_Velocity.plot(pen=(0,255,0), name='Forward')
    curve_LateralVelocity = plot_Velocity.plot(pen=(255,0,0), name='Lateral')


    ####Acceleration plot
    plot_Acceleration = grid_layout.addPlot(title="Mean absolute forward acceleration (m/s^2)", row=0, col=2)
    plot_Acceleration.showGrid(x=True, y=True)

    curve_ForwardAcc = plot_Acceleration.plot(pen=(0,255,0))


    ####Inclination plot
    plot_Inclination = grid_layout.addPlot(title="Pitch and Roll (°)", row=0, col=3)
    plot_Inclination.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Inclination.showGrid(x=True, y=True)

    curve_Pitch = plot_Inclination.plot(pen=(0,255,255), name='Pitch')
    curve_Roll = plot_Inclination.plot(pen=(255,0,255), name='Roll')
    

    ####Orientation plot
    plot_Orientation = grid_layout.addPlot(title="Angle to target direction (°)", row=0, col=4)
    plot_Orientation.showGrid(x=True, y=True)

    curve_gamma = plot_Orientation.plot(pen=(255,127,39), name='Agent2Target')

   
    ####Rewards plot
    plot_Rewards = grid_layout.addPlot(title="Total and individual Rewards", row=0, col=5)
    plot_Rewards.addLegend(offset=(1, 1), verSpacing=-1, horSpacing = 20, labelTextSize = '7pt', colCount=3)
    plot_Rewards.showGrid(x=True, y=True)

    curve_Reward = plot_Rewards.plot(pen=(255,255,0), name='Total')
    curve_Forward_vel_rwd = plot_Rewards.plot(pen=(0,255,0), name='Forward')
    curve_Lateral_vel_rwd = plot_Rewards.plot(pen=(255,0,0), name='Lateral')
    curve_Orientation_rwd = plot_Rewards.plot(pen=(255,127,39), name='Orientation')
    curve_Pitch_rwd = plot_Rewards.plot(pen=(0,255,255), name='Pitch')
    curve_Roll_rwd = plot_Rewards.plot(pen=(255,0,255), name='Roll')
    curve_Acc_rwd = plot_Rewards.plot(pen=(255,255,255), name='Fwd_Acc')


    ####Front body joints plot
    plot_FrontBody = grid_layout.addPlot(title="Front body joints (°)", row=1, col=0)
    plot_FrontBody.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_FrontBody.showGrid(x=True, y=True)

    curve_FrontBody_right_state = plot_FrontBody.plot(pen=(0,120,191), name='Right (state)')
    curve_FrontBody_left_state = plot_FrontBody.plot(pen=(255,127,39), name='Left (state)')

    curve_FrontBody_right_action = plot_FrontBody.plot(pen=pg.mkPen(color=(1,188,239), style=QtCore.Qt.DotLine), name='Right (action)')
    curve_FrontBody_left_action = plot_FrontBody.plot(pen=pg.mkPen(color=(255,89,143), style=QtCore.Qt.DotLine), name='Left (action)')


    ####Back body joints plot
    plot_BackBody = grid_layout.addPlot(title="Back body joints (°)", row=1, col=1)
    plot_BackBody.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_BackBody.showGrid(x=True, y=True)

    curve_BackBody_right_state = plot_BackBody.plot(pen=(252,198,4), name='Right (state)')
    curve_BackBody_left_state = plot_BackBody.plot(pen=(116,80,167), name='Left (state)')

    curve_BackBody_right_action = plot_BackBody.plot(pen=pg.mkPen(color=(255,129,6), style=QtCore.Qt.DotLine), name='Right (action)')
    curve_BackBody_left_action = plot_BackBody.plot(pen=pg.mkPen(color=(1,188,239), style=QtCore.Qt.DotLine), name='Left (action)')


    ####Front leg joints plot
    plot_FrontLeg = grid_layout.addPlot(title="Front leg joints (°)", row=1, col=2)
    plot_FrontLeg.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_FrontLeg.showGrid(x=True, y=True)

    curve_FrontLeg_right_state = plot_FrontLeg.plot(pen=(55,194,50), name='Right (state)')
    curve_FrontLeg_left_state = plot_FrontLeg.plot(pen=(254,107,177), name='Left (state)')

    curve_FrontLeg_right_action = plot_FrontLeg.plot(pen=pg.mkPen(color=(247,237,0), style=QtCore.Qt.DotLine), name='Right (action)')
    curve_FrontLeg_left_action = plot_FrontLeg.plot(pen=pg.mkPen(color=(186,24,248), style=QtCore.Qt.DotLine), name='Left (action)')


    ####Back leg joints plot
    plot_BackLeg = grid_layout.addPlot(title="Back leg joints (°)", row=1, col=3)
    plot_BackLeg.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_BackLeg.showGrid(x=True, y=True)

    curve_BackLeg_right_state = plot_BackLeg.plot(pen=(0,120,191), name='Right (state)')
    curve_BackLeg_left_state = plot_BackLeg.plot(pen=(255,127,39), name='Left (state)')

    curve_BackLeg_right_action = plot_BackLeg.plot(pen=pg.mkPen(color=(1,188,239), style=QtCore.Qt.DotLine), name='Right (action)')
    curve_BackLeg_left_action = plot_BackLeg.plot(pen=pg.mkPen(color=(255,89,143), style=QtCore.Qt.DotLine), name='Left (action)')


    ####Front paw joints plot
    plot_FrontPaw = grid_layout.addPlot(title="Front paw joints (°)", row=1, col=4)
    plot_FrontPaw.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_FrontPaw.showGrid(x=True, y=True)

    curve_FrontPaw_right_state = plot_FrontPaw.plot(pen=(252,198,4), name='Right (state)')
    curve_FrontPaw_left_state = plot_FrontPaw.plot(pen=(116,80,167), name='Left (state)')

    curve_FrontPaw_right_action = plot_FrontPaw.plot(pen=pg.mkPen(color=(255,129,6), style=QtCore.Qt.DotLine), name='Right (action)')
    curve_FrontPaw_left_action = plot_FrontPaw.plot(pen=pg.mkPen(color=(1,188,239), style=QtCore.Qt.DotLine), name='Left (action)')


    ####Back paw joints plot
    plot_BackPaw = grid_layout.addPlot(title="Back paw joints (°)", row=1, col=5)
    plot_BackPaw.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_BackPaw.showGrid(x=True, y=True)

    curve_BackPaw_right_state = plot_BackPaw.plot(pen=(55,194,50), name='Right (state)')
    curve_BackPaw_left_state = plot_BackPaw.plot(pen=(254,107,177), name='Left (state)')

    curve_BackPaw_right_action = plot_BackPaw.plot(pen=pg.mkPen(color=(247,237,0), style=QtCore.Qt.DotLine), name='Right (action)')
    curve_BackPaw_left_action = plot_BackPaw.plot(pen=pg.mkPen(color=(186,24,248), style=QtCore.Qt.DotLine), name='Left (action)')


    ####Returns plot
    plot_Returns = grid_layout.addPlot(title="Real Return vs Predicted Return", row=2, col=0)
    plot_Returns.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Returns.showGrid(x=True, y=True)

    curve_Real_Return = plot_Returns.plot(pen=(255,0,0), name='Real')
    curve_Predicted_Return = plot_Returns.plot(pen=(0,255,0), name='Predicted')


    ####ReturnError plot
    plot_Return_Error = grid_layout.addPlot(title="RMSD of Real and Predicted Return", row=2, col=1)
    plot_Return_Error.showGrid(x=True, y=True)

    curve_Return_Error = plot_Return_Error.plot(pen=(182,102,247))


    ####Qloss plot
    plot_Q_Loss = grid_layout.addPlot(title="State-Action Value Loss", row=2, col=2)
    plot_Q_Loss.showGrid(x=True, y=True)

    curve_Q_Loss = plot_Q_Loss.plot(pen=(0,255,0))


    ####Ploss plot
    plot_P_Loss = grid_layout.addPlot(title="Policy Loss", row=2, col=3)
    plot_P_Loss.showGrid(x=True, y=True)

    curve_P_Loss = plot_P_Loss.plot(pen=(0,128,255))


    ####Alpha plot
    plot_Alpha = grid_layout.addPlot(title="Alpha (Entropy Regularization Coefficient)", row=2, col=4)
    plot_Alpha.showGrid(x=True, y=True)

    curve_Alpha = plot_Alpha.plot(pen=(255,150,45))


    ####Entropy and Standard deviation plot
    plot_Entropy_Std = grid_layout.addPlot(title="Policy's Entropy and Standard deviation", row=2, col=5)
    plot_Entropy_Std.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Entropy_Std.showGrid(x=True, y=True)

    curve_Entropy = plot_Entropy_Std.plot(pen=(0,255,255), name='Entropy')
    curve_Std = plot_Entropy_Std.plot(pen=(255,0,0), name='Std')

####################################################################################################################
    
    #Force Grid minimum size
    grid_layout.ci.layout.setColumnMinimumWidth(0,300)
    grid_layout.ci.layout.setColumnMinimumWidth(1,300)
    grid_layout.ci.layout.setColumnMinimumWidth(2,300)
    grid_layout.ci.layout.setColumnMinimumWidth(3,300)
    grid_layout.ci.layout.setColumnMinimumWidth(4,300)
    grid_layout.ci.layout.setColumnMinimumWidth(5,300)
    grid_layout.ci.layout.setRowMinimumHeight(0,335)
    grid_layout.ci.layout.setRowMinimumHeight(1,335)
    grid_layout.ci.layout.setRowMinimumHeight(2,335)
    grid_layout.ci.layout.setHorizontalSpacing(10)
    grid_layout.ci.layout.setVerticalSpacing(0)

    #Timer to update plots every 1 second (if there is new data) in another thread
    timer = QtCore.QTimer()
    timer.timeout.connect(updatePlot)
    timer.start(1000)
    
    grid_layout.showMaximized()

    status = app.exec_()
    sys.exit(status)
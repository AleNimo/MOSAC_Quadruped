import numpy as np
import torch
from SAC import SAC_Agent
from TrainHistory import TrainHistory

import gymnasium as gym
import mo_gymnasium as mo_gym

import multiprocessing
import queue
import pyqtgraph as pg

import sys
import time

from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5 import QtCore

import atexit   #For saving before termination

data_type = np.float64

# Handler to run if the python process is terminated with keyboard interrupt, or if the socket doesn't respond (simulator or the real robot)
# IMPORTANT:   -If you want to intentionally interrupt the python process, do so only when the agent is interacting with the environment and not when it is learning
#               otherwise the networks will be saved in an incomplete state.
#              -If intentionally interrupted, the script takes time to save the training history, so don't close the console window right away
def exit_handler(agent, train_history):
    train_history.global_step -= 1
    agent.save_models(train_history.global_step)
    agent.replay_buffer.save(train_history.global_step)

    train_history.save()


def MOSAC_Agent_Training(q):

    env = mo_gym.make("mo-halfcheetah-v4", render_mode="human")

    episode_steps = 1000    #Default limit of MuJoCo mo_gym testbenchs

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    rwd_dim = env.unwrapped.reward_space.shape[0]

    #env = Environment(sim_measurement=7, obs_dim=17, act_dim=12, rwd_dim=6)

    #The 24 values from coppelia, in order:
        #---- The first 7 not seen by the agent but used in the reward (simulation only measurements)--------
        #position (x,y,z)
        #Mean forward and lateral velocities with the reference frame of the agent
        #Maximum absolute forward acceleration measured during the previous step with the reference frame of the agent
        #yaw of the agent's body
        
        #---- The last 17 seen by the agent in the state vector (observation dimension)------------------------------
        #target step rotation of the agent's body
        #pitch and roll of the agent's body
        #pitch and roll angular velocities of the agent's body
        #the 12 joint angles

    load_agent = True
    test_agent = False
    load_replay_buffer_and_history = True   #if test_agent == True, only the train history is loaded (the replay_buffer is not used)
    
    training_steps = 1000000
    save_period = 10000

    #Preference vector maximum and minimum values - [vel_forward, cost_action]
    #If pref_min_vector == pref_max_vector then the multi-objective approach is disabled, and constant reward weights 
    #equal to pref_max_vector are defined
    
    pref_max_vector = np.ones(rwd_dim)
    pref_min_vector = np.zeros(rwd_dim)

    agent = SAC_Agent('MOSAC_HalfCheetah_learn_each_step', obs_dim, act_dim, rwd_dim, pref_max_vector, pref_min_vector, replay_buffer_size=1000000)
    
    agent.replay_batch_size = 1000

    agent.update_Q = 1  # The Q function is updated every episode
    agent.update_P = 1  # The policy is updated every 1 episode

    if load_agent:
        agent.load_models()
 
    ep_hidden_obs = np.zeros(episode_steps, dtype=data_type)                # Episode's hidden observed states 
    ep_obs = np.zeros((episode_steps+1, obs_dim), dtype=data_type)          # Episode's observed states
    ep_act = np.zeros((episode_steps, act_dim), dtype=data_type)            # Episode's actions
    ep_rwd = np.zeros((episode_steps, rwd_dim), dtype=data_type)            # Episode's rewards
    returns = np.zeros((int(training_steps/episode_steps),3), dtype=data_type)
    train_history = TrainHistory(max_steps=training_steps)

    if load_replay_buffer_and_history:
        train_history.load()

        if test_agent == False:
            agent.replay_buffer.load(train_history.global_step)

    # Set the exit_handler only when training
    if test_agent == False: atexit.register(exit_handler, agent, train_history)

    start_time = time.time() - train_history.ep_time[train_history.global_step-1]*3600

    episode = 0

    while train_history.global_step <= training_steps:

        print("Episode ", episode)

        ep_obs[0], info = env.reset()

        ep_hidden_obs[0] = 0

        done_flag = False

        step = 0

        #Generate random preference for the episode
        pref = np.random.random_sample((1,rwd_dim)) * (pref_max_vector-pref_min_vector) + pref_min_vector
        print("Preference vector: ", pref)

        while not done_flag:
            # Decide action based on present observed state (random action with mean and std)
            ep_act[step] = agent.choose_action(ep_obs[step], pref)

            # Act in the environment
            ep_obs[step+1], ep_rwd[step], terminated, truncated, info = env.step(ep_act[step])

            ep_hidden_obs[step] = info['x_position']

            done_flag = terminated or truncated
            
            # Store in replay buffer
            agent.remember(ep_obs[step], ep_act[step], ep_rwd[step], ep_obs[step+1], done_flag)

            # Train networks
            agent.learn(step)

            # Save stats
            train_history.ep_time[train_history.global_step] = (time.time() - start_time)/3600 #hours since start of training
            train_history.ep_loss[train_history.global_step, 0] = agent.Q_loss.item()
            train_history.ep_loss[train_history.global_step, 1] = agent.P_loss.item()
            train_history.ep_alpha[train_history.global_step] = agent.log_alpha.exp().item()
            train_history.ep_entropy[train_history.global_step] = agent.entropy.item()

            step = step + 1
            train_history.global_step = train_history.global_step + 1

            # End episode on termination condition
            if done_flag: break

        ep_len = step

        # Compute total reward from partial rewards and preference of the episode
        tot_rwd = np.sum(pref * ep_rwd, axis=1)

        ######## Compute the real and expected returns, and the root mean square error ##########
        ### Real return (undiscounted for testing, discounted for training):
        # Auxiliary array for computing return without overwriting tot_rwd
        aux_ret = np.copy(tot_rwd)
        pref_tensor = torch.tensor(pref, dtype=torch.float64).to(agent.P_net.device)

        # If the episode ended because the agent reached the maximum steps allowed, the rest of the return is estimated with the Q function
        # Using the last state, and last action that the policy would have chosen in that state
        if not done_flag:
            last_state = torch.tensor(np.expand_dims(ep_obs[step], axis=0), dtype=torch.float64).to(agent.P_net.device)
            last_action = agent.choose_action(ep_obs[step], pref, random=not(test_agent), tensor=True)
            if test_agent:
                aux_ret[step] += agent.minimal_Q(last_state, last_action, pref_tensor).detach().cpu().numpy().reshape(-1)
            else:
                aux_ret[step] += agent.discount_factor * agent.minimal_Q(last_state, last_action, pref_tensor).detach().cpu().numpy().reshape(-1)

        if test_agent:
            for i in range(ep_len-2, -1, -1): aux_ret[i] = aux_ret[i] + aux_ret[i+1]
        else:
            for i in range(ep_len-2, -1, -1): aux_ret[i] = aux_ret[i] + agent.discount_factor * aux_ret[i+1]
        returns[episode, 0] = aux_ret[0]
        
        if test_agent:  #when testing only compute the real return
            print("Episode's undiscounted return: ", returns[0])

            # Send the information for plotting in the other process through a Queue
            q.put(( test_agent, ep_hidden_obs[0:ep_len+1], ep_obs[0:ep_len+1], tot_rwd[0:ep_len+1], ep_rwd[0:ep_len+1], ep_act[0:ep_len+1] ))

        else:
            print("Global steps: ", train_history.global_step)

            #### Expected return at the start of the episode:
            initial_state = torch.tensor(np.expand_dims(ep_obs[0], axis=0), dtype=torch.float64).to(agent.P_net.device)
            initial_action = torch.tensor(np.expand_dims(ep_act[0], axis=0), dtype=torch.float64).to(agent.P_net.device)
            returns[episode, 1] = agent.minimal_Q(initial_state, initial_action, pref_tensor).detach().cpu().numpy().reshape(-1)

            #### Root mean square error
            returns[episode, 2] = np.sqrt(np.square(returns[episode, 0] - returns[episode, 1]))
            
            
            print("Real return = ", returns[episode, 0])
            print("Expected return = ", returns[episode, 1])
            print("RMSE = ", returns[episode, 2])

            # Send the information for plotting in the other process through a Queue
            q.put(( test_agent, ep_hidden_obs[0:ep_len+1], ep_obs[0:ep_len+1], tot_rwd[0:ep_len+1], ep_rwd[0:ep_len+1],  ep_act[0:ep_len+1], \
                    episode, train_history.global_step, returns[0:episode+1], train_history.ep_loss[0:train_history.global_step], \
                    train_history.ep_alpha[0:train_history.global_step], train_history.ep_entropy[0:train_history.global_step], train_history.ep_time[0:train_history.global_step]))
        
            # Save the progress every save_period episodes, unless its being tested
            if train_history.global_step % save_period == 0:
                agent.save_models(train_history.global_step)
                agent.replay_buffer.save(train_history.global_step)
                train_history.save()
        print("------------------------------------------")
        episode += 1

def updatePlot():   
    global q, curve_Position, curve_Velocity, curve_ActionCost, curve_Reward, curve_Forward_Vel_rwd, curve_Action_Cost_rwd, \
        curve_Real_Return, curve_Predicted_Return, curve_Return_Error, curve_P_Loss, curve_Q_Loss, curve_Alpha, curve_Entropy
        
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()

        test_agent = results[0]

        ####Position update
        x_pos = results[1]      #For each step of the episode

        state_linspace = np.arange(0,len(x_pos), 1, dtype=int)

        curve_Position.setData(state_linspace, x_pos)

        ####Velocity update
        x_velocity = results[2][:,8]      #For each step of the episode

        state_linspace = np.arange(0,len(x_velocity), 1, dtype=int)

        curve_Velocity.setData(state_linspace, x_velocity)

        ####Action Cost update
        action_cost = np.sum(np.square(results[5]), axis=1)
        
        action_linspace = np.arange(1,len(action_cost)+1,1, dtype=int)    #Because no action has been made in the step 0

        curve_ActionCost.setData(action_linspace, action_cost)

        ####Rewards update
        total_rwd = results[3]
        forward_velocity_rwd = results[4][:,0]
        action_cost_rwd = results[4][:,1]

        rwd_linspace = np.arange(1,len(total_rwd)+1, 1, dtype=int)   #Because there is no reward in the first state (step 0)

        curve_Reward.setData(rwd_linspace, total_rwd)
        curve_Forward_Vel_rwd.setData(rwd_linspace, forward_velocity_rwd)
        curve_Action_Cost_rwd.setData(rwd_linspace, action_cost_rwd)

        if test_agent == False:
            last_episode = results[6]
            last_step = results[7]

            episode_linspace = np.arange(0,last_episode+1,1,dtype=int)
            global_steps_linspace = np.arange(0,last_step,1,dtype=int)

            time = results[12]

            ####Returns update
            Real_Return_data = results[8][:,0]
            Predicted_Return_data = results[8][:,1]
            Return_loss_data = results[8][:,2]

            curve_Real_Return.setData(episode_linspace, Real_Return_data)
            curve_Predicted_Return.setData(episode_linspace, Predicted_Return_data)
            curve_Return_Error.setData(episode_linspace, Return_loss_data)

            ####Qloss update
            Q_loss_data = results[9][:,0]

            curve_Q_Loss.setData(time, Q_loss_data)

            ####Ploss update
            P_loss_data = results[9][:,1]

            curve_P_Loss.setData(time, P_loss_data)

            ####Alpha update
            Alpha_data = results[10]
            curve_Alpha.setData(global_steps_linspace,Alpha_data)
            
            ####Entropy update
            Entropy_data = results[11]

            curve_Entropy.setData(global_steps_linspace, Entropy_data)

    except queue.Empty:
        #print("Empty Queue")
        pass

if __name__ == '__main__':
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    app = QApplication(sys.argv)

    #Create a queue to share data between processes
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    MOSAC_process=multiprocessing.Process(None,MOSAC_Agent_Training,args=(q,))
    MOSAC_process.start()

    # Create window
    grid_layout = pg.GraphicsLayoutWidget(title="HalfCheetah - Training information")
    grid_layout.resize(1500,800)
    
    pg.setConfigOptions(antialias=True)

############################################### PLOTS #####################################################################

    #### X position
    plot_Position = grid_layout.addPlot(title="X Position (m)", row=0, col=0)
    plot_Position.showGrid(x=True, y=True)

    curve_Position = plot_Position.plot(pen=(0,255,0))

    #### X Velocity
    plot_Velocity = grid_layout.addPlot(title="X Velocity (m/s)", row=0, col=1)
    plot_Velocity.showGrid(x=True, y=True)

    curve_Velocity = plot_Velocity.plot(pen=(255,127,39))


    ####Action cost plot
    plot_ActionCost = grid_layout.addPlot(title="Cost of the action (sum(action_vector^2))", row=0, col=2)
    plot_ActionCost.showGrid(x=True, y=True)

    curve_ActionCost = plot_ActionCost.plot(pen=(255,0,0))

    ####Rewards plot
    plot_Rewards = grid_layout.addPlot(title="Total and individual Rewards", row=0, col=3)
    plot_Rewards.addLegend(offset=(1, 1), verSpacing=-1, horSpacing = 20, labelTextSize = '7pt', colCount=3)
    plot_Rewards.showGrid(x=True, y=True)

    curve_Reward = plot_Rewards.plot(pen=(255,255,0), name='Total')
    curve_Forward_Vel_rwd = plot_Rewards.plot(pen=(0,255,0), name='Fwd_Vel')
    curve_Action_Cost_rwd = plot_Rewards.plot(pen=(255,0,0), name='Action_Cost')

    ####Returns plot
    plot_Returns = grid_layout.addPlot(title="Real Return vs Predicted Return", row=1, col=0)
    plot_Returns.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Returns.showGrid(x=True, y=True)

    curve_Real_Return = plot_Returns.plot(pen=(255,0,0), name='Real')
    curve_Predicted_Return = plot_Returns.plot(pen=(0,255,0), name='Predicted')
    curve_Return_Error = plot_Returns.plot(pen=(182,102,247), name='RMSD')

    ####Qloss plot
    plot_Q_Loss = grid_layout.addPlot(title="State-Action Value Loss", row=1, col=1)
    plot_Q_Loss.showGrid(x=True, y=True)

    curve_Q_Loss = plot_Q_Loss.plot(pen=(0,255,0))


    ####Ploss plot
    plot_P_Loss = grid_layout.addPlot(title="Policy Loss", row=1, col=2)
    plot_P_Loss.showGrid(x=True, y=True)

    curve_P_Loss = plot_P_Loss.plot(pen=(0,128,255))


    ####Entropy and Alpha plot
    plot_Entropy = grid_layout.addPlot(title="Policy's Entropy and Alpha", row=1, col=3)
    plot_Entropy.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Entropy.showGrid(x=True, y=True)

    curve_Alpha = plot_Entropy.plot(pen=(255,150,45), name='Alpha')
    curve_Entropy = plot_Entropy.plot(pen=(0,255,255), name='Entropy')

####################################################################################################################
    
    #Force Grid minimum size
    grid_layout.ci.layout.setColumnMinimumWidth(0,300)
    grid_layout.ci.layout.setColumnMinimumWidth(1,300)
    grid_layout.ci.layout.setColumnMinimumWidth(2,300)
    grid_layout.ci.layout.setColumnMinimumWidth(3,300)
    grid_layout.ci.layout.setRowMinimumHeight(0,315)
    grid_layout.ci.layout.setRowMinimumHeight(1,315)
    grid_layout.ci.layout.setHorizontalSpacing(5)
    grid_layout.ci.layout.setVerticalSpacing(0)

    #Timer to update plots every 1 second (if there is new data) in another thread
    timer = QtCore.QTimer()
    timer.timeout.connect(updatePlot)
    timer.start(1000)
    
    
    grid_layout.show()

    status = app.exec_()
    sys.exit(status)
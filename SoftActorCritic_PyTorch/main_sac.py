import numpy as np
import torch
from SAC import SAC_Agent
from Environment import Environment
from TrainHistory import TrainHistory

import multiprocessing
import queue
import pyqtgraph as pg

import sys

from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5 import QtCore

import atexit  # For saving before termination

data_type = np.float64

# Handler to run if the python process is terminated with keyboard interrupt, or if the socket doesn't respond (simulator or the real robot)
# IMPORTANT:   -If you want to intentionally interrupt the python process, do so only when the agent is interacting with the environment and not when it is learning
#               otherwise the networks will be saved in an incomplete state.
#              -If intentionally interrupted, the script takes time to save the training history, so don't close the console window right away


def exit_handler(agent, train_history):
    train_history.episode -= 1
    agent.save_models(train_history.episode)
    agent.replay_buffer.save(train_history.episode)

    train_history.save()

def testing_step(agent, pref):

    bod_llim, bod_ulim = -10.0, 15.0
    femur_llim, femur_ulim = -30.0, 30.0
    tibia_llim, tibia_ulim = -15.0, 15.0

    jointLowerLimit = np.array(
        [
            bod_llim,
            femur_llim,
            tibia_llim,
            bod_llim,
            femur_llim,
            tibia_llim,
            bod_llim,
            femur_llim,
            tibia_llim,
            bod_llim,
            femur_llim,
            tibia_llim,
        ]
    )
    jointUpperLimit = np.array(
        [
            bod_ulim,
            femur_ulim,
            tibia_ulim,
            bod_ulim,
            femur_ulim,
            tibia_ulim,
            bod_ulim,
            femur_ulim,
            tibia_ulim,
            bod_ulim,
            femur_ulim,
            tibia_ulim,
        ]
    )
    ep_act_denorm = np.zeros(12, dtype = data_type)

    #obs_hardcodeada = np.array([0,0,0,0,0,-0.30073669, -0.03526281, -0.08253123, -0.36763916, 0.06415126, 0.07235107, -0.05228882, -0.03702011, -0.1058787, -0.15217163, 0.060333, 0.07094371], dtype=np.float32)
    obs_hardcodeada = np.array([0,0,0,0,0,-0.28216614, -0.03775711, -0.10876465, -0.36136108,  0.06939341,  0.09264933, -0.06454285, -0.03170166, -0.1058787,  -0.10889832,  0.04998245,  0.07620646])
    #obs_hardcodeada = np.array([0,0,0,0,0,-0.75872253, 0.39220428, 0.9054479, 0.48749878, -0.19866409, -0.01899109, -0.33689026,-0.54882406, -0.88124644,  0.61813904, -0.01768926,  0.31552531])
    #obs_hardcodeada += np.random.rand(17) * 0.001
    # print("Estado = ", ep_obs[step][:env.obs_dim])
    # print("estado = ", obs_hardcodeada)

    action_policy = agent.choose_action(obs_hardcodeada.astype(np.float64), pref, random=False).reshape(-1)

    # print("action = ", ep_act[step])
    hardcoded_act = np.array([  0.31852595,  18.84316773, 12.84012393,   0.09273345,  -2.75779056,  -7.41834583,   2.98036139,  -5.60446045, -12.9553208,   5.24906797,  12.5214766,   11.09053167])
    #hardcoded_act = np.array( [-8.58855113, 9.94102821, 12.98046072,  9.20397931,  -6.65862796, 0.81378085, -3.59237606, -18.35150799, -11.72735273, 11.37932153, -2.1422362, 2.91314069])
    #hardcoded_act = np.array( [-0.16368698, 0.66670318, 0.93517696, -0.21142376, -0.16529772, -0.68989114, 0.07944237, -0.13915044, -0.92500561, 0.21374062, 0.51791875, 0.78412176])
    for i in range(12):
        ep_act_denorm[i] = ((jointUpperLimit[i]-jointLowerLimit[i])/2.0) * action_policy[i] + (jointUpperLimit[i]+jointLowerLimit[i])/2.0
        #hardcoded_act[i] = hardcoded_act[i] * (jointUpperLimit[i]-jointLowerLimit[i])/2.0 + (jointUpperLimit[i]+jointLowerLimit[i])/2.0

    print("difference policy - hardcoded act = ", np.max(np.abs(ep_act_denorm-hardcoded_act)))
    # print("action_denorm = ", ep_act_denorm)

def SAC_Agent_Training(q):
    global jointLowerLimit, jointUpperLimit

    env = Environment(sim_measurement=8, obs_dim=17, act_dim=12, rwd_dim=7)

    # The 24 values from coppelia, in order:

    # ---- The first 17 seen by the agent in the state vector (observation dimension)------------------------------
    # target step rotation of the agent's body
    # pitch and roll of the agent's body
    # pitch and roll angular velocities of the agent's body
    # the 12 joint angles
    

    # ---- The last 8 not seen by the agent but used in the rewards and/or plots (simulation only measurements)--------
    # position (x,y,z)
    # Mean forward and lateral velocities with the reference frame of the agent
    # Maximum absolute forward acceleration measured during the previous step with the reference frame of the agent
    # yaw of the agent's body
    # step ommited flag

    load_agent = False
    test_agent = False
    load_replay_buffer_and_history = False # if test_agent == True, only the train history is loaded (the replay_buffer is not used)

    episodes = 20000
    episode_steps = 200  # Maximum steps allowed per episode
    save_period = 100

    # Preference vector maximum and minimum values - [vel_forward, acceleration, vel_lateral, orientation, flat_back]
    # If pref_min_vector == pref_max_vector then the multi-objective approach is disabled, and constant reward weights
    # equal to pref_max_vector are defined
    pref_max_vector = np.array([1, 1, 1, 1, 1])
    pref_min_vector = np.array([0.5, 0, 0, 0, 0])
    # pref_max_vector = np.array([2, 1, 1, 1, 1, 1])
    # pref_min_vector = np.array([0.5, 0, 0, 0, 0, 0.5])

    pref_dim = pref_max_vector.size

    agent = SAC_Agent('Cuadruped', env, pref_max_vector,pref_min_vector, replay_buffer_size=1000000)

    agent.replay_batch_size = 10000

    agent.update_Q = 1  # The Q function is updated every episode
    agent.update_P = 1  # The policy is updated every 1 episode

    if load_agent:
        agent.load_models()

    ep_obs = np.zeros((episode_steps+1, env.obs_dim+env.sim_measurement),dtype=data_type)        # Episode's observed states
    ep_act = np.zeros((episode_steps, env.act_dim),dtype=data_type)            # Episode's actions
    ep_rwd = np.zeros((episode_steps, env.rwd_dim),dtype=data_type)            # Episode's rewards
    train_history = TrainHistory(max_episodes=episodes)

    if load_replay_buffer_and_history:
        train_history.load()

        if test_agent == False:
            agent.replay_buffer.load(train_history.episode)

        train_history.episode = train_history.episode + 1

    # Set the exit_handler only when training
    if test_agent == False:
        atexit.register(exit_handler, agent, train_history)

    while train_history.episode <= episodes:

        print("Episode: ", train_history.episode, flush=True)

        ep_obs[0], done_flag = env.reset(), False
        

        # Testing
        if test_agent:
            # Use the user input preference for the test: [vel_forward, acceleration, vel_lateral, orientation, flat_back]
            pref = np.array([[1, 1, 1, 1, 1]])
            print("Preference vector: ", pref, flush=True)
            for step in range(episode_steps):

                # testing_step(agent, pref)

                # Decide action based on present observed state (taking only the mean)
                ep_act[step] = agent.choose_action(ep_obs[step][:env.obs_dim], pref, random=False)

                # print("action = ", ep_act[step])
                # The agent doesn't receive the position and target direction although it is on the ep_obs vector for plotting reasons

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                if done_flag:
                    break

            ep_len = step + 1

        # Training
        else:
            # Generate random preference for the episode
            pref = np.random.random_sample((1, pref_dim)) * (pref_max_vector-pref_min_vector) + pref_min_vector
            print("Preference vector: ", pref, flush=True)
            for step in range(episode_steps):
                # Decide action based on present observed state (random action with mean and std)
                ep_act[step] = agent.choose_action(ep_obs[step][:env.obs_dim], pref)

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                # Store in replay buffer
                agent.remember(ep_obs[step][:env.obs_dim], ep_act[step],ep_rwd[step], ep_obs[step+1][:env.obs_dim], done_flag)

                # End episode on termination condition
                if done_flag:
                    break

            ep_len = step + 1

        # Compute total reward from partial rewards and preference of the episode
        tot_rwd = np.sum(pref * ep_rwd[:, :-2], axis=1) + ep_rwd[:, -2] + ep_rwd[:, -1]

        ######## Compute the real and expected returns, and the root mean square error ##########
        # Real return (undiscounted for testing, discounted for training):
        # Auxiliary array for computing return without overwriting tot_rwd
        aux_ret = np.copy(tot_rwd)
        pref_tensor = torch.tensor(pref, dtype=torch.float64).to(agent.P_net.device)

        # If the episode ended because the agent reached the maximum steps allowed, the rest of the return is estimated with the Q function
        # Using the last state, and last action that the policy would have chosen in that state
        # if not done_flag:
        #     last_state = torch.tensor(np.expand_dims(ep_obs[step+1][env.sim_measurement:], axis=0), dtype=torch.float64).to(agent.P_net.device)
        #     last_action = agent.choose_action(ep_obs[step+1][env.sim_measurement:], pref, random=not(test_agent), tensor=True)
        #     if test_agent:
        #         aux_ret[step] += agent.minimal_Q(last_state, last_action, pref_tensor).detach().cpu().numpy().reshape(-1)
        #     else:
        #         aux_ret[step] += agent.discount_factor * agent.minimal_Q(last_state, last_action, pref_tensor).detach().cpu().numpy().reshape(-1)

        if test_agent:
            for i in range(ep_len-2, -1, -1):
                aux_ret[i] = aux_ret[i] + aux_ret[i+1]
        else:
            for i in range(ep_len-2, -1, -1):
                aux_ret[i] = aux_ret[i] + agent.discount_factor * aux_ret[i+1]
        train_history.ep_ret[train_history.episode, 0] = aux_ret[0]

        if test_agent:  # when testing only compute the real return
            print("Episode's undiscounted return: ", aux_ret[0], flush=True)

            # Send the information for plotting in the other process through a Queue
            q.put((test_agent, ep_obs[0:ep_len+1], tot_rwd[0:ep_len+1],ep_rwd[0:ep_len+1],  ep_act[0:ep_len+1]))

        else:
            # Expected return at the start of the episode:
            initial_state = torch.tensor(np.expand_dims(ep_obs[0][:env.obs_dim], axis=0), dtype=torch.float64).to(agent.P_net.device)
            initial_action = torch.tensor(np.expand_dims(ep_act[0], axis=0), dtype=torch.float64).to(agent.P_net.device)
            train_history.ep_ret[train_history.episode, 1] = agent.minimal_Q(initial_state, initial_action, pref_tensor).detach().cpu().numpy().reshape(-1)

            # Root mean square error
            train_history.ep_ret[train_history.episode, 2] = np.sqrt(np.square(train_history.ep_ret[train_history.episode, 0] - train_history.ep_ret[train_history.episode, 1]))

            ######### Train the agent with batch_size samples for every step made in the episode ##########
            for step in range(ep_len):
                agent.learn(step)

            # Store the results of the episodes for plotting and printing on the console
            train_history.ep_loss[train_history.episode,0] = agent.Q_loss.item()
            train_history.ep_loss[train_history.episode,1] = agent.P_loss.item()
            train_history.ep_alpha[train_history.episode] = agent.log_alpha.exp().item()
            train_history.ep_entropy[train_history.episode] = agent.entropy.item()
            train_history.ep_std[train_history.episode] = agent.std.item()

            print("Replay_Buffer_counter: ", agent.replay_buffer.mem_counter, flush=True)
            print("Q_loss: ", train_history.ep_loss[train_history.episode, 0], flush=True)
            print("P_loss: ", train_history.ep_loss[train_history.episode, 1], flush=True)
            print("Alpha: ", train_history.ep_alpha[train_history.episode], flush=True)
            print("Policy's Entropy: ",
                  train_history.ep_entropy[train_history.episode], flush=True)

            # Send the information for plotting in the other process through a Queue
            q.put((test_agent, ep_obs[0:ep_len+1], tot_rwd[0:ep_len+1], ep_rwd[0:ep_len+1],  ep_act[0:ep_len+1],train_history.episode, train_history.ep_ret[0:train_history.episode +
                                                               1], train_history.ep_loss[0:train_history.episode+1],train_history.ep_alpha[0:train_history.episode+1], train_history.ep_entropy[0:train_history.episode+1], train_history.ep_std[0:train_history.episode+1]))

            # Save the progress every save_period episodes, unless its being tested
            if train_history.episode % save_period == 0 or train_history.episode == 50:
                agent.save_models(train_history.episode)
                agent.replay_buffer.save(train_history.episode)
                train_history.save()
        print("------------------------------------------")
        train_history.episode += 1


# Range of the joints for plotting the denormalized joint angles
body_min, body_max = -10.0, 15.0
body_mean = (body_min + body_max)/2
body_range = (body_max - body_min)/2

leg_min, leg_max = -30.0, 30.0
leg_mean = (leg_min + leg_max)/2
leg_range = (leg_max - leg_min)/2

paw_min, paw_max = -20.0,  20.0
paw_mean = (paw_min + paw_max)/2
paw_range = (paw_max - paw_min)/2


def updatePlot():
    global q, curve_Trajectory, curve_Trajectory_startPoint, curve_ForwardVelocity, curve_LateralVelocity, curve_ForwardAcc, curve_Pitch, \
        curve_Roll, curve_TargetRotation, curve_AgentRotation, curve_Reward, curve_Forward_vel_rwd, curve_Lateral_vel_rwd, curve_Orientation_rwd, curve_Back_rwd, \
        curve_Acc_rwd,curve_Ommited_rwd,curve_Not_Flipping_rwd, curve_P_Loss, curve_Q_Loss, curve_Real_Return, curve_Predicted_Return, curve_Return_Error, curve_Alpha, curve_Entropy, curve_Std, \
        curve_FrontBody_right_state, curve_FrontBody_left_state, curve_FrontBody_right_action, curve_FrontBody_left_action, \
        curve_BackBody_right_state, curve_BackBody_left_state, curve_BackBody_right_action, curve_BackBody_left_action, \
        curve_FrontLeg_right_state, curve_FrontLeg_left_state, curve_FrontLeg_right_action, curve_FrontLeg_left_action, \
        curve_BackLeg_right_state, curve_BackLeg_left_state, curve_BackLeg_right_action, curve_BackLeg_left_action, \
        curve_FrontPaw_right_state, curve_FrontPaw_left_state, curve_FrontPaw_right_action, curve_FrontPaw_left_action, \
        curve_BackPaw_right_state, curve_BackPaw_left_state, curve_BackPaw_right_action, curve_BackPaw_left_action, \
        body_joints_state, body_joints_action, leg_joints_state, leg_joints_action, paw_joints_state, paw_joints_action
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()), flush=True)
    try:
        results = q.get_nowait()

        test_agent = results[0]

        # Trajectory update
        Trajectory_x_data = results[1][:, 17 + 0]
        Trajectory_y_data = results[1][:, 17 + 1]

        curve_Trajectory.setData(Trajectory_x_data, Trajectory_y_data)
        curve_Trajectory_startPoint.setData([Trajectory_x_data[0]], [Trajectory_y_data[0]])

        # Velocities update
        # For each step of the episode
        forward_velocity = results[1][:, 17 + 3]
        lateral_velocity = results[1][:, 17 + 4]

        state_linspace = np.arange(0, len(forward_velocity), 1, dtype=int)
        next_state_linspace = np.arange(1, len(forward_velocity), 1, dtype=int)

        curve_ForwardVelocity.setData(state_linspace, forward_velocity)
        curve_LateralVelocity.setData(state_linspace, lateral_velocity)

        # Acceleration update
        forward_acc = results[1][:, 17 + 5]

        curve_ForwardAcc.setData(state_linspace, forward_acc)

        # Inclination update
        pitch = results[1][:, 1] * 180  # deg
        roll = results[1][:, 2] * 180  # deg

        curve_Pitch.setData(state_linspace, pitch)
        curve_Roll.setData(state_linspace, roll)

        # Rotation update
        next_angle = results[1][1:, 17 + 6] * 180/np.pi
        previous_angle = results[1][:-1, 17 + 6] * 180/np.pi
        agents_rotation = (next_angle - previous_angle)
        # Correcting any posible discontinuities:
        # Boolean mask to detect if there are discontinuities (change in sign with great values, +180 <-> -180, ignoring -5 <-> +5 for example):
        peak_mask = ((next_angle * previous_angle) <0) & (np.abs(next_angle) > 100)

        # Boolean mask to detect if the rotation should be positive (next angle becomes negative if it increments beyond the limit):
        pos_mask = (next_angle < 0) & peak_mask

        # Boolean mask to detect if the rotation should be negative (next angle becomes positive if it decrements beyond the limit):
        neg_mask = (next_angle > 0) & peak_mask

        # Apply corrections:
        agents_rotation = agents_rotation + 360*pos_mask - 360*neg_mask

        target_rotation = results[1][:, 0] * 180  # deg

        curve_TargetRotation.setData(state_linspace, target_rotation)
        curve_AgentRotation.setData(next_state_linspace, agents_rotation)

        # Rewards update
        total_rwd = results[2]
        forward_velocity_rwd = results[3][:, 0]
        acc_rwd = results[3][:, 1]
        lateral_velocity_rwd = results[3][:, 2]
        orientation_rwd = results[3][:, 3]
        back_rwd = results[3][:, 4]
        step_ommited_rwd = results[3][:, 5]
        not_flipping_rwd = results[3][:, 6]

        # Because there is no reward in the first state (step 0)
        rwd_linspace = np.arange(1, len(total_rwd)+1, 1, dtype=int)

        curve_Reward.setData(rwd_linspace, total_rwd)
        curve_Forward_vel_rwd.setData(rwd_linspace, forward_velocity_rwd)
        curve_Lateral_vel_rwd.setData(rwd_linspace, lateral_velocity_rwd)
        curve_Orientation_rwd.setData(rwd_linspace, orientation_rwd)
        curve_Back_rwd.setData(rwd_linspace, back_rwd)
        curve_Acc_rwd.setData(rwd_linspace, acc_rwd)
        curve_Ommited_rwd.setData(rwd_linspace, step_ommited_rwd)
        curve_Not_Flipping_rwd.setData(rwd_linspace, not_flipping_rwd)

        # Body joints update
        body_joints_state = []
        body_joints_action = []
        for i in range(4):
            body_joints_state.append(results[1][:, 5+i*3] * body_range + body_mean)
            body_joints_action.append(results[4][:, i*3] * body_range + body_mean)

        # Because no action has been made in the step 0
        action_linspace = np.arange(1, len(body_joints_action[0])+1, 1, dtype=int)

        curve_FrontBody_right_state.setData(state_linspace, body_joints_state[0])
        curve_FrontBody_left_state.setData(state_linspace, body_joints_state[1])
        curve_FrontBody_right_action.setData(action_linspace, body_joints_action[0])
        curve_FrontBody_left_action.setData(action_linspace, body_joints_action[1])

        curve_BackBody_right_state.setData(state_linspace, body_joints_state[2])
        curve_BackBody_left_state.setData(state_linspace, body_joints_state[3])
        curve_BackBody_right_action.setData(action_linspace, body_joints_action[2])
        curve_BackBody_left_action.setData(action_linspace, body_joints_action[3])

        # Leg joints update
        leg_joints_state = []
        leg_joints_action = []
        for i in range(4):
            leg_joints_state.append(results[1][:, 6+i*3] * leg_range + leg_mean)
            leg_joints_action.append(results[4][:, 1+i*3] * leg_range + leg_mean)

        curve_FrontLeg_right_state.setData(state_linspace, leg_joints_state[0])
        curve_FrontLeg_left_state.setData(state_linspace, leg_joints_state[1])
        curve_FrontLeg_right_action.setData(action_linspace, leg_joints_action[0])
        curve_FrontLeg_left_action.setData(action_linspace, leg_joints_action[1])

        curve_BackLeg_right_state.setData(state_linspace, leg_joints_state[2])
        curve_BackLeg_left_state.setData(state_linspace, leg_joints_state[3])
        curve_BackLeg_right_action.setData(action_linspace, leg_joints_action[2])
        curve_BackLeg_left_action.setData(action_linspace, leg_joints_action[3])

        # Paw joints update
        paw_joints_state = []
        paw_joints_action = []
        for i in range(4):
            paw_joints_state.append(results[1][:, 7+i*3] * paw_range + paw_mean)
            paw_joints_action.append(results[4][:, 2+i*3] * paw_range + paw_mean)

        curve_FrontPaw_right_state.setData(state_linspace, paw_joints_state[0])
        curve_FrontPaw_left_state.setData(state_linspace, paw_joints_state[1])
        curve_FrontPaw_right_action.setData(action_linspace, paw_joints_action[0])
        curve_FrontPaw_left_action.setData(action_linspace, paw_joints_action[1])

        curve_BackPaw_right_state.setData(state_linspace, paw_joints_state[2])
        curve_BackPaw_left_state.setData(state_linspace, paw_joints_state[3])
        curve_BackPaw_right_action.setData(action_linspace, paw_joints_action[2])
        curve_BackPaw_left_action.setData(action_linspace, paw_joints_action[3])

        if test_agent == False:
            last_episode = results[5]
            episode_linspace = np.arange(0, last_episode+1, 1, dtype=int)

            # Returns update
            Real_Return_data = results[6][:, 0]
            Predicted_Return_data = results[6][:, 1]

            curve_Real_Return.setData(episode_linspace, Real_Return_data)
            curve_Predicted_Return.setData(episode_linspace, Predicted_Return_data)

            # Returns error update
            Return_loss_data = results[6][:, 2]

            curve_Return_Error.setData(episode_linspace, Return_loss_data)

            # Qloss update
            Q_loss_data = results[7][:, 0]

            curve_Q_Loss.setData(episode_linspace, Q_loss_data)

            # Ploss update
            P_loss_data = results[7][:, 1]

            curve_P_Loss.setData(episode_linspace, P_loss_data)

            # Alpha update
            Alpha_data = results[8]
            curve_Alpha.setData(episode_linspace, Alpha_data)

            # Entropy update
            Entropy_data = results[9]

            curve_Entropy.setData(episode_linspace, Entropy_data)

            # Standard deviation update
            Std_data = results[10]

            curve_Std.setData(episode_linspace, Std_data)

    except queue.Empty:
        # print("Empty Queue", flush=True)
        pass


if __name__ == '__main__':
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()), flush=True)
    app = QApplication(sys.argv)

    # Create a queue to share data between processes
    q = multiprocessing.Queue()

    # Create and start the SAC_Agent_Training process
    SAC_process = multiprocessing.Process(None, SAC_Agent_Training, args=(q,))
    SAC_process.start()

    # Create window

    grid_layout = pg.GraphicsLayoutWidget(title="Cuadruped - Training information")
    grid_layout.resize(1200, 800)

    pg.setConfigOptions(antialias=True)

############################################### PLOTS #####################################################################

    # Trajectory plot
    plot_Trajectory = grid_layout.addPlot(title="Trajectory", row=0, col=0)
    plot_Trajectory.setAspectLocked()

    # Point in the center of the scene
    plot_Trajectory.plot([0], [0], pen=None, symbol='o', symbolPen=None,symbolSize=5, symbolBrush=(255, 255, 255, 200))

    # Square to delimitate floor of the scene
    scene_limit = QGraphicsRectItem(-12.5, -12.5, 25, 25)
    scene_limit.setPen(pg.mkPen((255, 255, 255, 100), width=2))
    scene_limit.setBrush(pg.mkBrush(None))

    # Circule to delimitate the end condition of the episode (the goal to reach)
    end_limit = QGraphicsEllipseItem(-11, -11, 22, 22)
    end_limit.setPen(pg.mkPen((0, 255, 0, 100), width=1,style=QtCore.Qt.DashLine))  # type: ignore
    end_limit.setBrush(pg.mkBrush(None))

    plot_Trajectory.addItem(scene_limit)
    plot_Trajectory.addItem(end_limit)

    plot_Trajectory.setRange(xRange=(-14, 14), yRange=(-14, 14),padding=None, update=True, disableAutoRange=True)

    # Curves to update them in updatePlot()
    curve_Trajectory = plot_Trajectory.plot()
    curve_Trajectory_startPoint = plot_Trajectory.plot(pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 255, 0, 150))

    # Velocity plot
    plot_Velocity = grid_layout.addPlot(title="Mean forward and lateral velocity (m/s)", row=0, col=1)
    plot_Velocity.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Velocity.showGrid(x=True, y=True)

    curve_ForwardVelocity = plot_Velocity.plot(pen=(0, 255, 0), name='Forward')
    curve_LateralVelocity = plot_Velocity.plot(pen=(255, 0, 0), name='Lateral')

    # Acceleration plot
    plot_Acceleration = grid_layout.addPlot(title="Peak absolute forward acceleration (m/s^2)", row=0, col=2)
    plot_Acceleration.showGrid(x=True, y=True)
    curve_ForwardAcc = plot_Acceleration.plot(pen=(0, 255, 0))

    # Inclination plot
    plot_Inclination = grid_layout.addPlot(title="Pitch and Roll (°)", row=0, col=3)
    plot_Inclination.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Inclination.showGrid(x=True, y=True)

    curve_Pitch = plot_Inclination.plot(pen=(0, 255, 255), name='Pitch')
    curve_Roll = plot_Inclination.plot(pen=(255, 0, 255), name='Roll')

    # Orientation plot
    plot_Orientation = grid_layout.addPlot(title="Target step rotation vs Agent's step rotation (°)", row=0, col=4)
    plot_Orientation.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Orientation.showGrid(x=True, y=True)

    curve_TargetRotation = plot_Orientation.plot(pen=(255, 201, 14), name='Target')
    curve_AgentRotation = plot_Orientation.plot(pen=(255, 127, 39), name='Agent')

    # Rewards plot
    plot_Rewards = grid_layout.addPlot(title="Total and individual Rewards", row=0, col=5)
    plot_Rewards.addLegend(offset=(1, 1), verSpacing=-1,horSpacing=20, labelTextSize='7pt', colCount=3)
    plot_Rewards.showGrid(x=True, y=True)

    curve_Reward = plot_Rewards.plot(pen=(255, 255, 0), name='Total')
    curve_Forward_vel_rwd = plot_Rewards.plot(pen=(0, 255, 0), name='Fwd_Vel')
    curve_Lateral_vel_rwd = plot_Rewards.plot(pen=(255, 0, 0), name='Lat_Vel')
    curve_Orientation_rwd = plot_Rewards.plot(pen=(255, 127, 39), name='Rotation')
    curve_Back_rwd = plot_Rewards.plot(pen=(0, 255, 255), name='Tilt')
    curve_Acc_rwd = plot_Rewards.plot(pen=(255, 255, 255), name='Fwd_Acc')
    curve_Ommited_rwd = plot_Rewards.plot(pen=(100, 255, 255), name='Step Ommited')
    curve_Not_Flipping_rwd = plot_Rewards.plot(pen=(255, 100, 255), name='Not flipping')

    # Front body joints plot
    plot_FrontBody = grid_layout.addPlot(title="Front body joints (°)", row=1, col=0)
    plot_FrontBody.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_FrontBody.showGrid(x=True, y=True)

    curve_FrontBody_right_state = plot_FrontBody.plot(pen=(0, 120, 191), name='Right (state)')
    curve_FrontBody_left_state = plot_FrontBody.plot(pen=(255, 127, 39), name='Left (state)')

    curve_FrontBody_right_action = plot_FrontBody.plot(pen=pg.mkPen(color=(1, 188, 239), style=QtCore.Qt.DotLine), name='Right (action)')  # type: ignore
    curve_FrontBody_left_action = plot_FrontBody.plot(pen=pg.mkPen(color=(255, 89, 143), style=QtCore.Qt.DotLine), name='Left (action)')  # type: ignore

    # Back body joints plot
    plot_BackBody = grid_layout.addPlot(title="Back body joints (°)", row=1, col=1)
    plot_BackBody.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_BackBody.showGrid(x=True, y=True)

    curve_BackBody_right_state = plot_BackBody.plot(pen=(252, 198, 4), name='Right (state)')
    curve_BackBody_left_state = plot_BackBody.plot(pen=(116, 80, 167), name='Left (state)')

    curve_BackBody_right_action = plot_BackBody.plot(pen=pg.mkPen(color=(255, 129, 6), style=QtCore.Qt.DotLine), name='Right (action)')  # type: ignore
    curve_BackBody_left_action = plot_BackBody.plot(pen=pg.mkPen(color=(1, 188, 239), style=QtCore.Qt.DotLine), name='Left (action)')  # type: ignore

    # Front leg joints plot
    plot_FrontLeg = grid_layout.addPlot(title="Front leg joints (°)", row=1, col=2)
    plot_FrontLeg.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_FrontLeg.showGrid(x=True, y=True)

    curve_FrontLeg_right_state = plot_FrontLeg.plot(pen=(55, 194, 50), name='Right (state)')
    curve_FrontLeg_left_state = plot_FrontLeg.plot(pen=(254, 107, 177), name='Left (state)')

    curve_FrontLeg_right_action = plot_FrontLeg.plot(pen=pg.mkPen(color=(247, 237, 0), style=QtCore.Qt.DotLine), name='Right (action)')  # type: ignore
    curve_FrontLeg_left_action = plot_FrontLeg.plot(pen=pg.mkPen(color=(186, 24, 248), style=QtCore.Qt.DotLine), name='Left (action)')  # type: ignore

    # Back leg joints plot
    plot_BackLeg = grid_layout.addPlot(title="Back leg joints (°)", row=1, col=3)
    plot_BackLeg.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_BackLeg.showGrid(x=True, y=True)

    curve_BackLeg_right_state = plot_BackLeg.plot(pen=(0, 120, 191), name='Right (state)')
    curve_BackLeg_left_state = plot_BackLeg.plot(pen=(255, 127, 39), name='Left (state)')

    curve_BackLeg_right_action = plot_BackLeg.plot(pen=pg.mkPen(color=(1, 188, 239), style=QtCore.Qt.DotLine), name='Right (action)')  # type: ignore
    curve_BackLeg_left_action = plot_BackLeg.plot(pen=pg.mkPen(color=(255, 89, 143), style=QtCore.Qt.DotLine), name='Left (action)')  # type: ignore

    # Front paw joints plot
    plot_FrontPaw = grid_layout.addPlot(title="Front paw joints (°)", row=1, col=4)
    plot_FrontPaw.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_FrontPaw.showGrid(x=True, y=True)

    curve_FrontPaw_right_state = plot_FrontPaw.plot(pen=(252, 198, 4), name='Right (state)')
    curve_FrontPaw_left_state = plot_FrontPaw.plot(pen=(116, 80, 167), name='Left (state)')

    curve_FrontPaw_right_action = plot_FrontPaw.plot(pen=pg.mkPen(color=(255, 129, 6), style=QtCore.Qt.DotLine), name='Right (action)')  # type: ignore
    curve_FrontPaw_left_action = plot_FrontPaw.plot(pen=pg.mkPen(color=(1, 188, 239), style=QtCore.Qt.DotLine), name='Left (action)')  # type: ignore

    # Back paw joints plot
    plot_BackPaw = grid_layout.addPlot(title="Back paw joints (°)", row=1, col=5)
    plot_BackPaw.addLegend(offset=(1, 1), verSpacing=-1, colCount=2)
    plot_BackPaw.showGrid(x=True, y=True)

    curve_BackPaw_right_state = plot_BackPaw.plot(pen=(55, 194, 50), name='Right (state)')
    curve_BackPaw_left_state = plot_BackPaw.plot(pen=(254, 107, 177), name='Left (state)')

    curve_BackPaw_right_action = plot_BackPaw.plot(pen=pg.mkPen(color=(247, 237, 0), style=QtCore.Qt.DotLine), name='Right (action)')  # type: ignore
    curve_BackPaw_left_action = plot_BackPaw.plot(pen=pg.mkPen(color=(186, 24, 248), style=QtCore.Qt.DotLine), name='Left (action)')  # type: ignore

    # Returns plot
    plot_Returns = grid_layout.addPlot(title="Real Return vs Predicted Return", row=2, col=0)
    plot_Returns.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Returns.showGrid(x=True, y=True)

    curve_Real_Return = plot_Returns.plot(pen=(255, 0, 0), name='Real')
    curve_Predicted_Return = plot_Returns.plot(pen=(0, 255, 0), name='Predicted')

    # ReturnError plot
    plot_Return_Error = grid_layout.addPlot(title="RMSD of Real and Predicted Return", row=2, col=1)
    plot_Return_Error.showGrid(x=True, y=True)

    curve_Return_Error = plot_Return_Error.plot(pen=(182, 102, 247))

    # Qloss plot
    plot_Q_Loss = grid_layout.addPlot(title="State-Action Value Loss", row=2, col=2)
    plot_Q_Loss.showGrid(x=True, y=True)

    curve_Q_Loss = plot_Q_Loss.plot(pen=(0, 255, 0))

    # Ploss plot
    plot_P_Loss = grid_layout.addPlot(title="Policy Loss", row=2, col=3)
    plot_P_Loss.showGrid(x=True, y=True)

    curve_P_Loss = plot_P_Loss.plot(pen=(0, 128, 255))

    # Alpha plot
    plot_Alpha = grid_layout.addPlot(title="Alpha (Entropy Regularization Coefficient)", row=2, col=4)
    plot_Alpha.showGrid(x=True, y=True)

    curve_Alpha = plot_Alpha.plot(pen=(255, 150, 45))

    # Entropy and Standard deviation plot
    plot_Entropy_Std = grid_layout.addPlot(title="Policy's Entropy and Standard deviation", row=2, col=5)
    plot_Entropy_Std.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Entropy_Std.showGrid(x=True, y=True)

    curve_Entropy = plot_Entropy_Std.plot(pen=(0, 255, 255), name='Entropy')
    curve_Std = plot_Entropy_Std.plot(pen=(255, 0, 0), name='Std')

####################################################################################################################

    # Force Grid minimum size
    grid_layout.ci.layout.setColumnMinimumWidth(0, 300)
    grid_layout.ci.layout.setColumnMinimumWidth(1, 300)
    grid_layout.ci.layout.setColumnMinimumWidth(2, 300)
    grid_layout.ci.layout.setColumnMinimumWidth(3, 300)
    grid_layout.ci.layout.setColumnMinimumWidth(4, 300)
    grid_layout.ci.layout.setColumnMinimumWidth(5, 300)
    grid_layout.ci.layout.setRowMinimumHeight(0, 315)
    grid_layout.ci.layout.setRowMinimumHeight(1, 315)
    grid_layout.ci.layout.setRowMinimumHeight(2, 315)
    grid_layout.ci.layout.setHorizontalSpacing(5)
    grid_layout.ci.layout.setVerticalSpacing(0)

    # Timer to update plots every 1 second (if there is new data) in another thread
    timer = QtCore.QTimer()
    timer.timeout.connect(updatePlot)
    timer.start(1000)

    grid_layout.show()

    status = app.exec_()
    sys.exit(status)

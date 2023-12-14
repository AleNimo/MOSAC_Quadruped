import os
import numpy as np
import torch
from SAC import SAC_Agent
from EnvironmentTetrapod import Environment

import multiprocessing
import queue
import threading
import pyqtgraph as pg
from dvg_pyqtgraph_threadsafe import PlotCurve
import matplotlib.pyplot as plt
import sys

from PyQt5.QtWidgets import QApplication, QGridLayout, QGraphicsEllipseItem

from PyQt5 import QtCore

from threading import Timer

import pyqtgraph.opengl as gl

data_type = np.float64

obs_dim = 19
act_dim = 12

# Variables for 3D plotting (shared between processes)
# Create mesh-grid
axis_1D = np.linspace(-2, 2, num=30)
mesh_x, mesh_y = np.meshgrid(axis_1D, axis_1D)
axis_2D = np.concatenate((np.array(mesh_x).reshape(-1,1), np.array(mesh_y).reshape(-1,1)), axis=1)

# Define obs and act variables to compute Q later
# obs = initial state (joints in zero) in every posible position with 30x30 points of resolution
# act = no initial action (all zeros in every state evaluated)
obs_plot = np.zeros((len(axis_2D),)+(obs_dim,), dtype=data_type)
obs_plot[:, 0:2] = axis_2D[:, 0:2]
act_plot = np.zeros((len(axis_2D),)+(12,), dtype=data_type)

cmap = plt.get_cmap('jet')

def SAC_Agent_Training(q):
    global obs_plot, act_plot
    env = Environment(obs_sp_shape=(obs_dim,), act_sp_shape=(act_dim,), dest_pos=(0,0))

    load_agent = True
    load_train_history = True
    load_replay_buffer = True   #(if load_train_history == false, the replay buffer is never loaded)
    
    episodes = 20000
    episode = 0
    episode_steps = 200 #Maximum steps allowed per episode
    save_period = 1000

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
    ep_alpha = np.zeros((episodes,), dtype=data_type)                # Alpha for each episode

    obs_plot_tensor = torch.tensor(obs_plot, dtype=torch.float64).to(agent.P_net.device)
    act_plot_tensor = torch.tensor(act_plot, dtype=torch.float64).to(agent.P_net.device)

    if load_train_history:
        # Check the last episode saved in Progress.txt
        if not os.path.isfile('./Train/Progress.txt'):
            print('Progress.txt could not be found')
            exit
        with open('./Train/Progress.txt', 'r') as file: episode = int(np.loadtxt(file))

        filename = './Train/Train_History_episode_{0:07d}.npz'.format(episode)
        loaded_arrays = np.load(filename)

        ep_ret[0:episode] = loaded_arrays['returns']
        ep_loss[0:episode] = loaded_arrays['loss']
        ep_alpha[0:episode] = loaded_arrays['alpha'][0:episode]

        if load_replay_buffer:
            agent.replay_buffer.load(episode)
        
        episode += 1

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
            agent.learn()
        
        ep_loss[episode, 0] = agent.P_loss.item()
        ep_loss[episode, 1] = agent.Q_loss.item()
        ep_alpha[episode] = agent.alpha.item()
        
        print("Episode: ", episode)
        print("Q_loss: ", ep_loss[episode, 0])
        print("P_loss: ", ep_loss[episode, 1])
        print("Alpha: ", ep_alpha[episode])

        # Q_evaluated = agent.minimal_Q(obs_plot_tensor, act_plot_tensor).numpy(force=True).reshape((30,30))

        q.put((episode, ep_ret, ep_loss, ep_alpha, ep_obs[0:ep_len+1, 0], ep_obs[0:ep_len+1, 1]))#, Q_evaluated))
        
        if episode % save_period == 0:
            agent.save_models()
            agent.replay_buffer.save(episode)
            
            filename = './Train/Train_History_episode_{0:07d}'.format(episode)
            np.savez_compressed(filename, returns = ep_ret[0:episode], loss = ep_loss[0:episode], alpha = ep_alpha[0:episode])
        
        episode += 1

class RepeatTimer(Timer):  
    def run(self):  
        while not self.finished.wait(self.interval):  
            self.function(*self.args,**self.kwargs)
        
def emit_signal(q):
    global signalComm
    signalComm.request_graph_update.emit(q)

def updateplot(q):   
    global curve_Trajectory, curve_P_Loss,curve_Q_Loss,curve_Real_Return, curve_Predicted_Return,curve_Return_Error,curve_Alpha
    #print('Thread ={}          Function = updateplot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()
        last_episode = results[0]
        episode_linspace = np.arange(0,last_episode+1,1,dtype=int)
        Real_Return_data = results[1][:,0]
        Predicted_Return_data = results[1][:,1]
        Return_loss_data = results[1][:,2]
        P_loss_data = results[2][:,1]
        Q_loss_data = results[2][:,0]
        Alpha_data = results[3]
        Trajectory_x_data = results[4]
        Trajectory_y_data = results[5]

        # Q_evaluated = results[6]

        # minZ=np.min(Q_evaluated)
        # maxZ=np.max(Q_evaluated)
        # rgba_img = cmap((Q_evaluated - minZ)/(maxZ - minZ))

        curve_Trajectory.setData(Trajectory_x_data,Trajectory_y_data)
        curve_P_Loss.setData(episode_linspace,P_loss_data[0:last_episode+1])
        curve_Q_Loss.setData(episode_linspace,Q_loss_data[0:last_episode+1])
        curve_Real_Return.setData(episode_linspace,Real_Return_data[0:last_episode+1])
        curve_Predicted_Return.setData(episode_linspace, Predicted_Return_data[0:last_episode+1])
        curve_Return_Error.setData(episode_linspace,Return_loss_data[0:last_episode+1])
        curve_Alpha.setData(episode_linspace,Alpha_data[0:last_episode+1])

        curve_Trajectory.update()
        curve_P_Loss.update()
        curve_Q_Loss.update()
        curve_Real_Return.update()
        curve_Predicted_Return.update()
        curve_Return_Error.update()
        curve_Alpha.update()

    except queue.Empty:
        #print("Empty Queue")
        pass


            
if __name__ == '__main__':
    global curve_Trajectory, curve_P_Loss, curve_Q_Loss, curve_Returns, curve_Return_Error, curve_Alpha, surface_Q_function, signalComm
    print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    app = QApplication(sys.argv)

    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    SAC_process=multiprocessing.Process(None,SAC_Agent_Training,args=(q,))
    SAC_process.start()

    # Create window
    
    window = pg.GraphicsLayoutWidget(title="Cuadruped - Training information")
    window.resize(1500,800)
    
    pg.setConfigOptions(antialias=True)

    grid_layout = QGridLayout()

    window.setLayout(grid_layout)

    plot_Trajectory = pg.PlotWidget(title="Last Trajectory")
    plot_Trajectory.setXRange(-3,3)
    plot_Trajectory.setYRange(-3,3)
    grid_layout.addWidget(plot_Trajectory, 0, 0)
    
    plot_Q_Loss = pg.PlotWidget(title="State-Value Loss")
    plot_Q_Loss.showGrid(x=True, y=True)
    grid_layout.addWidget(plot_Q_Loss, 0, 1)
    
    plot_P_Loss = pg.PlotWidget(title="Policy Loss")
    plot_P_Loss.showGrid(x=True, y=True)
    grid_layout.addWidget(plot_P_Loss, 0, 2)

    plot_Returns = pg.PlotWidget(title="Real return vs Predicted return")
    plot_Returns.addLegend()
    plot_Returns.showGrid(x=True, y=True)
    grid_layout.addWidget(plot_Returns, 1, 0)
    
    plot_Return_Error = pg.PlotWidget(title="RMSD of Real and Estimated")
    plot_Return_Error.showGrid(x=True, y=True)
    grid_layout.addWidget(plot_Return_Error, 1, 1)
    
    plot_Alpha = pg.PlotWidget(title="Alpha")
    plot_Alpha.showGrid(x=True, y=True)
    grid_layout.addWidget(plot_Alpha, 1, 2)

    N = 11
    M = 11

    x = np.linspace(0, 10, N)
    y = np.linspace(0, 10, M)

    z = np.random.random((N, M))

    gl_view_widget = gl.GLViewWidget()
    surface_Q_function = gl.GLSurfacePlotItem(x=x, y=y, z=z, colors=None)
    gl_view_widget.addItem(surface_Q_function)
    grid_layout.addWidget(gl_view_widget, 0, 3, 1, 1)

    gl_view_widget.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    plot_Trajectory.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    plot_Q_Loss.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    plot_P_Loss.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    plot_Returns.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    plot_Return_Error.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    plot_Alpha.sizeHint = lambda: pg.QtCore.QSize(100, 100)
    gl_view_widget.setSizePolicy(plot_Trajectory.sizePolicy())

    # Add circule to delimitate scene of trajectory plot
    p_ellipse = QGraphicsEllipseItem(0, 0, 3, 3)  # x, y, width, height
    # p_ellipse.setPen(pg.mkPen((0, 0, 0, 100)))
    # p_ellipse.setBrush(pg.mkBrush((50, 50, 200)))
    plot_Trajectory.addItem(p_ellipse)

    #Curves using dvg_pyqtgraph_threadsafe to update them in another thread
    curve_Trajectory = PlotCurve(linked_curve=plot_Trajectory.plot())
    curve_P_Loss=PlotCurve(linked_curve=plot_P_Loss.plot())
    curve_Q_Loss=PlotCurve(linked_curve=plot_Q_Loss.plot())
    curve_Real_Return=PlotCurve(linked_curve=plot_Returns.plot(pen=(255,0,0), name='Real'))
    curve_Predicted_Return=PlotCurve(linked_curve=plot_Returns.plot(pen=(0,255,0), name='Predicted'))
    curve_Return_Error=PlotCurve(linked_curve=plot_Return_Error.plot())
    curve_Alpha=PlotCurve(linked_curve=plot_Alpha.plot())
    
    #Timer to update plots every 1 second (if there is new data) in another thread
    t = RepeatTimer(1, updateplot,(q,))
    t.start() 
    
    window.show()

    status = app.exec_()
    sys.exit(status)
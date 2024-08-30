import numpy as np
import torch
from SAC import SAC_Agent

import mo_gymnasium as mo_gym

import multiprocessing
import queue
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore
from PyQt5.QtGui import QFont

import sys

from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction

data_type = np.float64

plot_resolution = 20

x_axis_pref = np.linspace(start=0,stop=1,num=plot_resolution+1,endpoint=True)
y_axis_pref = np.linspace(start=0,stop=1,num=plot_resolution+1,endpoint=True)

def MORL_Agent_Test(q):

    env = mo_gym.make("mo-halfcheetah-v4", render_mode="human")

    episode_steps = 1000    #Default limit of MuJoCo mo_gym testbenchs
    gamma = 0.99

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    rwd_dim = env.unwrapped.reward_space.shape[0]

    pref_max_vector = np.ones(rwd_dim)
    pref_min_vector = np.zeros(rwd_dim)

    ret = np.zeros((plot_resolution+1,plot_resolution+1), dtype=data_type)

    ep_rwd = np.zeros((episode_steps, rwd_dim), dtype=data_type)            # Episode's rewards

    #Load Agent
    agent = SAC_Agent('MOSAC_HalfCheetah_learn_each_step', obs_dim, act_dim, rwd_dim, pref_max_vector, pref_min_vector, replay_buffer_size=1000000)
    agent.load_models()

    # agent = CAPQL(env, device = "cuda")
    # agent.load(path="weights/CAPQL steps=1000000.tar", load_replay_buffer=False)

    # agent = GPIPDContinuousAction(env, device = "cuda")
    # agent.load(path="weights/GPI-PD gpi-ls iter=10.tar", load_replay_buffer=False)
    
    # Measure discounted return in tests
    for x_index, x_pref in enumerate(x_axis_pref):
        for y_index, y_pref in enumerate(y_axis_pref):
            print(f"Testing with preferences [{x_pref}, {y_pref}]: ")
            # pref = np.array([x_pref, y_pref]) #MORL_baselines
            pref = np.array([[x_pref, y_pref]]) #MOSAC
            acum_ret = 0
            for run in range(1,6):

                obs, info = env.reset()
                done_flag = False
                step = 0

                while not done_flag:
                    # Decide action based on present observed state (taking only the mean)
                    # action = agent.eval(obs, pref) #MORL_baselines
                    action = agent.choose_action(obs, pref, random=False) #MOSAC
                    # Act in the environment
                    # obs, ep_rwd[step], terminated, truncated, info = env.step(action) #MORL_baselines
                    obs, ep_rwd[step], terminated, truncated, info = env.step(action[0]) #MOSAC

                    done_flag = terminated or truncated
                    
                    # End episode on termination condition
                    if done_flag: break

                    step = step + 1

                ep_len = step + 1

                # Compute total reward from partial rewards and preference of the episode
                tot_rwd = np.sum(pref * ep_rwd, axis=1)

                ######## Compute the real return ##########
                # Auxiliary array for computing return without overwriting tot_rwd
                aux_ret = np.copy(tot_rwd)

                for i in range(ep_len-2, -1, -1): aux_ret[i] = aux_ret[i] + gamma * aux_ret[i+1]
                
                acum_ret += aux_ret[0]

                print(f"Run N°{run} - Discounted return = {aux_ret[0]}")

                # Send the information for plotting in the other process through a Queue
                q.put((ret))

            ret[x_index, y_index] = acum_ret/run
            print("Average return to plot: ", ret[x_index, y_index])
    
    #Save the testing data (for 3D plot)
    filename = f"./Return_Surface_MOSAC_2"
    np.savez_compressed(filename, return_3D = ret)

def updatePlot():   
    global q, plot
        
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()

        plot.setData(x=x_axis_pref, y=y_axis_pref, z=results)

    except queue.Empty:
        #print("Empty Queue")
        pass

if __name__ == '__main__':
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    
    #Create a queue to share data between processes
    q = multiprocessing.Queue()

    #Create and start the MORL_Algorithm_Test process
    MORL_Alg_process=multiprocessing.Process(None,MORL_Agent_Test,args=(q,))
    MORL_Alg_process.start()
    
    ## Create a GL View widget to display data
    app = pg.mkQApp()
    w = gl.GLViewWidget()
    
    w.setWindowTitle('Real return as function of Preferences - MORL Agent')
    w.setCameraPosition(distance=5)
    w.resize(1000,500)
    w.setBackgroundColor('w')    #Fondo blanco para paper

    ## Add a grids to the view
    g_xy = gl.GLGridItem(color=(10,10,10,76.5))
    g_xz = gl.GLGridItem(color=(10,10,10,76.5))
    g_yz = gl.GLGridItem(color=(10,10,10,76.5))

    g_xy.setSize(1,1,1)
    g_xz.setSize(1,1,1)
    g_yz.setSize(1,1,1)

    g_xy.setSpacing(0.1,0.1,0.1)
    g_xz.setSpacing(0.1,0.1,0.1)
    g_yz.setSpacing(0.1,0.1,0.1)

    g_xy.translate(0.5, 0.5, 0)

    g_xz.rotate(90, 1, 0, 0)
    g_xz.translate(0.5, 0, 0.5)

    g_yz.rotate(90, 0, 1, 0)
    g_yz.translate(0, 0.5, 0.5)

    g_xy.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    g_xz.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    g_yz.setDepthValue(10)  # draw grid after surfaces since they may be translucent

    w.addItem(g_xy)
    w.addItem(g_xz)
    w.addItem(g_yz)

    ## Add labels to the view
    ticks_font = QFont("Times", 10) #, QFont.Bold
    x_label = gl.GLTextItem(pos=np.array([1.2,0,0]), color=[0,0,0], text='X')
    x_tick = gl.GLTextItem(pos=np.array([1.05,0,-0.1]), color=[0,0,0], text='1', font=ticks_font)

    y_label = gl.GLTextItem(pos=np.array([0,1.2,0]), color=[0,0,0], text='Y')
    y_tick = gl.GLTextItem(pos=np.array([0,1.05,-0.1]), color=[0,0,0], text='1', font=ticks_font)

    z_label = gl.GLTextItem(pos=np.array([0,0,1.2]), color=[0,0,0], text='Z')
    z_tick = gl.GLTextItem(pos=np.array([0,0,1.05]), color=[0,0,0], text='1000', font=ticks_font)

    w.addItem(x_label)
    w.addItem(y_label)
    w.addItem(z_label)

    w.addItem(x_tick)
    w.addItem(y_tick)
    w.addItem(z_tick)

    ## Add axis to the view
    axis = gl.GLAxisItem()
    axis.setSize(1.1,1.1,1.1)
    axis.setDepthValue(0)
    w.addItem(axis)

    ## Add empty surface plot
    plot = gl.GLSurfacePlotItem(x=x_axis_pref,y=y_axis_pref,z=np.zeros((plot_resolution+1,plot_resolution+1)), shader='shaded', color=(1, 0.5, 0, 1))
    plot.scale(1,1,0.001)
    plot.setDepthValue(1)
    w.addItem(plot)

    pg.setConfigOptions(antialias=True)
    w.show()

    #Timer to update plot every 1 second (if there is new data) in another thread
    timer = QtCore.QTimer()
    timer.timeout.connect(updatePlot)
    timer.start(1000)

    status = pg.exec()
    sys.exit(status)
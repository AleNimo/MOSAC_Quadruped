from Policy import P_Network
import numpy as np
import torch

#----Observation vector----#
#   target step rotation of the agent's body
#   pitch and roll of the agent's body
#   pitch and roll angular velocities of the agent's body
#   the 12 joint angles
obs_dim=17

#----Action vector----#
#   the 12 joint angles (pwm active time for each servo)
act_dim=12

#----Preference vector----#
#   [vel_forward, acceleration, vel_lateral, orientation, flat_back]
pref_dim = 5

if __name__ == '__main__':
    P_net = P_Network(obs_dim, act_dim, pref_dim, hidden1_dim=64, hidden2_dim=32)
    pref = torch.tensor([1,1,1,1,1]).to(P_net.device)
    
    #Receive servo angles and target rotation step from Nucleo-F412ZG with SPI1
    target_rot = SPI_RECEIVE
    servo_angles = SPI_RECEIVE

    #Receive IMU data from XIAO_NRF52840 with I2C
    pitch = I2C_RECEIVE
    roll
    vel_pitch
    vel_roll

    state = [target_rot, pitch, roll, vel_pitch, vel_roll, servo_angles]
    state = torch.tensor(state).to(P_net.device)

    actions, _ = P_net(state, pref)
    actions = torch.tanh(actions) #Restrict the actions to (-1;1)

    #Send Action with SPI
    #Pin Output High to signal the Nucleo that the action is computed
    #Pin Output Low again

    



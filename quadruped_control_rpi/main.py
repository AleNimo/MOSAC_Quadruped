######Pinout information (https://pinout.xyz/)###########
#SPI1 (supports 16 bit words):  MOSI (GPIO20, pin 38); MISO (GPIO19, pin 35); SCLK (GPIO21, pin 40)
#                               CE0 (GPIO18, pin 12), CE1 (GPIO17, pin 11), CE2 (GPIO16, pin 36)
#I2C1 (I2C0 is for the SD): SDA (GPIO2, pin 3); SCL (GPIO3, pin 5)
spi_slave_ready = 25 #GPIO25, pin 22

import pigpio #For GPIO, SPI and I2C
from Policy import P_Network
import numpy as np
import torch


pi = pigpio.pi()

##States##
state = "RX_SPI"

########SPI SETUP#######
rise_edge_detected = 0

def rising_edge_callback(gpio, level, tick):
    global rise_edge_detected
    if gpio == spi_slave_ready:
        rise_edge_detected = 1
        print("rise: ", tick)

pi.set_mode(spi_slave_ready, pigpio.INPUT)
pi.callback(spi_slave_ready, pigpio.RISING_EDGE, rising_edge_callback)

#spi_flags:
#21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
# b  b  b  b  b  b  R  T  n  n  n  n  W  A u2 u1 u0 p2 p1 p0  m  m

#bbbbbb defines the word size in bits (0-32)
#R and T = 0 means msb first for both MISO and MOSI
#W = 0 means SPI is not 3-wire (MISO and MOSI in different lines)
#A is 0 for the main SPI, 1 for the auxiliary SPI
#Ux = 1 means CEx disabled
#px sets if CEx is active low or high
#mm defines the mode

spi_flags = 0b0100000000000111100000
hspi = pi.spi_open(0,500000,spi_flags)    #Auxiliary SPI, 16 bits per word, msb first, mode 0, without CE, 1MHz

#######I2C SETUP########
IMU_address = 0x1
hi2c = pi.i2c_open(1, IMU_address, 0)   #No configuration posible for I2C (not needed)


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
#    [vel_forward, acceleration, vel_lateral, orientation, flat_back]
pref_dim = 5

first_time = 1

#----Joint Denormalization----#
body_min, body_max = -10.0, 15.0
body_mean = (body_min + body_max)/2
body_range = (body_max - body_min)/2

leg_min, leg_max = -30.0, 30.0
leg_mean = (leg_min + leg_max)/2
leg_range = (leg_max - leg_min)/2

paw_min, paw_max = -15.0,  15.0
paw_mean = (paw_min + paw_max)/2
paw_range = (paw_max - paw_min)/2

cant_steps = 0

if __name__ == '__main__':
    P_net = P_Network(obs_dim, act_dim, pref_dim, hidden1_dim=64, hidden2_dim=32)
    P_net.load_checkpoint()

    pref = torch.tensor([[1, 1, 1, 1, 1]]).to(P_net.device)
    
    while True:

        match state:
            case "RX_SPI":    ###Receive servo angles from Nucleo-F412ZG with SPI1
                # if first_time:
                #     # print("RX_SPI")
                #     first_time = 0
                if rise_edge_detected:
                    rise_edge_detected = 0
                    print("rise=0")
                    first_time = 1

                    (bytes_read, joints_byte_list) = pi.spi_read(hspi, 12*4)
                    if bytes_read == 12*4: print("Se leyó todo")
                    else: print("no se leyeron la cantidad de bytes correcta - Nucleo")

                    measurements_nucleo = np.frombuffer(bytes(joints_byte_list), dtype='<f4')   #< little endian, > big endian
                    print("measurements_nucleo = ", measurements_nucleo)
                    
                    #Change the order from webots to the order the NUCLEO needs based on servo conections to the timers:
                    measured_joints = np.zeros(12,dtype = np.float64)

                    # Body  Front   Right
                    measured_joints[0] = measurements_nucleo[0]
                    # Femur   Front   Right
                    measured_joints[1] = measurements_nucleo[4]
                    # Tibia   Front   Right
                    measured_joints[2] = measurements_nucleo[3]
                    # Body  Front   Left
                    measured_joints[3] = measurements_nucleo[5]
                    # Femur   Front   Left
                    measured_joints[4] = - measurements_nucleo[6]
                    # Tibia   Front   Left
                    measured_joints[5] = - measurements_nucleo[7]
                    # Body  Back    Right
                    measured_joints[6] = - measurements_nucleo[1]
                    # Femur   Back    Right
                    measured_joints[7] = measurements_nucleo[11]
                    # Tibia   Back    Right
                    measured_joints[8] = measurements_nucleo[10]
                    # Body  Back    Left
                    measured_joints[9] = - measurements_nucleo[2]
                    # Femur   Back    Left
                    measured_joints[10] = - measurements_nucleo[8]
                    # Tibia   Back    Left
                    measured_joints[11] = - measurements_nucleo[9]

                    #Normalization
                    for i in range(4):
                        measured_joints[i*3] =      (measured_joints[i*3]    - body_mean)   /   body_range
                        measured_joints[1+i*3] =    (measured_joints[1+i*3]  - leg_mean)    /   leg_range
                        measured_joints[2+i*3] =    (measured_joints[2+i*3]  - paw_mean)    /   paw_range
                    
                    print("measured_joints = ", measured_joints)
                    # print("measured_joints.dtype", measured_joints.dtype)

                    state = "RX_I2C"

            case "RX_I2C": ###Receive IMU data from XIAO_NRF52840 with I2C
                # print("RX_I2C")

                #Read 4*4 bytes (4 floats)
                (count, data) = pi.i2c_read_device(hi2c, 4*4)
                if count == 4*4: print("Se leyó todo")
                else: print("no se leyeron la cantidad de bytes correcta - IMU")
                #print("bytes IMU",data)

                #IMU_data[4] = [roll, pitch, wx, wy]
                IMU_data = np.frombuffer(bytes(data), dtype='<f4')
                print("IMU_data = ", IMU_data)

                state = "POLICY"

            case "POLICY":
                cant_steps += 1
                # Assemble the state vector from the measurements of the NUCLEO and IMU
                target_rot = 0
                pitch = IMU_data[0]
                roll = IMU_data[1]
                vel_pitch = IMU_data[2]
                vel_roll = IMU_data[3]
                
                state = np.array([target_rot, pitch/np.pi, roll/np.pi, vel_pitch/np.pi, vel_roll/np.pi])
                state = np.concatenate((state, measured_joints), axis=0)
                state = np.expand_dims(state, axis=0)
                print("state", state)
                print("Pitch:" , pitch*180/np.pi)
                print("Roll:" , roll*180/np.pi)

                state = torch.tensor(state).to(P_net.device)

                #Compute the action with the policy network
                action, _ = P_net(state, pref)
                action = torch.tanh(action) #Restrict the actions to (-1;1)
                action = action.detach().cpu().numpy().reshape(-1).astype(np.float64)

                #Denormalize action with the ranges and midpoints of the joints
                for i in range(4):
                    action[i*3] = action[i*3] * body_range + body_mean
                    action[1+i*3] = action[1+i*3] * leg_range + leg_mean
                    action[2+i*3] = action[2+i*3] * paw_range + paw_mean
                
                #Change the order from webots to the order the NUCLEO needs based on servo connections to the timers:
                #Mid point is added inside the NUCLEO
                action_nucleo = np.zeros(12, dtype=np.float32)
                # BFR
                action_nucleo[0] = action[0]
                # BBR
                action_nucleo[1] = -action[6]
                # BBL
                action_nucleo[2] = -action[9]
                # TFR
                action_nucleo[3] = action[2]
                # FFR
                action_nucleo[4] = action[1]
                # BFL
                action_nucleo[5] = action[3]
                # FFL
                action_nucleo[6] = -action[4]
                # TFL
                action_nucleo[7] = -action[5]
                # FBL
                action_nucleo[8] = -action[10]
                # TBL
                action_nucleo[9] = -action[11]
                # TBR
                action_nucleo[10] = action[8]
                # FBR
                action_nucleo[11] = action[7]

                #Order of nucleo servos:
                #0PWM_BFR,1PWM_BBR, 2PWM_BBL, 3PWM_TFR, 4PWM_FFR, 5PWM_BFL, 6PWM_FFL, 7PWM_TFL, 8PWM_FBL, 9PWM_TBL, 10PWM_TBR, 11PWM_FBR
                print("cant_steps = ", cant_steps)
                print("Action = ",          action)
                print("Action Nucleo = ", action_nucleo)
                # print("action.dtype", action.dtype)

                #Convert float numpy array in byte list
                action_byte_list = list(action_nucleo.tobytes())

                state = "TX_SPI"

            case "TX_SPI":
                # if first_time:
                #     # print("TX_SPI")
                #     first_time = 0
                #Send Action with SPI
                if rise_edge_detected:
                    print("Flanco ascendente detectado, envío acción")
                    rise_edge_detected = 0
                    first_time = 1

                    pi.spi_write(hspi, action_byte_list)

                    state = "RX_SPI"

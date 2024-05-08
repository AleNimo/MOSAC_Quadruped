#Pinout information#
#SPI: SPI0: MOSI (GPIO10, pin 19); MISO (GPIO9, pin 21); SCLK (GPIO11, pin 23); CE0 (GPIO8), CE1 (GPIO7)
#Serial: TX (GPIO14); RX (GPIO15)

import RPi.GPIO as GPIO
import spidev   #For SPI
from smbus2 import SMBus, i2c_msg   #For I2C

from Policy import P_Network
import numpy as np
import torch

########SPI SETUP#######
GPIO.setmode(GPIO.BOARD)

spi_slave_ready = 22 #GPIO25
GPIO.setup(spi_slave_ready, GPIO.IN)
GPIO.add_event_detect(spi_slave_ready, GPIO.RISING, callback=rising_edge_callback)

rise_edge_detected = 0
def rising_edge_callback(channel):
    if channel == spi_slave_ready:
        rise_edge_detected = 1

spi = spidev.SpiDev()
spi.open(0,0)

#SPI Settings
spi.bits_per_word = 16
spi.mode = 0b00 #CPOL CPHA
spi.max_speed_hz = 1000000
spi.no_cs(True)

#######I2C SETUP########
IMU_address = 0x1 #Obtained using i2cdetect -y 0 in console


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
    
    while True:
        ###Receive servo angles from Nucleo-F412ZG with SPI1
        GPIO.wait_for_edge(spi_slave_ready, GPIO.RISING)    #Wait for NUCLEO to 
        
        joints_byte_list = spi.readbytes(12*4)
        measured_joints = np.frombuffer(bytes(joints_byte_list), dtype='<f4')   #< little endian, > big endian

        #Receive target step rotation with bluetooth
        target_rot = BLACK MAGIC PERRA

        ###Receive IMU data from XIAO_NRF52840 with I2C
        with SMBus(1) as bus:
            #Read 4*4 bytes (4 floats)
            msg = i2c_msg.read(IMU_address, 4*4)    #Prepare I2C read
            bus.i2c_rdwr(msg)                       #Execute reading

        #IMU_data[4] = [roll, pitch, wx, wy]
        IMU_data = np.frombuffer(bytes(msg.buf), dtype='<f4')

        pitch = IMU_data[0]
        roll = IMU_data[1]
        vel_pitch = IMU_data[2]
        vel_roll = IMU_data[3]

        state = [target_rot, pitch, roll, vel_pitch, vel_roll, measured_joints]
        state = torch.tensor(state).to(P_net.device)

        actions, _ = P_net(state, pref)
        actions = torch.tanh(actions) #Restrict the actions to (-1;1)

        #Send Action with SPI
        action_byte_list = list(actions.detach().cpu().numpy().tobytes())
        if rise_edge_detected:
            spi.writebytes(action_byte_list)
        else:
            GPIO.wait_for_edge(spi_slave_ready, GPIO.RISING)
            rise_edge_detected = 0
            spi.writebytes(action_byte_list)
        

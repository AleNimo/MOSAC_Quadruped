"""extremidad_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Accelerometer
from controller import Supervisor
import numpy as np
from collections import deque
import socket

from scipy.spatial.transform import Rotation

import random


# slider = False
programmed_target_rotation = False

critical_failure_angle = 50 #Must match angle in environment.py

supervisor = Supervisor()
root = supervisor.getSelf()
robot_node = supervisor.getFromDef('MOSAC')

paw_FR = supervisor.getFromDef('PFR')
paw_FL = supervisor.getFromDef('PFL')
paw_BR = supervisor.getFromDef('PBR')
paw_BL = supervisor.getFromDef('PBL')

# Create Socket
HOST, PORT = "127.0.0.1", 57175

# Create the client and initial connection
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    client.connect((HOST, PORT))
    client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
except socket.error as err:
    print(f"Error: {err}")

# Define the lengths as in the Lua code
Rx_float_length = 10
Tx_float_length = "{:010.5f}"
Tx_Rx_command_length = 5

reset_pos = np.array([0.0, 0.0, 0.134728])  # Quadruped short leg (original)

reset_orientation = np.array([0.0, 0.0, 0.0])

def ypr_to_axis_angle(yaw, pitch, roll):
    # Create a rotation object from yaw, pitch, roll (in degrees)
    rotation = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    
    # Convert to axis-angle representation
    axis_angle = rotation.as_rotvec()  # Rotation vector (axis * angle)
    angle = np.linalg.norm(axis_angle)  # Angle is the norm of the rotation vector
    axis = axis_angle / angle if angle != 0 else [1, 0, 0]  # Normalize axis
    
    return [axis[0], axis[1], axis[2], angle]


robot_node.getField("translation").setSFVec3f([reset_pos[0],reset_pos[1],reset_pos[2]])
robot_node.getField("rotation").setSFRotation(ypr_to_axis_angle(reset_orientation[2],reset_orientation[1],reset_orientation[0]))

# get the time step of the current world.
timestep = 10

acc_std = 0.00001  # 0.002 #Gs
gyro_std = 0.00001  # 1.2 #°/s

# Get sensor readings
accelerometer = supervisor.getDevice("accelerometer")
gyroscope = supervisor.getDevice("gyro")
accelerometer.enable(timestep)
gyroscope.enable(timestep)

G = 9.80665

I = np.identity(4)

H = I

K = np.zeros((4, 4))
A = np.zeros((4, 4))

Q = I * np.power(gyro_std * np.pi / 180, 2)
R = I * np.power(acc_std * G, 2)
P = I * 0.1
X = np.array([[1], [0], [0], [0]])

Z = np.zeros((4, 1))

DCM = np.zeros((3, 3))


## FIR FILTER
coef_acc = np.ones(10) / 10  # Moving average FIR filters
coef_gyr = np.ones(3) / 3

ax_vector = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=10)
ay_vector = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=10)
az_vector = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=10)

wx_vector = deque([0, 0, 0], maxlen=3)
wy_vector = deque([0, 0, 0], maxlen=3)
wz_vector = deque([0, 0, 0], maxlen=3)

yaw = 0
pitch = 0
roll = 0

vel_pitch = 0
vel_roll = 0


FL_servo_tibia_joint = supervisor.getDevice("FL_servo_tibia_joint")
FL_servo_femur_joint = supervisor.getDevice("FL_servo_femur_joint")
FL_servo_body_joint = supervisor.getDevice("BFL_joint")

FR_servo_tibia_joint = supervisor.getDevice("FR_servo_tibia_joint")
FR_servo_femur_joint = supervisor.getDevice("FR_servo_femur_joint")
FR_servo_body_joint = supervisor.getDevice("BFR_joint")

BR_servo_tibia_joint = supervisor.getDevice("BR_servo_tibia_joint")
BR_servo_femur_joint = supervisor.getDevice("BR_servo_femur_joint")
BR_servo_body_joint = supervisor.getDevice("BBR_joint")

BL_servo_tibia_joint = supervisor.getDevice("BL_servo_tibia_joint")
BL_servo_femur_joint = supervisor.getDevice("BL_servo_femur_joint")
BL_servo_body_joint = supervisor.getDevice("BBL_joint")

FL_servo_tibia_joint_sensor = supervisor.getDevice("FL_servo_tibia_joint_sensor")
FL_servo_femur_joint_sensor = supervisor.getDevice("FL_servo_femur_joint_sensor")
FL_servo_body_joint_sensor = supervisor.getDevice("BFL_joint_sensor")

FR_servo_tibia_joint_sensor = supervisor.getDevice("FR_servo_tibia_joint_sensor")
FR_servo_femur_joint_sensor = supervisor.getDevice("FR_servo_femur_joint_sensor")
FR_servo_body_joint_sensor = supervisor.getDevice("BFR_joint_sensor")

BR_servo_tibia_joint_sensor = supervisor.getDevice("BR_servo_tibia_joint_sensor")
BR_servo_femur_joint_sensor = supervisor.getDevice("BR_servo_femur_joint_sensor")
BR_servo_body_joint_sensor = supervisor.getDevice("BBR_joint_sensor")

BL_servo_tibia_joint_sensor = supervisor.getDevice("BL_servo_tibia_joint_sensor")
BL_servo_femur_joint_sensor = supervisor.getDevice("BL_servo_femur_joint_sensor")
BL_servo_body_joint_sensor = supervisor.getDevice("BBL_joint_sensor")

FL_servo_tibia_joint_sensor.enable(timestep)
FL_servo_femur_joint_sensor.enable(timestep)
FL_servo_body_joint_sensor.enable(timestep)

FR_servo_tibia_joint_sensor.enable(timestep)
FR_servo_femur_joint_sensor.enable(timestep)
FR_servo_body_joint_sensor.enable(timestep)

BR_servo_tibia_joint_sensor.enable(timestep)
BR_servo_femur_joint_sensor.enable(timestep)
BR_servo_body_joint_sensor.enable(timestep)

BL_servo_tibia_joint_sensor.enable(timestep)
BL_servo_femur_joint_sensor.enable(timestep)
BL_servo_body_joint_sensor.enable(timestep)

FL_servo_tibia_joint.enableTorqueFeedback(timestep)
FL_servo_femur_joint.enableTorqueFeedback(timestep)
FL_servo_body_joint.enableTorqueFeedback(timestep)

FR_servo_tibia_joint.enableTorqueFeedback(timestep)
FR_servo_femur_joint.enableTorqueFeedback(timestep)
FR_servo_body_joint.enableTorqueFeedback(timestep)

BR_servo_tibia_joint.enableTorqueFeedback(timestep)
BR_servo_femur_joint.enableTorqueFeedback(timestep)
BR_servo_body_joint.enableTorqueFeedback(timestep)

BL_servo_tibia_joint.enableTorqueFeedback(timestep)
BL_servo_femur_joint.enableTorqueFeedback(timestep)
BL_servo_body_joint.enableTorqueFeedback(timestep)


bod_llim, bod_ulim = -15.0 * np.pi / 180.0, 15.0 * np.pi / 180.0
femur_llim, femur_ulim = -35.0 * np.pi / 180.0, 35.0 * np.pi / 180.0
tibia_llim, tibia_ulim = -15.0 * np.pi / 180.0, 15.0 * np.pi / 180.0

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


# Get agent handles
A2 = np.array([0.0, 0.0])
A_orth = np.array([0.0, 0.0])

joints_number = 12
leg_number = 4
joint = [
    FR_servo_body_joint,
    FR_servo_femur_joint,
    FR_servo_tibia_joint,
    FL_servo_body_joint,
    FL_servo_femur_joint,
    FL_servo_tibia_joint,
    BR_servo_body_joint,
    BR_servo_femur_joint,
    BR_servo_tibia_joint,
    BL_servo_body_joint,
    BL_servo_femur_joint,
    BL_servo_tibia_joint,
]

joint_sensor = [
    FR_servo_body_joint_sensor,
    FR_servo_femur_joint_sensor,
    FR_servo_tibia_joint_sensor,
    FL_servo_body_joint_sensor,
    FL_servo_femur_joint_sensor,
    FL_servo_tibia_joint_sensor,
    BR_servo_body_joint_sensor,
    BR_servo_femur_joint_sensor,
    BR_servo_tibia_joint_sensor,
    BL_servo_body_joint_sensor,
    BL_servo_femur_joint_sensor,
    BL_servo_tibia_joint_sensor,
]



state = 0  # state 0 = idle / 1 = moving to intermediate position / 2 = moving to target position / 3 = reset
# Random Target step rotation for the step:
target_step_rotation = np.float32(0.0)

agentCreated = False  # To measure velocity only when there is an agent created (not between episodes where the agent is destroyed)
step_completed = False  # To compute the mean velocities only when each step is completed

step_counter = 0



prev_time = 0.0
prev_forward_velocity = 0.0
prev_lateral_velocity = 0.0
current_position =        np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32)
prev_angular_position =   np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32)
joint_angular_velocity =  np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
joint_torque=             np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 

forward_velocity = 0.0
lateral_velocity = 0.0

vel_samples = 0.0 #Number of samples of velocity sensed each agent step
measure = 50
mean_forward_velocity = 0.0
mean_lateral_velocity = 0.0

forward_acceleration = 0.0
lateral_acceleration = 0.0
max_forward_acc = 0.0

paws_heights = np.zeros(4, dtype=np.float32)
paws_up_acum = np.zeros(4, dtype=int)
paws_down_acum = np.zeros(4, dtype=int)
paws_ration_down_total = np.zeros(4, dtype=np.float32)

attitude = np.array([0.0,0.0])
ang_vel = np.array([0.0,0.0])
reset_orientation = np.array([0.0,0.0,0.0])

sim_measure_dim = 35
obs_dim = 17
environment_state = np.zeros(sim_measure_dim+obs_dim, dtype=np.float32)

## Servo_tibia_joint.setPosition(1.57)
#FL_servo_tibia_joint.setPosition(+20 * 3.14 / 180)
# Servo_femur_joint.setPosition(0*3.14/180)
#t = 0




def Kalman_filter():
    global accelerometer,acc_std,ax_vector,ay_vector,az_vector,coef_acc,gyroscope,gyro_std,wx_vector,wy_vector,wz_vector,coef_gyr,pitch,roll,vel_pitch,vel_roll,A,I,X,P,Q,R,K,H,Z
    # Accelerometer readings (changing the axis)
    acceleration = accelerometer.getValues()
    #print("ax,ay,az",acceleration[0],acceleration[1],acceleration[2])

    # Add noise:
    acceleration = acceleration + np.random.normal(0, acc_std * G, size=3)

    # Shift array of input vectors and add new value (Changing axis: adapt to axis of sensing device)
    ax_vector.appendleft(-acceleration[0])
    ay_vector.appendleft(acceleration[1])
    az_vector.appendleft(-acceleration[2])

    # Apply FIR filter to input vectors:
    ax = np.dot(coef_acc, ax_vector)
    ay = np.dot(coef_acc, ay_vector)
    az = np.dot(coef_acc, az_vector)

    # Gyroscope readings (changing the axis):
    angular_velocities = gyroscope.getValues()

    # Add noise:
    angular_velocities = angular_velocities + np.random.normal(0, gyro_std * np.pi / 180, size=3)

    # Shift array of input vectors and add new value (Changing axis: adapt to axis of sensing device)
    wx_vector.appendleft(-angular_velocities[0])
    wy_vector.appendleft(angular_velocities[1])
    wz_vector.appendleft(-angular_velocities[2])

    # Apply FIR filter to input vectors:
    wx = np.dot(coef_gyr, wx_vector)
    wy = np.dot(coef_gyr, wy_vector)
    wz = np.dot(coef_gyr, wz_vector)

    vel_pitch = wy
    vel_roll = wx

    dt = timestep * 1e-3

    A = np.array([  [ 0,   -wx,    -wy,    -wz],
                    [wx,     0,     wz,    -wy],
                    [wy,   -wz,      0,     wx],
                    [wz,    wy,    -wx,      0]])
    
    A = I + A * dt * 0.5

    X = A @ X

    P = (A @ P) @ A.transpose() + Q

    angles = EP_2_Euler321(X)

    yaw = angles[0]

    aux = np.linalg.inv((H @ P) @ H.transpose() + R)

    K = (P @ H.transpose()) @ aux

    angles = Accel_2_Euler(ax, ay, az)

    angles[0] = yaw

    Z = Euler321_2_EP(angles)

    X = X + K @ (Z - (H @ X))

    P = P - (K @ H) @ P

    angles = EP_2_Euler321(X)

    yaw = angles[0]
    pitch = angles[1]
    roll = angles[2]


def EP_2_Euler321(Q):
    
    Q = np.squeeze(Q)
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    e = np.zeros(3)

    e[0] = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    e[1] = np.arcsin(-2 * (q1 * q3 - q0 * q2))
    e[2] = np.arctan2(2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)

    return e


def Euler321_2_EP(e):
    c1 = np.cos(e[0] / 2)
    s1 = np.sin(e[0] / 2)
    c2 = np.cos(e[1] / 2)
    s2 = np.sin(e[1] / 2)
    c3 = np.cos(e[2] / 2)
    s3 = np.sin(e[2] / 2)

    Q = np.zeros((4, 1))

    Q[0] = c1 * c2 * c3 + s1 * s2 * s3
    Q[1] = c1 * c2 * s3 - s1 * s2 * c3
    Q[2] = c1 * s2 * c3 + s1 * c2 * s3
    Q[3] = s1 * c2 * c3 - c1 * s2 * s3

    return Q


def Accel_2_Euler(ax, ay, az):
    g = np.sqrt(np.power(ax, 2) + np.power(ay, 2) + np.power(az, 2))

    theta = np.arcsin(ax / g)  # Rotation in y axis, estimator of pitch

    phi = np.arctan2(-ay, -az)  # Rotation in x axis, estimator of roll

    angles = np.array([0, theta, phi])

    return angles
def resetKalman():
  global acc_std,ax_vector,ay_vector,az_vector,coef_acc,gyro_std,wx_vector,wy_vector,wz_vector,coef_gyr,pitch,roll,vel_pitch,vel_roll,A,I,X,P,Q,R,K,H,Z
  
  H = I

  K = np.zeros((4, 4))
  A = np.zeros((4, 4))

  Q = I * np.power(gyro_std * np.pi / 180, 2)
  R = I * np.power(acc_std * G, 2)
  P = I * 0.1
  X = np.array([[1], [0], [0], [0]])

  Z = np.zeros((4, 1))


  ## FIR FILTER
  coef_acc = np.ones(10) / 10  # Moving average FIR filters
  coef_gyr = np.ones(3) / 3

  ax_vector = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=10)
  ay_vector = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=10)
  az_vector = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=10)

  wx_vector = deque([0, 0, 0], maxlen=3)
  wy_vector = deque([0, 0, 0], maxlen=3)
  wz_vector = deque([0, 0, 0], maxlen=3)

  pitch = 0
  roll = 0

  vel_pitch = 0
  vel_roll = 0

#States for Main State Machine
RESET =	0
TX_RASPBERRY = 1
RX_RASPBERRY = 2
ACTUATION =3

#States for Actuation State Machine
RESET_ACTUATION = 0
DELAY_STATE = 1
COMPARE_MEASURE = 2
TIMEOUT_STATE = 3

#Parameters of the actuation and time-out algorithm
DEAD_BANDWIDTH_SERVO = 5 *np.pi/180.0 # Degrees
MAX_DELTA_ANGLE = 3.5*np.pi/180.0  # Degrees
MAX_DELTA_SAMPLE = 1.0 *np.pi/180.0 # Degrees

JOINT_MAX_NOISE = 0.5*np.pi/180.0  # degrees

SAMPLE_TIME  = 10	  #miliseconds
TIMEOUT = 500      #miliseconds

ALL_FINISHED = 0xFFF  #(12 ones)

state = RESET
delta_sample = 0
delta_target = 0
f_joint_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

f_last_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
state_actuation = RESET_ACTUATION
joints_finished = ALL_FINISHED # Each bit is a flag for a joint: 1-Finished, 0-Unfinished
stuck_servo = 0 # returned by State_Machine_Actuation

#Ticks of timer
sample_time = 0
timeout = 0



# Function to reset robot position and orientation
def reset_robot_position_orientation():
    global robot_node,supervisor,reset_orientation,reset_pos
    
    print("RESET: MOSAC reset")
    # Apply the reset position and orientation
    robot_node.getField("translation").setSFVec3f([reset_pos[0],reset_pos[1],reset_pos[2]])
    robot_node.getField("rotation").setSFRotation(ypr_to_axis_angle(reset_orientation[2],reset_orientation[1],reset_orientation[0]))

    # supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
    # Reset other robot states if needed (e.g., velocity, sensor data)
    supervisor.simulationReset()  # Reset physics without resetting the simulation

    for i in range(len(joint)):
      #! EN RADIANES HAY QUE PONERLO!!!!//Move the servomotor
      joint[i].setPosition(0)

def SendState():
  global environment_state, robot_node,Tx_float_length,reset_pos,reset_orientation,mean_forward_velocity,mean_lateral_velocity,max_forward_acc,target_step_rotation,slider,step_counter,attitude,ang_vel,joint_sensor,f_joint_angle,jointUpperLimit,jointLowerLimit,step_completed,joint_angular_velocity, paws_up_acum, paws_down_acum
  
  for i in range(len(joint)):
    #without noise in joints for now
    #added_noise = (math.random() * 2 - 1) * JOINT_MAX_NOISE * math.pi/180   - - [-JOINT_MAX_NOISE;JOINT_MAX_NOISE] in radians
    #jointPos[i] = (sim.getJointPosition(joint[i]) + added_noise - (jointUpperLimit[i]+jointLowerLimit[i])/2) / ((jointUpperLimit[i]-jointLowerLimit[i])/2)
    f_joint_angle[i] = (joint_sensor[i].getValue() - (jointUpperLimit[i] + jointLowerLimit[i])/2.0) / ((jointUpperLimit[i]-jointLowerLimit[i])/2.0)

  current_translation = robot_node.getField("translation").getSFVec3f()
  current_rotation = robot_node.getField("rotation").getSFRotation()
  # Convert axis-angle to a rotation object
  rotation = Rotation.from_rotvec(np.array(current_rotation[:3]) * current_rotation[3])
  # Extract yaw, pitch, and roll (in radians)
  current_rotation = rotation.as_euler('zyx', degrees=False)

  #Fill the state array with the observed state and additional measurements:
    #Measurements used for the observation state of the agent
  environment_state[0] = target_step_rotation
  environment_state[1:3] = attitude/np.pi
  environment_state[3:5] = ang_vel/np.pi
  environment_state[5:17] = f_joint_angle
  

    #Additional Measurements used to compute the rewards and/ or plots:
  environment_state[17:20] = current_translation
  environment_state[20] = mean_forward_velocity
  environment_state[21] = mean_lateral_velocity
  environment_state[22] = max_forward_acc
  environment_state[23] = current_rotation[0]
  environment_state[24:36] = joint_torque
  environment_state[36:48] = joint_angular_velocity

  
  environment_state[48] = 0#paws_up_acum[0] / (paws_up_acum[0] + paws_down_acum[0])
  environment_state[49] = 0#paws_up_acum[1] / (paws_up_acum[1] + paws_down_acum[1])
  environment_state[50] = 0#paws_up_acum[2] / (paws_up_acum[2] + paws_down_acum[2])
  environment_state[51] = 0#paws_up_acum[3] / (paws_up_acum[3] + paws_down_acum[3])

  client.sendall(environment_state.tobytes())

  step_completed = True


  # if slider:
  #   --Use value from slider(converting to radians normalized by pi)
  #     simUI.setLabelText(ui, 1, string.format(
  #         'Target Step Rotation = %d %s', targetRotSlider, utf8.char(176)))
  #     target_step_rotation = targetRotSlider / 180

  if programmed_target_rotation:
      if (step_counter >= 80) and (step_counter < 100):
        target_step_rotation = -6/180
      elif(step_counter >= 100) and (step_counter < 120):
        target_step_rotation = 5/180
      elif(step_counter >= 120) and (step_counter < 140):
        target_step_rotation = 0
      elif(step_counter >= 140) and (step_counter < 160):
        target_step_rotation = -10/180
      elif(step_counter >= 160) and (step_counter < 180):
        target_step_rotation = 10/180
      elif(step_counter >= 180):
        target_step_rotation = 0
  else:
    #Generate next random target step rotation (std = 5?)
    if (step_counter >= 50) and (step_counter % 50 == 0):
      mean = 0.0
      std_dev = 1/36
      clip_range = 1/18
      target_step_rotation = random.gauss(mean, std_dev) #? Cambiar a Uniforme
      #-Clip the target step rotation to + /- 10?
      target_step_rotation = max(-clip_range, min(clip_range, target_step_rotation))

  #Target Rotation for the step
  #print("target_step_rotation",target_step_rotation)
  # client.send(Tx_float_length.format(target_step_rotation).encode('utf-8'))

  
    
  
  #Pitch and roll angles of the back(world reference)
  
  # client.send(Tx_float_length.format(attitude[0]/np.pi).encode('utf-8')) #pitch angle
  # client.send(Tx_float_length.format(attitude[1]/np.pi).encode('utf-8'))#roll angle
  #print("pitch angular_vel",ang_vel[0])
  #print("roll angular_vel",ang_vel[1])
  #Pitch and roll angular velocities of the back(world reference)
  # client.send(Tx_float_length.format(ang_vel[0]/np.pi).encode('utf-8')) #pitch angular_vel
  # client.send(Tx_float_length.format(ang_vel[1]/np.pi).encode('utf-8')) #roll angular_vel

  #Joints angular positions
  # for i in range(len(joint)):
    #without noise in joints for now
    #added_noise = (math.random() * 2 - 1) * JOINT_MAX_NOISE * math.pi/180   - - [-JOINT_MAX_NOISE;JOINT_MAX_NOISE] in radians
    #jointPos[i] = (sim.getJointPosition(joint[i]) + added_noise - (jointUpperLimit[i]+jointLowerLimit[i])/2) / ((jointUpperLimit[i]-jointLowerLimit[i])/2)
    # f_joint_angle[i] = (joint_sensor[i].getValue() - (jointUpperLimit[i] + jointLowerLimit[i])/2.0) / ((jointUpperLimit[i]-jointLowerLimit[i])/2.0)
    
    # client.send(Tx_float_length.format(f_joint_angle[i]).encode('utf-8'))    
  #print("Joints angular positions",f_joint_angle)

  #World position(change the second parameter from -1 to another handle to get a relative position)
  # current_translation = robot_node.getField("translation").getSFVec3f()
  #print("World position",current_translation)
  # client.send(Tx_float_length.format(current_translation[0]).encode('utf-8')) #x
  # client.send(Tx_float_length.format(current_translation[1]).encode('utf-8')) #y
  # client.send(Tx_float_length.format(current_translation[2]).encode('utf-8')) #z

  #Mean forward and lateral velocities and peak absolute forward acceleration with reference of the agent
  #Measured and computed in sysCall_sensing
  #print("mean_forward_velocity",mean_forward_velocity)
  #print("mean_lateral_velocity",mean_lateral_velocity)
  #print("max_forward_acc",max_forward_acc)
  # client.send(Tx_float_length.format(mean_forward_velocity).encode('utf-8'))
  # client.send(Tx_float_length.format(mean_lateral_velocity).encode('utf-8'))
  # client.send(Tx_float_length.format(max_forward_acc).encode('utf-8'))

  # step_completed = True


  #Object world orientation(change the second parameter from -1 to another handle to get a relative position)
  #data = sim.getObjectOrientation(agent, -1)
  # current_rotation = robot_node.getField("rotation").getSFRotation()
  # Convert axis-angle to a rotation object
  # rotation = Rotation.from_rotvec(np.array(current_rotation[:3]) * current_rotation[3])
  # Extract yaw, pitch, and roll (in radians)
  # current_rotation = rotation.as_euler('zyx', degrees=False)
  # client.send(Tx_float_length.format(current_rotation[0]).encode('utf-8')) #Yaw angle for the reward
  
  #Send torque
  # print("joint_torque",joint_torque)
  # for i in range(len(joint)):
  #   client.send(Tx_float_length.format(joint_torque[i]).encode('utf-8'))

  #Send Angular Velocity
  #print("joint_angular_velocity",joint_angular_velocity)
  # for i in range(len(joint)):
  #   client.send(Tx_float_length.format(joint_angular_velocity[i]).encode('utf-8'))      
  
def State_Machine_Control():

  global state,state_actuation, joint_sensor, f_joint_angle,f_last_joint, joints_finished, target_joint, delta_target, joint,stuck_servo,prev_forward_velocity,target_step_rotation,prev_lateral_velocity,max_forward_acc,step_completed,step_counter,mean_forward_velocity,mean_lateral_velocity,max_forward_acc,vel_samples,prev_angular_position
  

  if state == RESET:
    
    for i in range(len(joint)):
      f_joint_angle[i] = joint_sensor[i].getValue()
    state = TX_RASPBERRY

  elif state == TX_RASPBERRY:
    SendState()
    state = RX_RASPBERRY

  elif state == RX_RASPBERRY:

    #Receive the agent's next action
    data = client.recv(Tx_Rx_command_length).decode('utf-8')
    

    if data == "RESET":

      print("RESET: Environment reset")

      # Receive new position
      # reset_pos[0] = float(client.recv(Rx_float_length).decode('utf-8'))
      # reset_pos[1] = float(client.recv(Rx_float_length).decode('utf-8'))

      # Receive new orientation and convert it to radians
      # reset_orientation[2] = float(client.recv(Rx_float_length).decode('utf-8'))

      pos_ang = np.copy(np.frombuffer(client.recv(8*3), dtype='<f8'))   #< little endian, > big endian

      reset_pos = pos_ang[0:2]
      reset_orientation[2] = np.pi * pos_ang[2]

      reset_robot_position_orientation()
      resetKalman()

      step_counter = 0.0

      mean_forward_velocity = 0.0
      mean_lateral_velocity = 0.0

      prev_forward_velocity = 0.0
      prev_lateral_velocity = 0.0
      max_forward_acc = 0.0

      prev_angular_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

      max_forward_acc = 0.0

      vel_samples = 0.0

      step_completed = False 

      target_step_rotation = 0.0

      f_joint_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      target_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      f_last_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

      state = RESET
      state_actuation = RESET_ACTUATION

    elif data == "ACT__":
      normalized_action = np.copy(np.frombuffer(client.recv(8*len(joint)), dtype='<f8'))   #< little endian, > big endian      
      for i in range(len(joint)):
        # data = float(client.recv(Rx_float_length).decode('utf-8'))
        target_joint[i] = ((jointUpperLimit[i]-jointLowerLimit[i])/2.0) * normalized_action[i] + (jointUpperLimit[i]+jointLowerLimit[i])/2.0

      state = ACTUATION

  elif state == ACTUATION:
    stuck_servo = State_Machine_Actuation()

    # if (pitch*180/np.pi >= critical_failure_angle or roll*180/np.pi >= critical_failure_angle):
    #     supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    #     state = TX_RASPBERRY

    if stuck_servo == 1:
      state = TX_RASPBERRY

    elif stuck_servo == -1:
      state = TX_RASPBERRY  # ESTADO STOP DE EMERGENCIA EN EL FUTURO




def State_Machine_Actuation():
  global sample_time,state_actuation,joint,joint_sensor,f_joint_angle,f_last_joint,timeout,joints_finished,delta_sample,target_joint,delta_target,joint

  if state_actuation == RESET_ACTUATION:
    # Check if the new target is too close to the actual position, in which case the servo wouldn't move because of the dead bandwidth
    # Therefore consider the joint already finished
    
    for i in range(len(joint)):
      f_joint_angle[i] = joint_sensor[i].getValue()
      f_last_joint[i] = f_joint_angle[i]

      delta_target = np.abs(f_joint_angle[i] - target_joint[i])
      if delta_target >= DEAD_BANDWIDTH_SERVO:
        joints_finished &= ~(1 << i)  # Set respective bit in 0
        #! EN RADIANES HAY QUE PONERLO!!!!//Move the servomotor
        joint[i].setPosition(target_joint[i]+np.random.uniform(-JOINT_MAX_NOISE,JOINT_MAX_NOISE))      

    # Start first measure of joints before ACTUATION

    if joints_finished == ALL_FINISHED:  # Si ninguna articulación se tiene que mover, salteo la actuacion
      print("Salteo paso")
      return 1
    else:
      sample_time = SAMPLE_TIME
      state_actuation = DELAY_STATE
      timeout = TIMEOUT
		
  elif state_actuation == DELAY_STATE:
    if timeout == 0:
      state_actuation = TIMEOUT_STATE
      
    elif sample_time == 0:    
      state_actuation = COMPARE_MEASURE
    
  elif state_actuation == COMPARE_MEASURE:
    if(timeout == 0):
      state_actuation = TIMEOUT_STATE
      #print("Timeout, joints finished",bin(joints_finished))

    else:
      for i in range(len(joint)):

        if (joints_finished & (1 << i)) == 0:  # Ignore joints that finished
          
          f_joint_angle[i] = joint_sensor[i].getValue()
          delta_sample = np.abs(f_joint_angle[i] - f_last_joint[i])

          #if delta_sample < MAX_DELTA_SAMPLE:  # Not Moving

          delta_target = np.abs(f_joint_angle[i] - target_joint[i])

          if delta_target < MAX_DELTA_ANGLE:  # If the stable joint reached target

            joints_finished |= 1 << i  # Set respective bit in 1
            #print(bin(joints_finished))
            if joints_finished == ALL_FINISHED:


              state_actuation = RESET_ACTUATION
              return 1
          #else:
            # timeout = TIMEOUT

        
      f_last_joint = np.copy(f_joint_angle)	# Remember joint

      sample_time = SAMPLE_TIME
      state_actuation = DELAY_STATE			

  elif state_actuation == TIMEOUT_STATE:
    print("Step Aborted")
    state_actuation = RESET_ACTUATION

    return -1
  
  return 0

def computeMeasurementsForRewards():
  global supervisor,root,forward_acceleration,forward_velocity,lateral_velocity,mean_forward_velocity,mean_lateral_velocity,lateral_acceleration,max_forward_acc,prev_time,prev_forward_velocity,prev_lateral_velocity,vel_samples,step_completed,step_counter,joint,current_position,prev_angular_position,joint_angular_velocity,joint_torque, paws_heights, paws_up_acum, paws_down_acum, paw_FR,paw_FL,paw_BR,paw_BL
  world_velocity = root.getVelocity()


  current_rotation = robot_node.getField("rotation").getSFRotation()
  # Convert axis-angle to a rotation object
  rotation = Rotation.from_rotvec(np.array(current_rotation[:3]) * current_rotation[3])
  # Extract yaw, pitch, and roll (in radians)
  current_rotation = rotation.as_euler('zyx', degrees=False)
  
  #print("Yaw", current_rotation[0]*180/np.pi)

  A2[0] = np.cos(current_rotation[0])
  A2[1] = np.sin(current_rotation[0])

  #print("Yaw,pitch,roll",current_rotation[0],current_rotation[1],current_rotation[2])

  A_orth[0] = A2[1]
  A_orth[1] = -A2[0]

  forward_velocity = world_velocity[0] * A2[0] + world_velocity[1] * A2[1]
  lateral_velocity = world_velocity[0] * A_orth[0] + world_velocity[1] * A_orth[1]

  time = supervisor.getTime()

  if time > 0.0:
    forward_acceleration = (forward_velocity - prev_forward_velocity)/(time - prev_time)
    lateral_acceleration = (lateral_velocity - prev_lateral_velocity)/(time - prev_time)
    #print(time-prev_time)

    if np.abs(forward_acceleration) > max_forward_acc:
        max_forward_acc = np.abs(forward_acceleration)

  
  prev_forward_velocity = forward_velocity
  prev_lateral_velocity = lateral_velocity

  # Compute mean in a recursive manner
  mean_forward_velocity = 1 / (vel_samples + 1.0) * (mean_forward_velocity * vel_samples + forward_velocity)
  mean_lateral_velocity = 1 / (vel_samples + 1.0) * (mean_lateral_velocity * vel_samples + lateral_velocity)  

  vel_samples = vel_samples + 1.0

  #Compute Joint's angular velocity
  for i in range(len(joint)):   
    current_position[i] = joint_sensor[i].getValue()
  
    # Calculate angular velocity (approximate)
    joint_angular_velocity[i] = (current_position[i] - prev_angular_position[i]) /(time - prev_time)  # Convert timestep to seconds  
    # Update previous position for the next loop
    prev_angular_position[i] = current_position[i]

  prev_time = time

  for i in range(len(joint)):
      joint_torque[i] = joint[i].getTorqueFeedback()
    
  
  #Paws measurements:
  # paws_heights[0] = paw_FR.getPosition()[2]
  # paws_heights[1] = paw_FL.getPosition()[2]
  # paws_heights[2] = paw_BR.getPosition()[2]
  # paws_heights[3] = paw_BL.getPosition()[2]

  # for paw in range(len(paws_heights)):
  #   if(paws_heights[paw] > 0.06) and (paws_heights[paw] < 0.08):
  #     paws_up_acum[paw] += 1
  #   else:
  #     paws_down_acum[paw] += 1
    

  if step_completed == True:

      step_counter = step_counter + 1

      mean_forward_velocity = 0.0
      mean_lateral_velocity = 0.0

      max_forward_acc = 0.0

      vel_samples = 0.0

      paws_up_acum[:] = 0
      paws_down_acum[:] = 0

      step_completed = False
      #print("HOLASS")


if __name__ == "__main__":
    
    ####Init the floor####
    # terrain_x_size = 20          #meters
    # terrain_y_size = 20          #meters
    # terrain_resolution = 0.05   #meters

    # terrain_x_dimension = int(terrain_x_size/terrain_resolution)
    # terrain_y_dimension = int(terrain_y_size/terrain_resolution)

    #Set position of solid node based on size:
    # terrain_node = supervisor.getFromDef('uneven_terrain')
    # terrain_translation_field = terrain_node.getField('translation')
    # terrain_translation_field.setSFVec3f([-terrain_x_size/2, -terrain_y_size/2, 0.01])

    # #Set dimension and spacing fields of the terrain grid node:
    # terrain_grid_node = supervisor.getFromDef('terrain_grid')

    # terrain_grid_node.getField('xDimension').setSFInt32(terrain_x_dimension)
    # terrain_grid_node.getField('yDimension').setSFInt32(terrain_y_dimension)
    # terrain_grid_node.getField('xSpacing').setSFFloat(terrain_resolution)
    # terrain_grid_node.getField('ySpacing').setSFFloat(terrain_resolution)

    #Set the height array of the terrain grid node:
    # terrain_grid_height_field = terrain_grid_node.getField('height')

    # print('hola')
    # for x_index in range(terrain_x_dimension):

    #   for y_index in range(terrain_y_dimension):
        
    #     if x_index % 3 == 0 or y_index % 3 == 0:
    #       terrain_grid_height_field.setMFFloat(x_index*terrain_x_dimension + y_index, -0.01)
    #     else:
    #       terrain_grid_height_field.setMFFloat(x_index*terrain_x_dimension + y_index, 0)
    
    # Main loop:
    while supervisor.step(timestep) != -1:
      
      if sample_time >0: sample_time-=timestep
      if timeout >0: timeout-=timestep

      #Compute every 50ms instead of 10ms to avoid high peaks of acceleration
      measure-=timestep
      if(measure<=0):          
        computeMeasurementsForRewards() 
        measure = 50

      Kalman_filter()
      attitude[0],attitude[1] = pitch,roll
      ang_vel[0],ang_vel[1] = vel_pitch,vel_roll   

      State_Machine_Control()

    # Optionally add code here clean when simulation is finished (supervisor.step(timestep) == -1)

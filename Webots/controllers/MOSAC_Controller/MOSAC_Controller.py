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

import csv

# CSV output filename
output_file = "webots_sin_pid_test.csv"
data = [] 

# Initialize header
header = ["timestamp"]

header.append("BFR_Target")
header.append("BFR_Filt")
header.append("BFR_PID_OUT")
header.append("FFR_Target")
header.append("FFR_Filt")
header.append("FFR_PID_OUT")
header.append("TFR_Target")
header.append("TFR_Filt")
header.append("TFR_PID_OUT")

header.append("BFL_Target")
header.append("BFL_Filt")
header.append("BFL_PID_OUT")
header.append("FFL_Target")
header.append("FFL_Filt")
header.append("FFL_PID_OUT")
header.append("TFL_Target")
header.append("TFL_Filt")
header.append("TFL_PID_OUT")

header.append("BBR_Target")
header.append("BBR_Filt")
header.append("BBR_PID_OUT")
header.append("FBR_Target")
header.append("FBR_Filt")
header.append("FBR_PID_OUT")
header.append("TBR_Target")
header.append("TBR_Filt")
header.append("TBR_PID_OUT")

header.append("BBL_Target")
header.append("BBL_Filt")
header.append("BBL_PID_OUT")
header.append("FBL_Target")
header.append("FBL_Filt")
header.append("FBL_PID_OUT")
header.append("TBL_Target")
header.append("TBL_Filt")
header.append("TBL_PID_OUT")

#To print touch sensor signals
# header.append("PBL_sensor")
# header.append("PFL_sensor")
# header.append("PBR_sensor")
# header.append("PFR_sensor")

# header.append("PBL_target")
# header.append("PFL_target")
# header.append("PBR_target")
# header.append("PFR_target")

# slider = False
programmed_target_rotation = False

critical_failure_angle = 50 #Must match angle in environment.py

supervisor = Supervisor()
root = supervisor.getSelf()
robot_node = supervisor.getFromDef('MOSAC')

PFR_node = supervisor.getFromDef('PFR')
PFL_node = supervisor.getFromDef('PFL')
PBR_node = supervisor.getFromDef('PBR')
PBL_node = supervisor.getFromDef('PBL')

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

reset_pos = np.array([0.0, 0.0, 0.139])

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
timestep = int(supervisor.getBasicTimeStep())

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

#Joint position sensors
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

#Paw touch sensors
FR_touch_sensor = supervisor.getDevice("touch_FR")
FL_touch_sensor = supervisor.getDevice("touch_FL")
BR_touch_sensor = supervisor.getDevice("touch_BR")
BL_touch_sensor = supervisor.getDevice("touch_BL")

FR_touch_sensor.enable(timestep)
FL_touch_sensor.enable(timestep)
BR_touch_sensor.enable(timestep)
BL_touch_sensor.enable(timestep)

bod_llim, bod_ulim = -10.0 , 15.0 
femur_llim, femur_ulim = -20.0 , 30.0 
tibia_llim, tibia_ulim = -15.0 , 15.0 

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

A2 = np.array([0.0, 0.0])
A_orth = np.array([0.0, 0.0])


state = 0  # state 0 = idle / 1 = moving to intermediate position / 2 = moving to target position / 3 = reset
# Random Target step rotation for the step:
target_step_rotation = np.float32(0.0)

agentCreated = False  # To measure velocity only when there is an agent created (not between episodes where the agent is destroyed)
step_omitted = 0

step_counter = 0

prev_time = 0.0

prev_velocity_magnitude = 0.0
prev_forward_velocity = 0.0
prev_lateral_velocity = 0.0

forward_velocity = 0.0
lateral_velocity = 0.0

vel_samples = 0.0 #Number of samples of velocity sensed each agent step
measure = 50
mean_forward_velocity = 0.0
mean_lateral_velocity = 0.0

forward_acceleration = 0.0
lateral_acceleration = 0.0
max_forward_acc = 0.0


#Servo Control Variables
joint_angular_velocity_servo = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
prev_joint_angular_position_servo = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
joint_torque = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
error_servo = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
prev_error_servo= np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
delta_error_servo = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
acum_error_servo = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
Ia = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
enable_torque_control = 0
pid_sample = 0

KP_servo = 18.847
KD_servo = 0.023541
KI_SERVO = 2.0231
KM = 0.9853919147
RA = 1.45
TF = 0.03262365753

#PID Control Variables
prev_joint_angular_position = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
prev_joint_angular_position = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
prev_error= np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
delta_error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
acum_error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
WINDOW_SIZE = 40
TIMER_MEDIAN_FILTER = 0.5#ms
delay_buffer = np.zeros((12,int((WINDOW_SIZE/2 *  TIMER_MEDIAN_FILTER)/timestep)),dtype=np.float32) #For median filter
KP = 0.5
KD = 0.0
KI = 0.1
K_TON = 1000.0 / 140.0

step_complete = 0

paw_transitions = np.zeros(4,dtype=np.float32)
transition_phase = np.zeros(4,dtype=np.float32)
paw_previous_state = [True, True, True, True]
paw_transition_enable = [True, True, True, True]

attitude = np.array([0.0, 0.0])
ang_vel = np.array([0.0, 0.0])
reset_orientation = np.array([0.0, 0.0, 0.0])

obs_dim = 17
sim_measure_dim = 21
environment_state = np.zeros(sim_measure_dim + obs_dim, dtype=np.float32)

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
DEAD_BANDWIDTH_SERVO = 3  # Degrees
MAX_DELTA_ANGLE = 2  # Degrees

JOINT_MAX_NOISE = 0.5  # degrees

TIMEOUT = 1000      #miliseconds

ALL_FINISHED = 0xFFF  #(12 ones)

state = RESET
delta_sample = 0
delta_target = 0
f_joint_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


state_actuation = RESET_ACTUATION
joints_finished = ALL_FINISHED # Each bit is a flag for a joint: 1-Finished, 0-Unfinished
stuck_servo = 0 # returned by State_Machine_Actuation

#Ticks of timer
timeout = 0

sign = 1



# Function to reset robot position and orientation
def reset_robot_position_orientation():
    global robot_node,supervisor,reset_orientation,reset_pos
    
    print("RESET: MOSAC reset")

    # Reset other robot states if needed (e.g., velocity, sensor data)
    supervisor.simulationReset()  # Reset physics without resetting the simulation

    # Apply the reset position and orientation
    print("reset_pos = ", reset_pos)
    robot_node.getField("translation").setSFVec3f([reset_pos[0],reset_pos[1],reset_pos[2]])
    robot_node.getField("rotation").setSFRotation(ypr_to_axis_angle(reset_orientation[2],reset_orientation[1],reset_orientation[0]))

    for i in range(len(joint)):
      #! EN RADIANES HAY QUE PONERLO!!!!//Move the servomotor
      # joint[i].setPosition(float('inf'))
      # joint[i].setVelocity(0)
      joint[i].setPosition(0)

def SendState():
  global environment_state, robot_node,Tx_float_length,reset_pos,reset_orientation,mean_forward_velocity
  global mean_lateral_velocity,max_forward_acc,target_step_rotation,slider,step_counter,attitude,ang_vel
  global joint_sensor,f_joint_angle,jointUpperLimit,jointLowerLimit
  global step_omitted, paw_transitions, transition_phase
  
  f_joint_angle_norm = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  
  for i in range(len(joint)):
    #without noise in joints for now
    #added_noise = (math.random() * 2 - 1) * JOINT_MAX_NOISE * math.pi/180   - - [-JOINT_MAX_NOISE;JOINT_MAX_NOISE] in radians
    #jointPos[i] = (sim.getJointPosition(joint[i]) + added_noise - (jointUpperLimit[i]+jointLowerLimit[i])/2) / ((jointUpperLimit[i]-jointLowerLimit[i])/2)
    f_joint_angle_norm[i] = (f_joint_angle[i] - (jointUpperLimit[i] + jointLowerLimit[i])/2.0) / ((jointUpperLimit[i]-jointLowerLimit[i])/2.0)

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
  environment_state[5:17] = f_joint_angle_norm

    #Additional Measurements used to compute the rewards and/ or plots:
  environment_state[17] = current_phase
  environment_state[18:21] = current_translation
  environment_state[21] = mean_forward_velocity
  environment_state[22] = mean_lateral_velocity
  environment_state[23] = max_forward_acc
  environment_state[24] = current_rotation[0]
  environment_state[25:29] = paw_transitions
  environment_state[29:33] = transition_phase
  environment_state[33] = BL_touch_sensor.getValue()
  environment_state[34] = FL_touch_sensor.getValue()
  environment_state[35] = BR_touch_sensor.getValue()
  environment_state[36] = FR_touch_sensor.getValue()
  environment_state[37] = step_omitted

  client.sendall(environment_state.tobytes())

  step_omitted = 0

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
    #Generate next random target step rotation (uniform distribution +-10°)
    if (step_counter >= 100) and (step_counter % 50 == 0):
      clip_range = 1/18
      target_step_rotation = random.uniform(-clip_range, clip_range)
  
def State_Machine_Control():

  global state,state_actuation, joint_sensor
  global reset_pos, reset_orientation
  global enable_torque_control, f_joint_angle
  global joints_finished, target_joint, delta_target, joint
  global prev_velocity_magnitude, prev_forward_velocity,target_step_rotation
  global prev_lateral_velocity,max_forward_acc,step_counter
  global mean_forward_velocity,mean_lateral_velocity,max_forward_acc,vel_samples,prev_time
  global prev_joint_angular_position_servo,acum_error_servo,prev_joint_angular_position,acum_error
  global pid_sample,step_complete,delay_buffer
  global paw_transitions, transition_phase, paw_previous_state, paw_transition_enable
  global repeat
  # global debug_delay
  global sign
  

  if state == RESET:
    state = TX_RASPBERRY
    repeat = 1

  elif state == TX_RASPBERRY:


    # if debug_delay == 0:
    SendState()
    resetMeasurementsForRewards()
      # debug_delay = 3000
    state = RX_RASPBERRY
    repeat = 1

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

      # reset_pos[0:2] = pos_ang[0:2]
      # reset_orientation[2] = np.pi * pos_ang[2]

      reset_robot_position_orientation()
      resetKalman()

      step_counter = 0.0

      mean_forward_velocity = 0.0
      mean_lateral_velocity = 0.0
      
      prev_velocity_magnitude = 0.0
      prev_forward_velocity = 0.0
      prev_lateral_velocity = 0.0
      max_forward_acc = 0.0
      prev_time = 0.0

      prev_joint_angular_position_servo = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      acum_error_servo = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      
      prev_joint_angular_position= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      acum_error = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      delay_buffer = np.zeros((12,int((WINDOW_SIZE/2 *  TIMER_MEDIAN_FILTER)/timestep)),dtype=np.float32) #For median filter

      
      max_forward_acc = 0.0

      vel_samples = 0.0
 
      step_complete = 0

      target_step_rotation = 0.0

      enable_torque_control = 0
      pid_sample = 0

      f_joint_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      target_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

      paw_transitions = np.zeros(4,dtype=np.float32)
      transition_phase = np.zeros(4,dtype=np.float32)
      
      paw_previous_state = [BL_touch_sensor.getValue(), FL_touch_sensor.getValue(), BR_touch_sensor.getValue(), FR_touch_sensor.getValue()]
      paw_transition_enable = [True, True, True, True]
      
      state = RESET
      state_actuation = RESET_ACTUATION
      repeat = 1

    elif data == "ACT__":
      normalized_action = np.copy(np.frombuffer(client.recv(8*len(joint)), dtype='<f8'))   #< little endian, > big endian      
      for i in range(len(joint)):
        target_joint[i] = ((jointUpperLimit[i]-jointLowerLimit[i])/2.0) * normalized_action[i] + (jointUpperLimit[i]+jointLowerLimit[i])/2.0

    # sign = - sign
    # # target_joint[1] = 5 * sign
    # target_joint[2] = -5 * sign
    # # target_joint[4] = 5 * sign
    # target_joint[5] = -5 * sign
    # # target_joint[7] = 5 * sign 
    # target_joint[8] = -5 * sign 
    # # target_joint[10] = 5 * sign
    # target_joint[11] = -5 * sign
    # # print("Nuevo Target")

      state = ACTUATION
      repeat = 1

  elif state == ACTUATION:
    step_complete = State_Machine_Actuation()

    # if (pitch*180/np.pi >= critical_failure_angle or roll*180/np.pi >= critical_failure_angle):
    #     supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    #     state = TX_RASPBERRY

    if step_complete == 1:
      state = TX_RASPBERRY


    elif step_complete == -1:
      state = TX_RASPBERRY  # ESTADO STOP DE EMERGENCIA EN EL FUTURO

def State_Machine_Actuation():
  global state_actuation,joint,joint_sensor
  global f_joint_angle,timeout,joints_finished
  global delta_sample,target_joint,delta_target,joint
  global step_omitted,enable_torque_control, pid_sample, step_complete
  global repeat

  if state_actuation == RESET_ACTUATION:
    # Check if the new target is too close to the actual position, in which case the servo wouldn't move because of the dead bandwidth
    # Therefore consider the joint already finished
    
    for i in range(len(joint)):      

      delta_target = np.abs(f_joint_angle[i] - target_joint[i])
      if delta_target >= DEAD_BANDWIDTH_SERVO:
        joints_finished &= ~(1 << i)  # Set respective bit in 0
        #! EN RADIANES HAY QUE PONERLO!!!!//Move the servomotor
        #joint[i].setPosition(target_joint[i]+np.random.uniform(-JOINT_MAX_NOISE,JOINT_MAX_NOISE))
             

    # Start first measure of joints before ACTUATION

    if joints_finished == ALL_FINISHED :  # Si ninguna articulación se tiene que mover, salteo la actuacion
      print("Salteo paso")
      step_omitted = 1
      repeat  = 0
      return 1
    else:
      state_actuation = DELAY_STATE
      enable_torque_control = 1 
      timeout = TIMEOUT
      pid_sample = 0  
      step_complete = 0
      repeat = 1 
      
		
  elif state_actuation == DELAY_STATE:
    if timeout == 0:
      state_actuation = TIMEOUT_STATE
      repeat = 1
      
    elif pid_sample == 1:    
      state_actuation = COMPARE_MEASURE
      repeat = 1
    
  elif state_actuation == COMPARE_MEASURE:
    if(timeout == 0):
      state_actuation = TIMEOUT_STATE
      repeat  = 1

    else:
      for i in range(len(joint)):

        if (joints_finished & (1 << i)) == 0:  # Ignore joints that finished
          
          delta_target = np.abs(f_joint_angle[i] - target_joint[i])

          if delta_target < MAX_DELTA_ANGLE:  # If the stable joint reached target

            joints_finished |= 1 << i  # Set respective bit in 1
            if joints_finished == ALL_FINISHED:

              #enable_torque_control = 0
              state_actuation = RESET_ACTUATION
              repeat  = 1
              return 1
          #else:
            # timeout = TIMEOUT

      pid_sample = 0
      state_actuation = DELAY_STATE		
      repeat = 1	

  elif state_actuation == TIMEOUT_STATE:
    print("Step Aborted")
    print("Joints finished",bin(joints_finished))
    print("current_time", current_time)
    #enable_torque_control = 0
    state_actuation = RESET_ACTUATION
    repeat  = 1
    return -1
  
  return 0

def computeMeasurementsForRewards():
  global supervisor,root,forward_acceleration,forward_velocity,lateral_velocity,mean_forward_velocity,mean_lateral_velocity
  global lateral_acceleration,max_forward_acc,prev_time,prev_velocity_magnitude,prev_forward_velocity,prev_lateral_velocity,vel_samples
  global step_counter,joint,current_position,prev_joint_angular_position_servo,paw_transitions, transition_phase, paw_previous_state
  global prev_error_servo, acum_error_servo

  ## Velocity of Robot
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

  # velocity_magnitude = np.linalg.norm(world_velocity)

  ##Absolute Velocity of PAWs
  PFR_velocity = PFR_node.getVelocity()
  PFL_velocity = PFL_node.getVelocity()
  PBR_velocity = PBR_node.getVelocity()
  PBL_velocity = PBL_node.getVelocity()

  PFR_velocity = np.sqrt(PFR_velocity[0]**2 + PFR_velocity[1]**2 + PFR_velocity[2]**2)
  PFL_velocity = np.sqrt(PFL_velocity[0]**2 + PFL_velocity[1]**2 + PFL_velocity[2]**2)
  PBR_velocity = np.sqrt(PBR_velocity[0]**2 + PBR_velocity[1]**2 + PBR_velocity[2]**2)
  PBL_velocity = np.sqrt(PBL_velocity[0]**2 + PBL_velocity[1]**2 + PBL_velocity[2]**2)

  time = supervisor.getTime()

  if time > 0.0:
    total_acceleration = (velocity_magnitude - prev_velocity_magnitude)/(time - prev_time)
    # forward_acceleration = (forward_velocity - prev_forward_velocity)/(time - prev_time)
    # lateral_acceleration = (lateral_velocity - prev_lateral_velocity)/(time - prev_time)

    if np.abs(forward_acceleration) > max_forward_acc:
        max_forward_acc = np.abs(forward_acceleration)

  # prev_velocity_magnitude = velocity_magnitude
  prev_forward_velocity = forward_velocity
  # prev_lateral_velocity = lateral_velocity

  # Compute mean in a recursive manner
  mean_forward_velocity = 1 / (vel_samples + 1.0) * (mean_forward_velocity * vel_samples + forward_velocity)
  mean_lateral_velocity = 1 / (vel_samples + 1.0) * (mean_lateral_velocity * vel_samples + lateral_velocity)  

  vel_samples = vel_samples + 1.0
  prev_time = time

  PFR_state = FR_touch_sensor.getValue()
  PFL_state = FL_touch_sensor.getValue()
  PBR_state = BR_touch_sensor.getValue()
  PBL_state = BL_touch_sensor.getValue()

  if paw_transition_enable[0] and (paw_previous_state[0] != PBL_state):
    paw_transitions[0] = PBL_state + 1
    transition_phase[0] = current_phase
    paw_transition_enable[0] = False
    # print("BL transition = ", paw_transitions[0])
    # print("current_time = ", current_time)
  
  if paw_transition_enable[1] and (paw_previous_state[1] != PFL_state):
    paw_transitions[1] = PFL_state + 1
    transition_phase[1] = current_phase
    paw_transition_enable[1] = False
    # print("FL transition = ", paw_transitions[1])
    # print("current_time = ", current_time)

  if paw_transition_enable[2] and (paw_previous_state[2] != PBR_state):
    paw_transitions[2] = PBR_state + 1
    transition_phase[2] = current_phase
    paw_transition_enable[2] = False
    # print("BR transition = ", paw_transitions[2])
    # print("current_time = ", current_time)
  
  if paw_transition_enable[3] and (paw_previous_state[3] != PFR_state):
    paw_transitions[3] = PFR_state + 1
    transition_phase[3] = current_phase
    paw_transition_enable[3] = False
    # print("FR transition = ", paw_transitions[3])
    # print("current_time = ", current_time)
  
  paw_previous_state = [PBL_state, PFL_state, PBR_state, PFR_state]


def resetMeasurementsForRewards():
  global step_counter, mean_forward_velocity, mean_lateral_velocity, max_forward_acc, vel_samples, prev_error, acum_error
  global prev_error_servo, acum_error_servo, paw_transitions, transition_phase, paw_previous_state, paw_transition_enable
  step_counter = step_counter + 1

  mean_forward_velocity = 0.0
  mean_lateral_velocity = 0.0

  max_forward_acc = 0.0

  vel_samples = 0.0

  prev_error= np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
  acum_error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32)

  prev_error_servo= np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
  acum_error_servo = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32)

  paw_transitions = np.zeros(4,dtype=np.float32)
  transition_phase = np.zeros(4,dtype=np.float32)
  
  paw_previous_state = [BL_touch_sensor.getValue(), FL_touch_sensor.getValue(), BR_touch_sensor.getValue(), FR_touch_sensor.getValue()]
  paw_transition_enable = [True, True, True, True]

def computeTorques(target_servo):
    global joint, joint_angular_velocity_servo, joint_sensor
    global prev_joint_angular_position_servo, joint_torque, error_servo, delta_error_servo, acum_error_servo
    global prev_error_servo, Ia, enable_torque_control

    # print("New Simulation Step")

    if enable_torque_control == 0:
       return
    
    internal_angle_servo = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(joint)):        
      internal_angle_servo[i] = joint_sensor[i].getValue()
      joint_angular_velocity_servo[i] = (internal_angle_servo[i] - prev_joint_angular_position_servo[i]) / (timestep * 1e-3)
      prev_joint_angular_position_servo[i] = internal_angle_servo[i]

    # PID control
    error_servo = target_servo - internal_angle_servo
    delta_error_servo = (error_servo - prev_error_servo) / (timestep * 1e-3)
    acum_error_servo += error_servo * timestep * 1e-3
    prev_error_servo = error_servo

    # Voltage and current calculations
    VA = KP_servo * error_servo + KD_servo * delta_error_servo + KI_SERVO * acum_error_servo
    VA = np.clip(VA, -8.4, 8.4)

    ####### Not used, only useful to know real back emf, current and torque #######
    Eb = joint_angular_velocity_servo * KM
    Ia = (VA - Eb) / RA
    joint_torque = Ia * KM
    ###############################################################################

    # Apply torque to joints
    for i in range(len(joint)):
      #Make sure the constant friction always opposes the motor's torque
      if VA[i] < 0:
        friction_direction = -1
      else:
        friction_direction = +1

      joint[i].setTorque(VA[i]*KM/RA - friction_direction*TF)
      #After computing the DC Motor Control, add noise for the rest of the controller 
      #f_joint_angle[i] = f_joint_angle[i]+ np.random.normal(0,0.5,None)  #!(Optional noise)

pid_out = 0

def PIDControl():
  global joint, joint_angular_velocity, joint_sensor, f_joint_angle
  global prev_joint_angular_position, error, delta_error, acum_error
  global prev_error
  global pid_sample,step_complete
  global jointUpperLimit,jointLowerLimit
  global delay_buffer
  global pid_out
  
  
    
  # if step_complete == 0:

  error = target_joint - f_joint_angle
  delta_error = (error - prev_error) / (timestep * 1e-3)
  acum_error += error * timestep * 1e-3
  prev_error = error

  control_signal = np.round(KP * error + KD * delta_error + KI * acum_error) / K_TON

  pid_out = pid_out + control_signal
      
  out_of_max_range = pid_out > jointUpperLimit

  out_of_min_range = pid_out < jointLowerLimit

  # if out_of_max_range.any() or out_of_min_range.any():
    # print("---------------")
    # print("fuera de rango maximo: ", out_of_max_range)
    # print("fuera de rango minimo: ", out_of_min_range)
    # print("target_joints = ", target_joint)
    # print("control_signal = ", control_signal)
    # print("f_joint_angle = ", f_joint_angle)
    
  #If any joint steps out of the range, and the PID
  pid_out = pid_out * (1 - (out_of_max_range+out_of_min_range)) + jointUpperLimit * out_of_max_range + jointLowerLimit * out_of_min_range
  
  # print("pid_out = ", pid_out)

  

if __name__ == "__main__":
  repeat = 1

  current_time = 0
  current_phase = 0

  GAIT_PERIOD = 2 #sec

  pid_timer = 0
  median_filter_delay = 0

  # debug_delay = 0
  # Main loop:
  while supervisor.step(timestep) != -1:

    current_time = supervisor.getTime()

    #Instantaneous phase, normalized from 0 to 1. Defines the cycle of each paw to walk
    current_phase = (current_time % GAIT_PERIOD) / GAIT_PERIOD

    if median_filter_delay > 0:
      median_filter_delay-=timestep
    if median_filter_delay == 0:
      for i in range(len(joint)):
        delay_buffer[i][1:]=  delay_buffer[i][0:-1] 
        delay_buffer[i][0] = joint_sensor[i].getValue() * 180/np.pi
      median_filter_delay = timestep

    if pid_timer > 0:
      pid_timer-=timestep    
    if pid_timer == 0:
      for i in range(len(joint)):            
        f_joint_angle[i] = delay_buffer[i][-1]
      PIDControl()
      pid_sample = 1
      pid_timer = 10

    if timeout >0: 
      timeout-=timestep

    # if debug_delay > 0:
    #   debug_delay-=timestep

    computeTorques(pid_out*np.pi/180)

    computeMeasurementsForRewards()

    # print("Debug Delay",debug_delay)

    Kalman_filter()

    attitude[0],attitude[1] = pitch,roll
    ang_vel[0],ang_vel[1] = vel_pitch,vel_roll   

    while repeat:
      repeat = 0
      State_Machine_Control()
    repeat = 1

    # To record joints in a csv
    # row = [current_time]  # Start with the timestamp

    # # Joints
    # for i in range(12):
    #   row.append(target_joint[i])
    #   row.append(f_joint_angle[i])
    #   row.append(pid_out[i])

    # Touch sensors
    # row.append(int(BL_touch_sensor.getValue() == True))
    # row.append(int(FL_touch_sensor.getValue() == True))
    # row.append(int(BR_touch_sensor.getValue() == True))
    # row.append(int(FR_touch_sensor.getValue() == True))

    # #target touch values
    # # BL 
    # if current_phase > 0.1 and current_phase < 0.3:
    #   row.append(0)
    # else:
    #   row.append(1)

    # # FL 
    # if current_phase > 0.35 and current_phase < 0.55:
    #   row.append(0)
    # else:
    #   row.append(1)
    
    # # BR 
    # if current_phase > 0.6 and current_phase < 0.8:
    #   row.append(0)
    # else:
    #   row.append(1)

    # # FR 
    # if current_phase > 0.85 or current_phase < 0.05:
    #   row.append(0)
    # else:
    #   row.append(1)


    # Append the row to the data list
    # data.append(row)
    # if current_time >= 25:
    #   # Write all collected data to CSV
    #   with open(output_file, "w", newline="") as csvfile:
    #       writer = csv.writer(csvfile)
    #       writer.writerow(header)  # Write header
    #       writer.writerows(data)  # Write all rows
    #   print(f"Data saved to {output_file}")
    #   break  

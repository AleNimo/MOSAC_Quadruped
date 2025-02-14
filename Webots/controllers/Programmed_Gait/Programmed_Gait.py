"""extremidad_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Accelerometer
from controller import Supervisor
import numpy as np

import csv

from scipy.spatial.transform import Rotation

#*PROGRAMMED GAIT CONSTANTS
NUM_STAGES = 16

STANDING_HEIGHT = 160 #mm

GAIT_PERIOD = 16

PARABOLA_HEIGHT = 40 #mm
PARABOLA_LENGTH = 100 #mm

L_FEMUR = 100 #mm
L_TIBIA = 100.70866 #mm
PSI = 2.51591*np.pi/180 #rad

A = 24 #mm
B = 40.5 #mm
C = 28.66176 #mm
D = 27 #mm

H = np.zeros(4,dtype=np.float32) + STANDING_HEIGHT
# D1 =  np.zeros(4,dtype=np.float32)
D1 = -np.array([1,-2,-1,0],dtype=np.float32)*PARABOLA_LENGTH/4

BL = 0
FL = 1
BR = 2
FR = 3

phi_femur_servo = np.zeros(4,dtype=np.float32)
phi_tibia_servo = np.zeros(4,dtype=np.float32)

DELTA = 0.7984834029 #rad
EPSILON = 1.570796327 #rad

MID_POINT_SOLID_FEMUR = 42.65021 #deg
MID_POINT_SOLID_TIBIA = 16.75397157 #deg

supervisor = Supervisor()
root = supervisor.getSelf()
robot_node = supervisor.getFromDef('MOSAC')

reset_pos = np.array([0.0, 0.0, 0.141])  

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

target_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])*np.pi/180

# CSV output filename
output_file = "webots_joint_data.csv"

# Initialize header
header = ["timestamp"]

header.append(f"BFR_Target")
header.append(f"BFR_Filt")
header.append(f"FFR_Target")
header.append(f"FFR_Filt")
header.append(f"TFR_Target")
header.append(f"TFR_Filt")

header.append(f"BFL_Target")
header.append(f"BFL_Filt")
header.append(f"FFL_Target")
header.append(f"FFL_Filt")
header.append(f"TFL_Target")
header.append(f"TFL_Filt")

header.append(f"BBR_Target")
header.append(f"BBR_Filt")
header.append(f"FBR_Target")
header.append(f"FBR_Filt")
header.append(f"TBR_Target")
header.append(f"TBR_Filt")

header.append(f"BBL_Target")
header.append(f"BBL_Filt")
header.append(f"FBL_Target")
header.append(f"FBL_Filt")
header.append(f"TBL_Target")
header.append(f"TBL_Filt")

joint_targets = [[] for _ in range(12)]
joint_filtereds = [[] for _ in range(12)]


data = [] 
current_time = 0
t = 0
phase = 0

def computeAngles(H,D1):

  for joint in range(4):
    #*Compute Femur and tibia angles
    D2_sqrd = H[joint]**2 + D1[joint]**2
    D2 = np.sqrt(D2_sqrd)
    
    phi_femur = np.pi/2 - np.arctan(D1[joint]/H[joint]) - np.arccos((D2_sqrd + L_FEMUR*L_FEMUR-L_TIBIA*L_TIBIA)/(2*L_FEMUR*D2))
    phi_tibia = np.pi/2 + np.arctan(D1[joint]/H[joint]) - np.arccos((D2_sqrd + L_TIBIA*L_TIBIA-L_FEMUR*L_FEMUR)/(2*L_TIBIA*D2))
    
    phi_tibia = phi_tibia + PSI

    #*Translate tibia angle to tibia servo angle
    W_sqrd = C**2 + D**2 - 2*C*D*np.cos(np.pi+DELTA-EPSILON-phi_tibia)
    W = np.sqrt(W_sqrd)

    phi_tibia_servo[joint] = np.pi - DELTA - np.arccos((C**2 + W_sqrd - D**2) / (2*C*W)) - np.arccos((A**2 + W_sqrd - B**2) / (2*A*W))

    #*Translate angles used in kinematics equations to calibrated servo values
    phi_femur_servo[joint] = -phi_femur + MID_POINT_SOLID_FEMUR*np.pi/180
    phi_tibia_servo[joint] = -phi_tibia_servo[joint] + MID_POINT_SOLID_TIBIA*np.pi/180

    
  BL_servo_femur_joint.setPosition(-phi_femur_servo[BL])
  BL_servo_tibia_joint.setPosition(-phi_tibia_servo[BL])

  FL_servo_femur_joint.setPosition(-phi_femur_servo[FL])
  FL_servo_tibia_joint.setPosition(-phi_tibia_servo[FL])

  BR_servo_femur_joint.setPosition(-phi_femur_servo[BR])
  BR_servo_tibia_joint.setPosition(-phi_tibia_servo[BR])
  
  FR_servo_femur_joint.setPosition(-phi_femur_servo[FR])
  FR_servo_tibia_joint.setPosition(-phi_tibia_servo[FR])

def prepare_for_parabola(primary_limb, secondary_limb, initial_phase):
  H[primary_limb] = STANDING_HEIGHT - 20/(1/NUM_STAGES) * (phase - initial_phase)
  H[secondary_limb] = STANDING_HEIGHT - 10/(1/NUM_STAGES) * (phase - initial_phase)

def parabola(limb, initial_phase):
  H[limb] = STANDING_HEIGHT - PARABOLA_HEIGHT * ( 1 - 4*(-(phase-initial_phase)/(1/NUM_STAGES)+0.5)**2 )
  D1[limb] = PARABOLA_LENGTH/2 - (phase-initial_phase) * PARABOLA_LENGTH/(1/NUM_STAGES)

def restore_assisting_limbs(primary_limb, secondary_limb, initial_phase):
  H[primary_limb] = STANDING_HEIGHT - 20 + 20/(1/NUM_STAGES) * (phase-initial_phase)
  H[secondary_limb] = STANDING_HEIGHT - 10 + 10/(1/NUM_STAGES) * (phase-initial_phase)

def move_forward():
  D1[BL] += (timestep*1e-3/GAIT_PERIOD) * (PARABOLA_LENGTH/4)/(1/NUM_STAGES)
  D1[FL] += (timestep*1e-3/GAIT_PERIOD) * (PARABOLA_LENGTH/4)/(1/NUM_STAGES)
  D1[BR] += (timestep*1e-3/GAIT_PERIOD) * (PARABOLA_LENGTH/4)/(1/NUM_STAGES)
  D1[FR] += (timestep*1e-3/GAIT_PERIOD) * (PARABOLA_LENGTH/4)/(1/NUM_STAGES)

if __name__ == "__main__":
    
    #Init position
    computeAngles(H,D1)

    delay_init = 2

    while supervisor.step(timestep) != -1:

      if delay_init > 0:
        delay_init-=timestep*1e-3 
        continue

      # print("Start")
      phase = (t % GAIT_PERIOD)/GAIT_PERIOD
        
      #*Compute H and D1 (0-BL, 1-FL, 2-BR, 3-FR)
      #Stage 0
      if phase >= 0 and phase < 1/NUM_STAGES:
        prepare_for_parabola(BR,BL,0)
      
      #Stage 1
      if phase >= 1/NUM_STAGES and phase < 2/NUM_STAGES:
        parabola(FL, 1/NUM_STAGES)
      
      #Stage 2
      if phase >= 2/NUM_STAGES and phase < 3/NUM_STAGES:
        restore_assisting_limbs(BR,BL,2/NUM_STAGES)

      #Stage 3
      if phase >= 3/NUM_STAGES and phase < 4/NUM_STAGES:
        move_forward()

      #Stage 4
      if phase >= 4/NUM_STAGES and phase < 5/NUM_STAGES:
        prepare_for_parabola(FL,FR,4/NUM_STAGES)

      #Stage 5
      if phase >= 5/NUM_STAGES and phase < 6/NUM_STAGES:
        #Parabola forward BR
        parabola(BR, 5/NUM_STAGES)
      
      #Stage 6
      if phase >= 6/NUM_STAGES and phase < 7/NUM_STAGES:
        #Restore FR and FL
        restore_assisting_limbs(FL,FR,6/NUM_STAGES)
      
      #Stage 7
      if phase >= 7/NUM_STAGES and phase < 8/NUM_STAGES:
        move_forward()

      #Stage 8
      if phase >= 8/NUM_STAGES and phase < 9/NUM_STAGES:
        #Lift BL (BR assists)
        prepare_for_parabola(BL,BR,8/NUM_STAGES)
      
      #Stage 9
      if phase >= 9/NUM_STAGES and phase < 10/NUM_STAGES:
        #Parabola forward FR
        parabola(FR, 9/NUM_STAGES)
      
      #Stage 10
      if phase >= 10/NUM_STAGES and phase < 11/NUM_STAGES:
        #Restore BL and BR
        restore_assisting_limbs(BL,BR,10/NUM_STAGES)
      
      #Stage 11
      if phase >= 11/NUM_STAGES and phase < 12/NUM_STAGES:
        move_forward()

      #Stage 12
      if phase >= 12/NUM_STAGES and phase < 13/NUM_STAGES:
        #Lift FR (FL assists)
        prepare_for_parabola(FR,FL,12/NUM_STAGES)

      #Stage 13
      if phase >= 13/NUM_STAGES and phase < 14/NUM_STAGES:
        #Parabola forward BL
        parabola(BL, 13/NUM_STAGES)
      
      #Stage 14
      if phase >= 14/NUM_STAGES and phase < 15/NUM_STAGES:
        #Restore FR and FL
        restore_assisting_limbs(FR,FL,14/NUM_STAGES)

      #Stage 15
      if phase >= 15/NUM_STAGES and phase < 16/NUM_STAGES:
        move_forward()

      t += timestep * 1e-3

      computeAngles(H,D1)

      print("H[FL]" , H[FL])
      print("D1[FL]" ,D1[FL])
      
      print("servo_femur" , phi_femur_servo[FL]*180/np.pi)
      print("servo_tibia" , phi_tibia_servo[FL]*180/np.pi)

        
      # Simulate joint data (replace with actual data collection from your simulation)
      # current_time = supervisor.getTime()  # Convert seconds to milliseconds
      # row = [current_time]  # Start with the timestamp
      # for i in range(12):
      #   row.append(target_joint[i] * 180 / np.pi)
      #   # row.append(joint_angular_position[i] * 180 / np.pi+np.random.normal(0,0.5,None))
      #   row.append(joint_angular_position[i] * 180 / np.pi)
      # # Append the row to the data list
      # data.append(row)

      # if current_time >= 10.0:
      #   # Write all collected data to CSV
      #   with open(output_file, "w", newline="") as csvfile:
      #       writer = csv.writer(csvfile)
      #       writer.writerow(header)  # Write header
      #       writer.writerows(data)  # Write all rows
      #   print(f"Data saved to {output_file}")
      #   break        
        
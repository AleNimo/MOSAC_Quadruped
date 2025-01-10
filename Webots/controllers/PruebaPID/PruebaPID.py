"""extremidad_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Accelerometer
from controller import Supervisor
import numpy as np

import csv

from scipy.spatial.transform import Rotation




# slider = False
programmed_target_rotation = False



supervisor = Supervisor()
root = supervisor.getSelf()
robot_node = supervisor.getFromDef('MOSAC')



reset_pos = np.array([0.0, 0.0, 0.15])  

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





joint_angular_position = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
prev_joint_angular_position = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
joint_angular_velocity = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
joint_torque = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
prev_error= np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
delta_error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
acum_error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
Ia = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=np.float32) 
enable_torque_control = 0

KP = 18.847
KD = 0.023541
KI = 2.0231
KM = 0.9853919147
RA = 1.45
TF = 0.03262365753

target_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])*np.pi/180

# for i in range(len(joint)):
#       #! EN RADIANES HAY QUE PONERLO!!!!//Move the servomotor
#       joint[i].setPosition(float('inf'))
#       joint[i].setVelocity(0)
#       #joint[i].setTorque(0)
print_counter = 0
def computeTorques():
    global joint, joint_angular_velocity, joint_sensor, joint_angular_position
    global prev_joint_angular_position, joint_torque, error, delta_error, acum_error
    global prev_error, Ia, enable_torque_control
    # print("New Simulation Step")
   
    MAX_TORQUE = 3.43  # Limit torque to the motor's max capacity
    DAMPING_FACTOR = 0  # Optional damping
    ALPHA = 0  # Low-pass filter for velocity

    for i in range(len(joint)):
        # Get position and compute velocity with smoothing
        joint_angular_position[i] = joint_sensor[i].getValue()
        # print("Angular Position %d: %f" % (i, joint_angular_position[i]*180/np.pi))
        # print("Previous Angular Position %d: %f" % (i,prev_joint_angular_position[i]*180/np.pi))
        raw_velocity = (joint_angular_position[i] - prev_joint_angular_position[i]) / (timestep * 1e-3)
        joint_angular_velocity[i] = ALPHA * joint_angular_velocity[i] + (1 - ALPHA) * raw_velocity
        prev_joint_angular_position[i] = joint_angular_position[i]

    # PID control
    error = target_joint - joint_angular_position
    delta_error = (error - prev_error) / (timestep * 1e-3)
    acum_error += error * timestep * 1e-3
    prev_error = error
    
    # Voltage and current calculations
    VA = KP * error + KD * delta_error + KI * acum_error

    # for i in range(len(joint)):
    #   if abs(VA[i]) > 7.4:
    #     VA[i] = sign(VA[i]) * 7.4

    Eb = joint_angular_velocity * KM
    Ia = (VA - Eb) / RA
    
    # Torque calculation
    joint_torque = Ia * KM
    joint_torque = np.clip(joint_torque - DAMPING_FACTOR * joint_angular_velocity, -MAX_TORQUE, MAX_TORQUE)

    if print_counter%100 == 0:
      print("VA", VA)
      print("Error", error*180/np.pi)
    # print("Eb", Eb)
    
    # print("Omega", joint_angular_velocity*180/np.pi)  
    # print("Ia", Ia)
    # print("joint_torque", joint_torque)
    # Apply torque to joints
    
    for i in range(len(joint)):
        # if np.abs(VA[i]) >500e-3:
        joint[i].setTorque(VA[i]*KM/RA - TF)
        # else:
          # joint[i].setTorque(0)
        #print("Torque %d: %f" %(i,joint[i].getTorqueFeedback()))



t = 0
for i in range(len(joint)):
  joint[i].setTorque(0)

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
sign = 1

if __name__ == "__main__":
    
   
    while supervisor.step(timestep) != -1:
      current_time = supervisor.getTime()  # Convert seconds to milliseconds
      print_counter+=1

      t = t + timestep *1e-3


      # angle_tibia = 10 * np.sin(2 * np.pi * 0.5 * t * 1e-3)
      # angle_femur = 10 * np.sin(2 * np.pi * 0.5 * t * 1e-3)

      # target_joint[1] = angle_femur * np.pi / 180
      # target_joint[2] = -angle_tibia * np.pi / 180
      # target_joint[4] = angle_femur * np.pi / 180
      # target_joint[5] = -angle_tibia * np.pi / 180
      # target_joint[7] = angle_femur * np.pi / 180
      # target_joint[8] = -angle_tibia * np.pi / 180
      # target_joint[10] = angle_femur * np.pi / 180
      # target_joint[11] = -angle_tibia * np.pi / 180
    
         
      if(t >=1.0):
      
        sign = - sign
        t = 0.0
        target_joint[1] = 5 * sign* np.pi / 180
        target_joint[2] = -5 * sign * np.pi / 180
        target_joint[4] = 5 * sign* np.pi / 180
        target_joint[5] = -5 * sign * np.pi / 180
        target_joint[7] = 5 * sign * np.pi / 180
        target_joint[8] = -5 * sign * np.pi / 180
        target_joint[10] = 5 * sign* np.pi / 180
        target_joint[11] = -5 * sign* np.pi / 180
      
          
      computeTorques()
      for i in range(12):
        joint_targets[i].append(target_joint[i])  
        joint_filtereds[i].append(joint_angular_position[i]) 

      # Simulate joint data (replace with actual data collection from your simulation)
      row = [current_time]  # Start with the timestamp
      for i in range(12):
        row.append(target_joint[i] * 180 / np.pi)
        # row.append(joint_angular_position[i] * 180 / np.pi+np.random.normal(0,0.5,None))
        row.append(joint_angular_position[i] * 180 / np.pi)
      # Append the row to the data list
      data.append(row)

      if current_time >= 10.0:
        # Write all collected data to CSV
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write header
            writer.writerows(data)  # Write all rows
        print(f"Data saved to {output_file}")
        break        
        
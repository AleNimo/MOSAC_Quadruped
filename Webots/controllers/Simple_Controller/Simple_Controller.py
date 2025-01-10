"""extremidad_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Supervisor
import numpy as np

# get the time step of the current world.
timestep = 10

supervisor = Supervisor()

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

if __name__ == "__main__":

    t = 0

    FL_servo_tibia_joint.setPosition(0)
    FL_servo_femur_joint.setPosition(0)

    BR_servo_tibia_joint.setPosition(0)
    BR_servo_femur_joint.setPosition(0)

    BL_servo_tibia_joint.setPosition(0)
    BL_servo_femur_joint.setPosition(0)

    FR_servo_tibia_joint.setPosition(0)
    FR_servo_femur_joint.setPosition(0)

    while supervisor.step(timestep) != -1:

      t = t + timestep

      # Read the sensors:

      angle_tibia = 10 * np.sin(2 * np.pi * 0.5 * t * 1e-3)
      angle_femur = 10 * np.sin(2 * np.pi * 0.5 * t * 1e-3)
      FL_servo_tibia_joint.setPosition(-angle_tibia * np.pi / 180)
      FL_servo_femur_joint.setPosition(angle_femur * np.pi / 180)

      BR_servo_tibia_joint.setPosition(-angle_tibia * np.pi / 180)
      BR_servo_femur_joint.setPosition(angle_femur * np.pi / 180)

      BL_servo_tibia_joint.setPosition(-angle_tibia * np.pi / 180)
      BL_servo_femur_joint.setPosition(angle_femur * np.pi / 180)

      FR_servo_tibia_joint.setPosition(-angle_tibia * np.pi / 180)
      FR_servo_femur_joint.setPosition(angle_femur * np.pi / 180)
      
      pass
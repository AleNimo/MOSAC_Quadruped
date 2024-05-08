#python
import numpy as np

def sysCall_init():
    sim = require('sim')

    self.servo_handle = sim.getObject("/servo")

    #Potentiometer parameters
    self.k_pot = 180/(69.23 * np.pi)
    self.bias_pot = -22.15 * np.pi / 180

    #DC motor parameters
    self.Ra = 0.2
    
    self.kb = 0.4
    self.kt = 0.4

    self.max_vel = 5.51     #rad/seg
    self.max_torque = 0.402 #N*m

    #Plotting variables
    self.torque_graph = sim.getObject('/torque_graph')
    self.torque_stream = sim.addGraphStream(self.torque_graph, 'torque', 'N*m', 0, [1, 0, 0])

    self.current_graph = sim.getObject('/current_graph')
    self.current_stream = sim.addGraphStream(self.current_graph, 'current', 'A', 0, [1, 0, 0])

    self.vel_graph = sim.getObject('/vel_graph')
    self.vel_stream = sim.addGraphStream(self.vel_graph, 'vel', 'm/s', 0, [1, 0, 0])

    self.pos_graph = sim.getObject('/pos_graph')
    self.pos_stream = sim.addGraphStream(self.pos_graph, 'ang', 'deg', 0, [1, 0, 0])
    self.target_pos_stream = sim.addGraphStream(self.pos_graph, 'target_ang', 'deg', 0, [0, 1, 0])

    self.current = 0
    self.torque = 0
    self.vel = 0
    self.pos = 0
    self.target_pos = 0

def sysCall_joint(inData):
    # inData['mode'] : sim.jointmode_kinematic or sim.jointmode_dynamic
    # inData['handle'] : the handle of the joint to control
    # inData['revolute'] : whether the joint is revolute or prismatic
    # inData['cyclic'] : whether the joint is cyclic or not
    # inData['lowLimit'] : the lower limit of the joint (if the joint is not cyclic)
    # inData['highLimit'] : the higher limit of the joint (if the joint is not cyclic)
    # inData['dt'] : the step size used for the calculations
    # inData['pos'] : the current position
    # inData['vel'] : the current velocity
    # inData['targetPos'] : the desired position (if joint is dynamic, or when sim.setJointTargetPosition was called)
    # inData['targetVel'] : the desired velocity (if joint is dynamic, or when sim.setJointTargetVelocity was called)
    # inData['initVel'] : the desired initial velocity (if joint is kinematic and when sim.setJointTargetVelocity
    #                  was called with a 4th argument)
    # inData['error'] : targetPos-currentPos (with revolute cyclic joints, the shortest cyclic distance)
    # inData['maxVel'] : a maximum velocity
    # inData['maxAccel'] : a maximum acceleration
    # inData['maxJerk'] : a maximum jerk
    # inData['first'] : whether this is the first call from the physics engine, since the joint
    #                was initialized (or re-initialized) in it.
    # inData['passCnt'] : the current dynamics calculation pass. 1-10 by default
    # inData['rk4pass'] : if Runge-Kutta 4 solver is selected, will loop from 1 to 4 for each inData['passCnt']
    # inData['totalPasses'] : the number of dynamics calculation passes for each "regular" simulation pass.
    # inData['effort'] : the last force or torque that acted on this joint along/around its axis. With Bullet,
    #                 torques from joint limits are not taken into account
    # inData['force'] : the joint force/torque, as set via sim.setJointTargetForce

    if inData['mode'] == sim.jointmode_dynamic:

        self.vel = inData['vel']
        self.pos = inData['pos']
        self.target_pos = inData['targetPos']

        #Translate angles to voltage with potentiometer and references
        v_ref = self.k_pot * (self.target_pos - self.bias_pot)
        v_pot = self.k_pot * (self.pos - self.bias_pot)

        #Subtract to have negative feedback
        vx = v_ref - v_pot

        #Compute back emf on the rotor based on velocity from the physics engine
        v_eb = self.kb * self.vel

        #Compute current in the rotor
        self.current = (vx - v_eb) / self.Ra

        #Compute the resulting torque based on the rotor current
        self.torque = self.kt * self.current
        
        #Clamp torque using max value from the datasheet
        if self.torque > self.max_torque: self.torque = self.max_torque
        if self.torque < -self.max_torque: self.torque = -self.max_torque
        
        #Select sign of the max velocity to match the sign of the torque
        if self.torque >= 0: vel = self.max_vel
        else: vel = -self.max_vel
        
        # Following data must be returned to CoppeliaSim:

        # Both velocity and torque are limits for the physics engine. Its analog to the voltage and current of a DC source with current limitation:
        #   * if the velocity limit is not yet reached the torque is equal to the one being set (the limit, in order to try to reach the velocity)
        #   * if the velocity limit is reached, the torque is reduced (less than the limit set, in order to maintain the velocity)

        # To mimic a real motor where the only output is the torque (and velocity is a consequence), 
        # We always set the velocity to be a high limit matching the sign of the torque (using the max value from the datasheet), so the torque is equal to the limit being set
        outData = {'vel': vel, 'force': self.torque}
        return outData
    
    # Expected return data:
    # For kinematic joints:
    # outData = {'pos': pos, 'vel': vel, 'immobile': False}
    # 
    # For dynamic joints:
    # outData = {'force': f, 'vel': vel}

def sysCall_actuation():
    # put your actuation code here
    pass

def sysCall_sensing():
    # put your sensing code here
    sim.setGraphStreamValue(self.torque_graph, self.torque_stream, self.torque)
    sim.setGraphStreamValue(self.current_graph, self.current_stream, self.current)
    sim.setGraphStreamValue(self.vel_graph, self.vel_stream, self.vel)
    sim.setGraphStreamValue(self.pos_graph, self.pos_stream, self.pos * 180 / np.pi)
    sim.setGraphStreamValue(self.pos_graph, self.target_pos_stream, self.target_pos * 180 / np.pi)
    pass

def sysCall_cleanup():
    # do some clean-up here
    pass

# See the user manual or the available code snippets for additional callback functions and details

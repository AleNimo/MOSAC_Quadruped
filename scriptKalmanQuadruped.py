#python
import numpy as np
from collections import deque

def sysCall_init():
    sim = require('sim')

    self.acc_std = 0.00001#0.002 #Gs
    self.gyro_std = 0.00001#1.2 #°/s

    #Graphs
    # self.accel_graph = sim.getObject('/Acc_graph')
    # self.gyro_graph = sim.getObject('/Gyr_graph')

    # self.accel_filt_graph = sim.getObject('/Acc_filt_graph')
    # self.gyro_filt_graph = sim.getObject('/Gyr_filt_graph')

    self.kalman_graph = sim.getObject('/Kalman_graph')
    # self.accel_corr_graph = sim.getObject('/Acc_corr_graph')

    # self.vel_graph = sim.getObject('/Vel_graph')

    # self.x_accel = sim.addGraphStream(self.accel_graph, 'x_accel', 'm/s^2', 0, [1, 0, 0])
    # self.y_accel = sim.addGraphStream(self.accel_graph, 'y_accel', 'm/s^2', 0, [0, 1, 0])
    # self.z_accel = sim.addGraphStream(self.accel_graph, 'z_accel', 'm/s^2', 0, [0, 0, 1])

    # self.x_ang_vel = sim.addGraphStream(self.gyro_graph, 'x_ang_vel', 'rad/s', 0, [1, 0, 0])
    # self.y_ang_vel = sim.addGraphStream(self.gyro_graph, 'y_ang_vel', 'rad/s', 0, [0, 1, 0])
    # self.z_ang_vel = sim.addGraphStream(self.gyro_graph, 'z_ang_vel', 'rad/s', 0, [0, 0, 1])

    # self.x_accel_filt = sim.addGraphStream(self.accel_filt_graph, 'x_accel', 'm/s^2', 0, [1, 0, 0])
    # self.y_accel_filt = sim.addGraphStream(self.accel_filt_graph, 'y_accel', 'm/s^2', 0, [0, 1, 0])
    # self.z_accel_filt = sim.addGraphStream(self.accel_filt_graph, 'z_accel', 'm/s^2', 0, [0, 0, 1])

    # self.x_ang_vel_filt = sim.addGraphStream(self.gyro_filt_graph, 'x_ang_vel', 'rad/s', 0, [1, 0, 0])
    # self.y_ang_vel_filt = sim.addGraphStream(self.gyro_filt_graph, 'y_ang_vel', 'rad/s', 0, [0, 1, 0])
    # self.z_ang_vel_filt = sim.addGraphStream(self.gyro_filt_graph, 'z_ang_vel', 'rad/s', 0, [0, 0, 1])

    self.yaw_stream = sim.addGraphStream(self.kalman_graph, 'yaw', 'rad', 0, [1, 0, 0])
    self.pitch_stream = sim.addGraphStream(self.kalman_graph, 'pitch', 'rad', 0, [0, 1, 0])
    self.roll_stream = sim.addGraphStream(self.kalman_graph, 'roll', 'rad', 0, [0, 0, 1])

    # self.x_accel_corr = sim.addGraphStream(self.accel_corr_graph, 'x_accel', 'm/s^2', 0, [1, 0, 0])
    # self.y_accel_corr = sim.addGraphStream(self.accel_corr_graph, 'y_accel', 'm/s^2', 0, [0, 1, 0])
    # self.z_accel_corr = sim.addGraphStream(self.accel_corr_graph, 'z_accel', 'm/s^2', 0, [0, 0, 1])

    # self.x_vel = sim.addGraphStream(self.vel_graph, 'x_vel', 'm/s', 0, [1, 0, 0])
    # self.y_vel = sim.addGraphStream(self.vel_graph, 'y_vel', 'm/s', 0, [0, 1, 0])

    #Get sensor readings
    accelerometer = sim.getObject('/Accelerometer')
    self.accelScript = sim.getScript(sim.scripttype_childscript, accelerometer)

    gyroscope = sim.getObject('/GyroSensor')
    self.gyroScript = sim.getScript(sim.scripttype_childscript, gyroscope)

    self.G = 9.80665

    self.I = np.identity(4)

    self.H = self.I
    
    self.K = np.zeros((4,4))
    self.A = np.zeros((4,4))
    
    self.Q = self.I * np.power(self.gyro_std * np.pi / 180, 2)
    self.R = self.I * np.power(self.acc_std * self.G, 2)
    self.P = self.I * 0.1
    self.X = np.array([[1],
                       [0],
                       [0],
                       [0]])

    self.Z = np.zeros((4,1))

    self.lastTime=sim.getSimulationTime() - 0.05
    # do some initialization here
    #
    # Instead of using globals, you can do e.g.:
    # self.myVariable = 21000000

    self.DCM = np.zeros((3,3))


    ## FIR FILTER
    self.coef_acc = np.ones(20)/20    #Moving average FIR filters
    self.coef_gyr = np.ones(3)/3

    self.ax_vector = deque([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], maxlen=20)
    self.ay_vector = deque([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], maxlen=20)
    self.az_vector = deque([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], maxlen=20)

    self.wx_vector = deque([0,0,0], maxlen=3)
    self.wy_vector = deque([0,0,0], maxlen=3)
    self.wz_vector = deque([0,0,0], maxlen=3)

    self.yaw = 0
    self.pitch = 0
    self.roll = 0

    self.acceleration_corrected = np.zeros(3)

    self.vx = 0
    self.vy = 0


def sysCall_actuation():
    # put your actuation code here
    pass

def EP_2_Euler321(Q):
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    e = np.zeros(3)

    e[0] = np.arctan2(2*(q1*q2+q0*q3),q0*q0+q1*q1-q2*q2-q3*q3)
    e[1] = np.arcsin(-2*(q1*q3-q0*q2))
    e[2] = np.arctan2(2*(q2*q3+q0*q1),q0*q0-q1*q1-q2*q2+q3*q3)

    return e

def Euler321_2_EP(e):
    c1 = np.cos(e[0]/2)
    s1 = np.sin(e[0]/2)
    c2 = np.cos(e[1]/2)
    s2 = np.sin(e[1]/2)
    c3 = np.cos(e[2]/2)
    s3 = np.sin(e[2]/2)

    Q = np.zeros((4,1))

    Q[0] = c1*c2*c3+s1*s2*s3
    Q[1] = c1*c2*s3-s1*s2*c3
    Q[2] = c1*s2*c3+s1*c2*s3
    Q[3] = s1*c2*c3-c1*s2*s3

    return Q

def Accel_2_Euler(ax, ay, az):
    g = np.sqrt(np.power(ax,2) + np.power(ay,2) + np.power(az,2))

    theta = np.arcsin(ax/g)  #Rotation in y axis, estimator of pitch

    phi = np.arctan2(-ay,-az)   #Rotation in x axis, estimator of roll

    angles = np.array([0, theta, phi])

    return angles

def sysCall_sensing():
    # Put some sensing code here.
    
    #Accelerometer readings (changing the axis)
    acceleration = sim.callScriptFunction('getAccelData', self.accelScript)

    #Add noise:
    #acceleration = acceleration + np.random.normal(0,self.acc_std * self.G,size=3)

    # sim.setGraphStreamValue(self.accel_graph, self.x_accel, acceleration[0])
    # sim.setGraphStreamValue(self.accel_graph, self.y_accel, acceleration[1])
    # sim.setGraphStreamValue(self.accel_graph, self.z_accel, acceleration[2])

    #Shift array of input vectors and add new value (Changing axis: adapt to axis of sensing device)
    # self.ax_vector.appendleft(-acceleration[0])
    # self.ay_vector.appendleft( acceleration[1])
    # self.az_vector.appendleft( acceleration[2])


    #Apply FIR filter to input vectors:
    # ax = np.dot(self.coef_acc, self.ax_vector)
    # ay = np.dot(self.coef_acc, self.ay_vector)
    # az = np.dot(self.coef_acc, self.az_vector)
    ax = -acceleration[0]
    ay = acceleration[1]
    az = acceleration[2]

    # sim.setGraphStreamValue(self.accel_filt_graph, self.x_accel_filt, ax)
    # sim.setGraphStreamValue(self.accel_filt_graph, self.y_accel_filt, ay)
    # sim.setGraphStreamValue(self.accel_filt_graph, self.z_accel_filt, az)
    
    #Gyroscope readings (changing the axis):
    angular_velocities = sim.callScriptFunction('getGyroData', self.gyroScript)

    #Add noise:
    # angular_velocities = angular_velocities + np.random.normal(0,self.gyro_std * np.pi / 180 ,size=3)

    # sim.setGraphStreamValue(self.gyro_graph, self.x_ang_vel, angular_velocities[0])
    # sim.setGraphStreamValue(self.gyro_graph, self.y_ang_vel, angular_velocities[1])
    # sim.setGraphStreamValue(self.gyro_graph, self.z_ang_vel, angular_velocities[2])

    #Shift array of input vectors and add new value (Changing axis: adapt to axis of sensing device)
    # self.wx_vector.appendleft( angular_velocities[0])
    # self.wy_vector.appendleft(-angular_velocities[1])
    # self.wz_vector.appendleft(-angular_velocities[2])

    #Apply FIR filter to input vectors:
    # wx = np.dot(self.coef_gyr, self.wx_vector)
    # wy = np.dot(self.coef_gyr, self.wy_vector)
    # wz = np.dot(self.coef_gyr, self.wz_vector)

    wx = angular_velocities[0]
    wy = -angular_velocities[1]
    wz = -angular_velocities[2]

    # sim.setGraphStreamValue(self.gyro_filt_graph, self.x_ang_vel_filt, wx)
    # sim.setGraphStreamValue(self.gyro_filt_graph, self.y_ang_vel_filt, wy)
    # sim.setGraphStreamValue(self.gyro_filt_graph, self.z_ang_vel_filt, wz)

    currentTime=sim.getSimulationTime()

    dt=currentTime-self.lastTime

    self.lastTime=currentTime

    self.A = np.array([  [ 0,   -wx,    -wy,    -wz],
                         [wx,     0,     wz,    -wy],
                         [wy,   -wz,      0,     wx],
                         [wz,    wy,    -wx,      0]])
    
    self.A = self.I + self.A * dt * 0.5

    self.X = self.A @ self.X

    self.P = (self.A @ self.P) @ self.A.transpose() + self.Q

    angles = EP_2_Euler321(self.X)

    self.yaw = angles[0]

    aux = np.linalg.inv((self.H @ self.P) @ self.H.transpose() + self.R)

    self.K = (self.P @ self.H.transpose()) @ aux

    angles = Accel_2_Euler(ax, ay, az)

    angles[0] = self.yaw

    self.Z = Euler321_2_EP(angles)

    self.X = self.X + self.K @ (self.Z - (self.H @ self.X))

    self.P = self.P - (self.K @ self.H) @ self.P

    angles = EP_2_Euler321(self.X)

    self.yaw = angles[0]
    self.pitch = angles[1]
    self.roll = angles[2]

    sim.setGraphStreamValue(self.kalman_graph, self.yaw_stream, float(self.yaw*180/np.pi))
    sim.setGraphStreamValue(self.kalman_graph, self.pitch_stream, float(self.pitch*180/np.pi))
    sim.setGraphStreamValue(self.kalman_graph, self.roll_stream, float(self.roll*180/np.pi))

    # #### Compute the accelerations in the reference frame of the agent, without interference of gravity
    # #Compute direct cosine matrix:
    # self.DCM = np.array([[np.cos(self.roll),                 0,              -np.sin(self.pitch)],
    #                      [np.sin(self.pitch)*np.sin(self.roll),   np.cos(self.roll),   np.cos(self.pitch)*np.sin(self.roll)],
    #                      [np.sin(self.pitch)*np.cos(self.roll),   -np.sin(self.roll),  np.cos(self.pitch)*np.cos(self.roll)]])
    
    # acceleration = np.array([ax, ay, az])

    # self.acceleration_corrected = acceleration - np.array([0, 0, -self.G]) @ self.DCM.transpose()

    # sim.setGraphStreamValue(self.accel_corr_graph, self.x_accel_corr, self.acceleration_corrected[0])
    # sim.setGraphStreamValue(self.accel_corr_graph, self.y_accel_corr, self.acceleration_corrected[1])
    # sim.setGraphStreamValue(self.accel_corr_graph, self.z_accel_corr, self.acceleration_corrected[2])

    # #### Compute velocities with numerical integration of the corrected acceleration
    # self.vx += self.acceleration_corrected[0] * dt
    # self.vy += self.acceleration_corrected[1] * dt

    # sim.setGraphStreamValue(self.vel_graph, self.x_vel, self.vx)
    # sim.setGraphStreamValue(self.vel_graph, self.y_vel, self.vy)

def sysCall_cleanup():
    # do some clean-up here
    pass

def getAttitude():
    return [self.yaw, self.pitch, self.roll]

# def getAcceleration():
#     return self.acceleration_corrected[0]

# def getVelocity():
#     return [self.vx, self.vy]

# See the user manual or the available code snippets for additional callback functions and details

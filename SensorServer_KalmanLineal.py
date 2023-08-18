"""
Python code to show real time plot from live accelerometer's
data recieved via SensorServer app over websocket 

"""

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import websocket
import json
import threading

import numpy as np

g = 9.80665

#shared data
yaw_vector = []
pitch_vector = []
roll_vector = []
time_vector = []

wx = 0
wy = 0
wz = 0

ax = 0
ay = 0
az = 0

#Separo el tiempo en 2 para poder correr gyro y corrección en simultaneo al igual que accel y estimación
#y también para evaluar qué tarea del kalman se debe realizar primero (la del menor timestamp)
time_gyro = 0  #nanosegundos
time_accel = 0
prevTime_gyro = 0

initial_timestamp = 0

yaw_data_color = "#d32f2f"   # red
pitch_data_color = "#7cb342"   # green
roll_data_color = "#0288d1"   # blue

background_color = "#fafafa" # white (material)


#flags para comunicar threads
first_run = True
gyro_done = False
accel_done = False

class Sensors:
    
    # called each time when sensor data is recieved
    def on_message(self,ws, message):
        global gyro_done, accel_done,wx,wy,wz,ax,ay,az,time_gyro,time_accel, first_run, initial_timestamp
        
        sensor_type = json.loads(message)['type']
        values = json.loads(message)['values']
        
        if sensor_type =='android.sensor.gyroscope':
            if gyro_done == False:
                wx = values[1]
                wy = values[0]
                wz = -values[2]
                time_gyro = json.loads(message)['timestamp']
                if first_run:
                    initial_timestamp = time_gyro
                gyro_done = True
                first_run = False


        elif sensor_type =='android.sensor.accelerometer':
            if accel_done == False and first_run == False:
                ax = values[1]
                ay = values[0]
                az = -values[2]
                time_accel = json.loads(message)['timestamp']
                accel_done = True

                
    def on_error(self,ws, error):
        print("error occurred")
        print(error)

    def on_close(self,ws, close_code, reason):
        print("connection close")
        print("close code : ", close_code)
        print("reason : ", reason  )

    def on_open(self,ws):
        print(f"connected to : {self.address}")

    # Call this method on seperate Thread
    def make_websocket_connection(self):
        ws = websocket.WebSocketApp('ws://192.168.1.13:8081/sensors/connect?types=["android.sensor.accelerometer","android.sensor.gyroscope"]',
                                on_open=self.on_open,
                                on_message=self.on_message,
                                on_error=self.on_error,
                                on_close=self.on_close)

        # blocking call
        ws.run_forever() 
    
    # make connection and start recieving data on sperate thread
    def connect(self):
        thread = threading.Thread(target=self.make_websocket_connection)
        thread.start()           

class Kalman:
    #constructor
    def __init__(self):    
        self.euler_angles = np.zeros((3,1)) #[Yaw, pitch, roll]
        
        self.X = np.array([[1],[0],[0],[0]])    #Rest Orientation
        self.P = 0.01 * np.identity(4)
        self.Q = (0.028 * np.pi/180)**2 * np.identity(4)
        self.R = (0.7e-3 * g)**2 * np.identity(4)
        
        self.H = np.identity(4)
        
    def estimation(self):
        global wx, wy, wz, time_gyro, prevTime_gyro, yaw_vector, pitch_vector, roll_vector, time_vector

        beta = np.array([[ 0,   -wx,    -wy,    -wz],
                         [wx,     0,     wz,    -wy],
                         [wy,   -wz,      0,     wx],
                         [wz,    wy,    -wx,      0]])
        
        if prevTime_gyro == 0:
            deltaT = 0.0049133
        else:
            deltaT = (time_gyro-prevTime_gyro)/1e9
        
        A = np.identity(4) + deltaT * 0.5 * beta
        
        prevTime_gyro = time_gyro
        
        self.X = np.matmul(A,self.X)
        
        self.euler_angles = self.EP2Euler321(self.X)
        # print("Yaw = ", self.euler_angles.item(0))
        # print("pitch = ", self.euler_angles.item(1))
        # print("roll = ", self.euler_angles.item(2))
        
        yaw_vector.append(self.euler_angles.item(0)*180/np.pi)
        pitch_vector.append(self.euler_angles.item(1)*180/np.pi)
        roll_vector.append(self.euler_angles.item(2)*180/np.pi)
        time_vector.append((time_gyro-initial_timestamp)/1e9)
        
        self.P = np.matmul(np.matmul(A,self.P), np.transpose(A)) + self.Q
    
    def correction(self):
        global ax, ay, time_accel, yaw_vector, pitch_vector, roll_vector, time_vector

        pitch_acc, roll_acc = self.EulerAccel(ax, ay)
        
        Z = self.Euler3212EP([ self.euler_angles[0], pitch_acc, roll_acc ])
        
        #Calculo de ganancia de kalman:
        k1 = np.matmul(self.P, np.transpose(self.H))
        
        K = np.matmul(k1, np.linalg.inv(np.matmul(self.H, k1) + self.R))

        self.X = self.X + np.matmul(K, Z - np.matmul(self.H, self.X))
        
        self.euler_angles = self.EP2Euler321(self.X)
        # print("Yaw = ", self.euler_angles.item(0))
        # print("pitch = ", self.euler_angles.item(1))
        # print("roll = ", self.euler_angles.item(2))
        
        yaw_vector.append(self.euler_angles.item(0)*180/np.pi)
        pitch_vector.append(self.euler_angles.item(1)*180/np.pi)
        roll_vector.append(self.euler_angles.item(2)*180/np.pi)
        time_vector.append((time_accel-initial_timestamp)/1e9)
        
        self.P = np.matmul((np.identity(4) - np.matmul(K,self.H)), self.P)
    
    def EulerAccel(self, ax, ay): #Estima roll y pitch a partir de acelerómetro
        global g    # g should be the average of sqrt(ax^2 + ay^2 + az^2)

        pitch = np.arcsin(  ax / g );
        roll   = np.arcsin( -ay / (g*np.cos(pitch)) );
        
        return pitch, roll
    
    def Euler3212EP(self, e):
        c1 = np.cos(e[0]/2)
        s1 = np.sin(e[0]/2)
        c2 = np.cos(e[1]/2)
        s2 = np.sin(e[1]/2)
        c3 = np.cos(e[2]/2)
        s3 = np.sin(e[2]/2)
        
        q = np.zeros((4,1))
        
        q[0] = c1*c2*c3+s1*s2*s3
        q[1] = c1*c2*s3-s1*s2*c3
        q[2] = c1*s2*c3+s1*c2*s3
        q[3] = s1*c2*c3-c1*s2*s3
        
        return q
    
    def EP2Euler321(self, q):

        # EP2Euler321
        
        # E = EP2Euler321(Q) translates the Euler parameter vector
        # Q into the corresponding (3-2-1) Euler angle set.
        	

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        e = np.zeros((3,1))

        e[0] = np.arctan2(2*(q1*q2+q0*q3),q0*q0+q1*q1-q2*q2-q3*q3)
        
        aux = 2*(q0*q2-q1*q3)
        e[1] = -np.pi/2 + 2*np.arctan2(np.sqrt(np.abs(1+aux)), np.sqrt(np.abs(1-aux)))   #np.arcsin(-2*(q1*q3-q0*q2))
        
        e[2] = np.arctan2(2*(q2*q3+q0*q1),q0*q0-q1*q1-q2*q2+q3*q3)
        
        return e
    
    def implementation(self):
        global first_run,gyro_done,accel_done, time_gyro, time_accel
        while True:
            if gyro_done and time_gyro<=time_accel: #Si ya terminó el gyro
                self.estimation()
                gyro_done = False
                
                
            if accel_done and time_accel<=time_gyro:
                self.correction()
                accel_done = False
            
    def Start(self):
        thread_kalman = threading.Thread(target=self.implementation)

        thread_kalman.start() 

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setBackground(background_color)

        self.graphWidget.setTitle("Atittude Plot", color="#8d6e63", size="20pt")
        
        # Add Axis Labels
        styles = {"color": "#f00", "font-size": "15px"}
        self.graphWidget.setLabel("left", "m/s^2", **styles)
        self.graphWidget.setLabel("bottom", "Time (miliseconds)", **styles)
        self.graphWidget.addLegend()

        self.yaw_data_line =  self.graphWidget.plot([],[], name="yaw", pen=pg.mkPen(color=yaw_data_color))
        self.pitch_data_line =  self.graphWidget.plot([],[], name="pitch", pen=pg.mkPen(color=pitch_data_color))
        self.roll_data_line =  self.graphWidget.plot([],[], name="roll", pen=pg.mkPen(color=roll_data_color))
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_plot_data) # call update_plot_data function every 50 milisec
        self.timer.start()

    def update_plot_data(self):
        
        # limit lists data to 1000 items 
        limit = -50 

        # Update the data
        self.yaw_data_line.setData(time_vector[limit:], yaw_vector[limit:])  
        self.pitch_data_line.setData(time_vector[limit:], pitch_vector[limit:])
        self.roll_data_line.setData(time_vector[limit:], roll_vector[limit:])

sensor = Sensors()
sensor.connect() # asynchronous call

kalman = Kalman()
kalman.Start()

app = QtWidgets.QApplication(sys.argv)

# call on Main thread
window = MainWindow()
window.show()
sys.exit(app.exec_())        

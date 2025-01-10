import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

#Joint to analyze
joint_selected = 5
joint_names = ['Body Front Right', 'Femur Front Right', 'Tibia Front Right', 'Body Front Left', 'Femur Front Left', 'Tibia Front Left', 'Body Back Right', 'Femur Back Right', 'Tibia Back Right', 'Body Back Left', 'Femur Back Left', 'Tibia Back Left']

# Load the data
file_path_sim = 'webots_joint_data.csv'
file_path_real = 'curvas_reales_cuadrada.txt'

data_sim = pd.read_csv(file_path_sim)
data_real = pd.read_csv(file_path_real)

# Sort data by timestamp
data_sim = data_sim.sort_values(by='timestamp')
data_real = data_real.sort_values(by='timestamp')

# Prepare the app and the plot window
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Plot Data with pyqtgraph")
win.resize(1000, 600)
win.setWindowTitle('Sim vs Real')

pg.setConfigOptions(antialias=True)

# Create a plots
plot_sim = win.addPlot(title=f"{joint_names[joint_selected]} - Simulation", row=0, col=0)
plot_sim.setLabel('left', 'Angle °')
plot_sim.setLabel('bottom', 'time ms')
plot_sim.addLegend()  # Add a legend to the plot
plot_sim.showGrid(x=True, y=True)

plot_real = win.addPlot(title=f"{joint_names[joint_selected]} - Reality", row=1, col=0)
plot_real.setLabel('left', 'Angle °')
plot_real.setLabel('bottom', 'time ms')
plot_real.addLegend()  # Add a legend to the plot
plot_real.showGrid(x=True, y=True)

# Extract curves to plot

##SIMULATION##
timestamp_sim = data_sim['timestamp'] - data_sim['timestamp'][0]
sim_targets = np.zeros((12,len(timestamp_sim)))
sim_angles = np.zeros((12,len(timestamp_sim)))

# Body  Front   Right
sim_targets[0,:] = data_sim['BFR_Target']
sim_angles[0,:] = data_sim['BFR_Filt']
# Femur   Front   Right
sim_targets[1,:] = data_sim['FFR_Target']
sim_angles[1,:] = data_sim['FFR_Filt']
# Tibia   Front   Right
sim_targets[2,:] = data_sim['TFR_Target']
sim_angles[2,:] = data_sim['TFR_Filt']
# Body  Front   Left
sim_targets[3,:] = data_sim['BFL_Target']
sim_angles[3,:] = data_sim['BFL_Filt']
# Femur   Front   Left
sim_targets[4,:] = data_sim['FFL_Filt']
sim_angles[4,:] = data_sim['FFL_Target']
# Tibia   Front   Left
sim_targets[5,:] = data_sim['TFL_Target']
sim_angles[5,:] = data_sim['TFL_Filt']
# Body  Back    Right
sim_targets[6,:] = data_sim['BBR_Target']
sim_angles[6,:] = data_sim['BBR_Filt']
# Femur   Back    Right
sim_targets[7,:] = data_sim['FBR_Target']
sim_angles[7,:] = data_sim['FBR_Filt']
# Tibia   Back    Right
sim_targets[8,:] = data_sim['TBR_Target']
sim_angles[8,:] = data_sim['TBR_Filt']
# Body  Back    Left
sim_targets[9,:] = data_sim['BBL_Target']
sim_angles[9,:] = data_sim['BBL_Filt']
# Femur   Back    Left
sim_targets[10,:] = data_sim['FBL_Target']
sim_angles[10,:] = data_sim['FBL_Filt']
# Tibia   Back    Left
sim_targets[11,:] = data_sim['TBL_Target']
sim_angles[11,:] = data_sim['TBL_Filt']

plot_sim.plot(timestamp_sim, sim_targets[joint_selected,:], pen=(255, 201, 14), name='Target')
plot_sim.plot(timestamp_sim, sim_angles[joint_selected,:], pen=(255, 127, 39), name='Angle')

##REALITY##
timestamp_real = np.arange(0, len(data_real['timestamp'])*10e-3, 10e-3)
real_targets = np.zeros((12,len(timestamp_real)))
real_angles = np.zeros((12,len(timestamp_real)))

# Body  Front   Right
real_targets[0,:] = data_real['BFR_Target'] - 90
real_angles[0,:] = data_real['BFR_none'] - 90
# Femur   Front   Right
real_targets[1,:] = data_real['FFR_Target'] - 89
real_angles[1,:] = data_real['FFR_none'] - 89
# Tibia   Front   Right
real_targets[2,:] = data_real['TFR_Target'] - 90
real_angles[2,:] = data_real['TFR_none'] - 90
# Body  Front   Left
real_targets[3,:] = data_real['BFL_Target'] - 95
real_angles[3,:] = data_real['BFL_none'] - 95
# Femur   Front   Left
real_targets[4,:] = 88 - data_real['FFL_none']
real_angles[4,:] = 88 - data_real['FFL_Target']
# Tibia   Front   Left
real_targets[5,:] = 95 - data_real['TFL_Target']
real_angles[5,:] = 95 - data_real['TFL_none']
# Body  Back    Right
real_targets[6,:] = 89 - data_real['BBR_Target']
real_angles[6,:] = 89 - data_real['BBR_none']
# Femur   Back    Right
real_targets[7,:] = data_real['FBR_Target'] - 93
real_angles[7,:] = data_real['FBR_none'] - 93
# Tibia   Back    Right
real_targets[8,:] = data_real['TBR_Target'] - 85
real_angles[8,:] = data_real['TBR_none'] - 85
# Body  Back    Left
real_targets[9,:] = 83 - data_real['BBL_Target']
real_angles[9,:] = 83 - data_real['BBL_none']
# Femur   Back    Left
real_targets[10,:] = 86 - data_real['FBL_Target']
real_angles[10,:] = 86 - data_real['FBL_none']
# Tibia   Back    Left
real_targets[11,:] = 100 - data_real['TBL_Target']
real_angles[11,:] = 100 - data_real['TBL_none']

plot_real.plot(timestamp_real, real_targets[joint_selected,:], pen=(255, 201, 14), name='Target')
plot_real.plot(timestamp_real, real_angles[joint_selected,:], pen=(255, 127, 39), name='Angle')

# Show the window
if __name__ == '__main__':
    QApplication.instance().exec_()

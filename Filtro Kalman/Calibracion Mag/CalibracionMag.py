import time, datetime, serial, re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from collections import deque
import numpy as np

%matplotlib notebook

PORT = "COM13"

HISTORY_SIZE = 5000


INTERVAL = 10


try:
	serialport.close()
except:
	pass

serialport = serial.Serial(PORT,115200,timeout=0.1)
print("Opened",serialport.name)


def get_imu_data(serialport):
	
	line = str(serialport.readline(),'utf-8')

	if not line:
		return None

	if not "Uni:" in line:
		return None

	vals = line.replace("Uni:","").strip().split(',')
	if len(vals) != 9:
		return None
	try:
		vals = [float(i) for i in vals]

	except ValueError:
		return None

	return vals




mag_x = deque(maxlen=HISTORY_SIZE)
mag_y = deque(maxlen=HISTORY_SIZE)
mag_z = deque(maxlen=HISTORY_SIZE)

fig,ax = plt.subplots(1,1)
ax.set_aspect(1)
sc1 = ax.scatter(mag_x,mag_y)
sc1.set_color('r')
sc2 = ax.scatter(mag_y,mag_z)
sc2.set_color('g')
sc3 = ax.scatter(mag_z,mag_x)
sc3.set_color('b')

def animate(i):

	for _ in range(30):
		ret = get_imu_data(serialport)

		if not ret:
			continue

		x,y,z = ret[6:9]
		mag_x.append(np.array(x))
		mag_y.append(np.array(y))
		mag_z.append(np.array(z))

	min_val = min(min(mag_x),min(mag_y),min(mag_z))
	max_val = max(max(mag_x),max(mag_y),max(mag_z))

anim = FuncAnimation(fig,animate,interval = INTERVAL)
plt.show()


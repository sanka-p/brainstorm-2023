#!/usr/bin/python3

# import libraries
import sys
import glob
import serial
import datetime as dt
from pyfirmata import Arduino, util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count
import time
from scipy.fft import fft, fftfreq
import numpy as np

# set parameters
PLT_HISTORY_SIZE = 100
PLT_REFRESH_PERIOD = 100

# detect os and list all ports
if sys.platform.startswith('win'):
    ports = ['COM%s' % (i + 1) for i in range(256)]
elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
    # this excludes your current terminal "/dev/tty"
    ports = glob.glob('/dev/tty[A-Za-z]*')
elif sys.platform.startswith('darwin'):
    ports = glob.glob('/dev/tty.*')
else:
    print('Unsupported platform')
    exit()

avail_ports = []
for port in ports:
    try:
        s = serial.Serial(port)
        s.close()
        avail_ports.append(port)
    except (OSError, serial.SerialException):
        pass

if len(avail_ports) == 0:
    print("No devices detected.")
    exit()

board = None

for i in range(len(avail_ports)):
    try:
        board = Arduino(avail_ports[i])
        print("Connected")
        break
    except (Exception):
        print("Connection failed")
        pass

# check if a connection was instantiated
if (board == None):
    print("All ports busy")
    exit()
'''
it = util.Iterator(board)
it.start()

# set up the AD8232 ECG module
ecg_pin = board.get_pin('a:0:i')
ecg_threshold = 100  # adjust this threshold to suit your needs

# loop forever, reading the ECG module and detecting muscle movement
while True:
    ecg_value = ecg_pin.read()
    print(ecg_value)
    if ecg_value is not None and ecg_value > ecg_threshold:
        print("Muscle movement detected!")

    # wait for a short period to avoid flooding the serial connection
    time.sleep(0.1)
'''

#board.samplingOn(10)

plt.style.use('fivethirtyeight')

x_vals = [0, 0]
y_vals = [0, 0]

# start communication with board
it = util.Iterator(board)
it.start()


# detetct if leads are off
L0_plus = board.get_pin('d:10:i')
L0_minus = board.get_pin('d:11:i')

# if (board.digital[L0_plus].read() == 1 or
#         board.digital[L0_minus].read() == 1):
#     print("Leads disconnected!")
#     exit()


sensor_1 = board.get_pin('a:1:i')
'''
while True:
    print(sensor_1.read())
    time.sleep(0.2)
'''
index = count()

'''
fig, (ax1, ax2) = plt.subplots(2, 1)

# Time domain plot
line1 = ax1.plot([], [])
ax1.set_xlim(0, PLT_HISTORY_SIZE)
ax1.set_ylim(0, 1)

# Freq domain plot
line2 = ax1.plot([], [])
ax2.set_xlim(0, 100) # TODO: Avoid hard coded values
ax2.set_ylim(0, 1000)
'''

def animate(i):
    # Hardcoded values. Change later
    TENSED_RANGE = 0.05
    RELAXED_RANGE = 0.1

    global x_vals
    global y_vals
    global board

    y_vals.append(sensor_1.read())
    x_vals.append(next(index))

    if y_vals[-1] == None:
        return None

    # variation = y_vals[-1] - y_vals[-2]
    # print(variation, end=" ")
    if y_vals[-1] < 0.5:
        print("Tensed")
    elif y_vals[-1] > 0.5:
        print("Relax")
    else:
        print("Undefined")

    # Limit x and y lists to 20 items
    if (len(x_vals) > PLT_HISTORY_SIZE):
        x_vals = x_vals[-PLT_HISTORY_SIZE:]
        y_vals = y_vals[-PLT_HISTORY_SIZE:]

	
    # Compute Fourier Transform
    y_fft = np.abs(fft(y_vals))
    freqs = fftfreq(len(y_vals)) * 19200

    # clear previous plots
    plt.cla()

    plt.subplot(211)
    plt.plot(x_vals, y_vals)

    plt.subplot(212)
    plt.plot(freqs, y_fft)
    plt.xlim(0, 100)
    

ani = FuncAnimation(plt.gcf(), animate, interval=PLT_REFRESH_PERIOD)

plt.tight_layout()
plt.show()

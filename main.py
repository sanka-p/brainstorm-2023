# import libraries

import sys
import glob
import serial
import time
from pyfirmata import Arduino, util
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# global parameters
g_refresh_period = 0.05
g_callibration_points = 100
g_training_points = 1000

# connect to arduino
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

# start communication with the board
it = util.Iterator(board)
it.start()

# define sensor pin
sensor = board.get_pin('a:0:i')

callibration_arr = []
train_arr = []

# read callibration data
print("Reading callibration values")
while len(callibration_arr) < g_callibration_points:
    callibration_arr.append(sensor.read())
    print(callibration_arr[-1])

    if callibration_arr[-1] == None:
        callibration_arr.pop()
        continue

    time.sleep(g_refresh_period)

print("Done reading callibration values")

print("Now move your head")

while len(train_arr) < g_training_points:
    train_arr.append(sensor.read())

    if train_arr[-1] == None:
        train_arr.pop()
        continue

    time.sleep(0.005)

print("Done")

# Define the range of sensor values corresponding to the correct posture
state_range = [min(callibration_arr), max(callibration_arr)]

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
x_train = np.array(train_arr)
y_train = np.zeros((1000,), dtype=int)
y_train[(x_train >= state_range[0]) & (x_train <= state_range[1])] = 1
model.fit(x_train, y_train, epochs=10)

# Test the model on live sensor data
while True:
    data = sensor.read()

    # normalize the data
    data = (data - state_range[0]) / (state_range[1] - state_range[0])

    # make a prediction using the model
    prediction = model.predict(np.array([data]))

    # print the prediction
    if prediction[0][0] > prediction[0][1]:
        print("Wrong Posture.")
    else:
        print("Correct Posture.")

    time.sleep(1)

import numpy as np
import pandas as pd
import scipy.io as sio
import os

from humanactivity import AESKF as AttitudeFilter, quaternion

dirpath = os.getcwd()

filename = '\\Input\\AccGyrMagBar_2016114_132857.csv'
filepath = dirpath+filename
sensorData = pd.read_csv(filepath, sep=',')

matfile = '\\Input\\xpypzp_DP13_011.mat'

matsensor = dirpath + matfile

test = sio.loadmat(matsensor)



DIMS = sensorData.shape

acc = sensorData[['accX', 'accY', 'accZ']]
gyr = sensorData[['gyrX', 'gyrY', 'gyrZ']]
mag = sensorData[['magX', 'magY', 'magZ']]

aeskfObj = AttitudeFilter.AESKF( np.array([1, 0, 0, 0], dtype=np.float64))

# store quaternions
quats = np.zeros([DIMS[0], 4], dtype=np.float64)
q = np.zeros([DIMS[0], 4], dtype=np.float64)
# sensorData.index returns a RangeIndex
for i in sensorData.index:
    aeskfObj.updateAttitude(acc.values[i], gyr.values[i], mag.values[i])
    qNow = aeskfObj.getAttitude()
    # quats[i] = qNow
    q[i, :] = qNow
    #print(str(i)+ " " + aeskfObj.toString())

angles = quaternion.to_nautical_angles(q)

q1 = np.array([1.5, 0.5, -0.5, 2.5], dtype=np.float64)
v = np.array([1, 2, 3], dtype=np.float64)
q = np.copy(q1)
v_rot = quaternion.rotate(q, v)

qv = np.copy(v)
q2 = np.array([[1.5, 0.5, -0.5, 2.5], [-0.5, 2.5, 0.1, 1.5]], dtype=np.float64)
v2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
q = np.copy(q2)
#matplotlib inline
import matplotlib.pyplot as plt
plt.interactive(True)
plt.plot(q)

plt.figure()
plt.plot(sensorData[['accT']], np.rad2deg(angles))
print('Finished')
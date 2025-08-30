from scipy import io
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import matplotlib.pyplot as mp
from rotplot import rotplot
import time
import matplotlib.animation as animation

for p in range(1, 6):
    imuParams = io.loadmat('../IMUParams.mat')
    imuData = io.loadmat('../Data/Train/IMU/imuRaw' + str(p) + '.mat')
    viconData = io.loadmat('../Data/Train/Vicon/viconRot' + str(p) + '.mat')
    
    rots = R.from_matrix(viconData['rots'].transpose(2, 0, 1)) 
    slerp = Slerp(viconData['ts'].flatten(), rots)
    valid_points = (imuData['ts'].flatten() >= viconData['ts'].flatten()[0]) & (imuData['ts'].flatten() <= viconData['ts'].flatten()[-1]) 
    common_ts = imuData['ts'].flatten()[valid_points]

    vicon_reducedframes = slerp(common_ts)
    imu_reducedframes = imuData['vals'][:, valid_points]

    rot_obj = R.from_matrix(vicon_reducedframes.as_matrix())
    vicon_euler = rot_obj.as_euler('zyx', degrees=False)
   
    current_rot = vicon_reducedframes[0]
    orientations = []

    for i, dt in enumerate(common_ts):
        delta_rot = R.from_rotvec(vicon_reducedframes.as_matrix()[i, :] * dt)
        current_rot = current_rot * delta_rot
        orientations.append(current_rot)
    euler_list = []
    for rot in orientations:
        euler_angles = rot.as_euler('zyx', degrees=True)
        euler_list.append(euler_angles)
    orienttion_from_gyro = np.array(euler_list)


    acc = imu_reducedframes[:3,:]
    gyro = imu_reducedframes[3:6, :]
    scale = imuParams['IMUParams'][0,:]
    bias = imuParams['IMUParams'][1,:]

    #ax_hat = (acc[0] + bias[0]) / scale[0]
    #ay_hat = (acc[1] + bias[1]) / scale[1]
    #az_hat = (acc[2] + bias[2]) / scale[2]

    ax_hat = acc[0] * scale[0] - bias[0]
    ay_hat = acc[1] * scale[1] - bias[1]
    az_hat = acc[2] * scale[2] - bias[2]   
    a_hat = np.array([ax_hat, ay_hat, az_hat])

    gbias = imuData['vals'][3:6, :100].sum(axis=1)/100

    gz_hat = (3300/1023) * (np.pi/180) * 0.3 * (gyro[0] - gbias[0])
    gx_hat = (3300/1023) * (np.pi/180) * 0.3 * (gyro[1] - gbias[1])
    gy_hat = (3300/1023) * (np.pi/180) * 0.3 * (gyro[2] - gbias[2])      
    g_hat = np.array([gx_hat, gy_hat, gz_hat])


    gyro_orient = np.zeros((3, gyro.shape[1]))
    acc_orient = np.zeros((3, acc.shape[1]))

    for i in range(0, gyro.shape[1]-1): 
        gyro_orient[:, i+1] = gyro_orient[:, i] + g_hat[:, i] * (common_ts[i+1] - common_ts[i])

    acc_orient[0, :] = np.arctan2(a_hat[1, :], np.sqrt( (a_hat[0, :]*a_hat[0, :]) + (a_hat[2, :]*a_hat[2, :])))
    acc_orient[1, :] = np.arctan2(a_hat[0, :], np.sqrt( (a_hat[1, :]*a_hat[1, :]) + (a_hat[2, :]*a_hat[2, :])))               
    acc_orient[2, :]  = np.arctan2(np.sqrt( (a_hat[0, :]*a_hat[0, :]) + (a_hat[1, :]*a_hat[1, :])), a_hat[2, :])

    common_orient = np.zeros((3, gyro.shape[1]))

    alpha = 0.01
    beta = 0.8
    gamma = 0.95
    for i in range(1, common_orient.shape[1]):
        high_pass = (alpha * (common_orient[0, i-1] + gyro_orient[:, i] - gyro_orient[:, i-1]))
        low_pass = (beta* gyro_orient[:, i] + (1-beta)*common_orient[:, i-1])  
        common_orient[:, i] = (1-gamma)*high_pass + gamma*low_pass
    

    mp.title('imuRaw' + str(p) + '.mat')
    mp.subplot(2, 1, 1)                  
    mp.plot(common_ts, a_hat[0, :], label='IMU Acc X')
    mp.plot(common_ts, a_hat[1, :], label='IMU Acc Y')
    mp.plot(common_ts, a_hat[2, :], label='IMU Acc Z')
    mp.xlabel("Time [s]")
    mp.ylabel("IMU Acc (m/s^2)")
    mp.grid(True)
    mp.legend()

    mp.subplot(2, 1, 2) 
    mp.plot(common_ts, g_hat[0, :], label='IMU Gyro X')
    mp.plot(common_ts, g_hat[1, :], label='IMU Gyro Y')
    mp.plot(common_ts, g_hat[2, :], label='IMU Gyro Z')
    mp.xlabel("Time [s]")
    mp.ylabel("IMU Gyro (rad/s)")
    mp.grid(True)
    mp.legend()

    mp.figure() 
    mp.title('imuRaw' + str(p) + '.mat')
    mp.subplot(2, 1, 1)
    mp.plot(common_ts, acc_orient[0, :], label='IMU Acc Attitude X')
    mp.plot(common_ts, acc_orient[1, :], label='IMU Acc Attitude Y')
    mp.plot(common_ts, acc_orient[2, :], label='IMU Acc Attitude Z')
    mp.xlabel("Time [s]")
    mp.ylabel("IMU Acc Attitude (rad)")
    mp.grid(True)
    mp.legend()

    mp.subplot(2, 1, 2) 
    mp.plot(common_ts, gyro_orient[0, :], label='IMU Gyro Attitude X')
    mp.plot(common_ts, gyro_orient[1, :], label='IMU Gyro Attitude Y')
    mp.plot(common_ts, gyro_orient[2, :], label='IMU Gyro Attitude Z')
    mp.xlabel("Time [s]")
    mp.ylabel("IMU Gyro Attitude (rad)")
    mp.grid(True)
    mp.legend()

    mp.figure()
    mp.title('imuRaw' + str(p) + '.mat')
    mp.subplot(2, 1, 1) 
    mp.plot(common_ts, common_orient[0, :], label='IMU Common Attitude X')
    mp.plot(common_ts, common_orient[1, :], label='IMU Common Attitude Y')
    mp.plot(common_ts, common_orient[2, :], label='IMU Common Attitude Z')
    mp.xlabel("Time [s]")
    mp.ylabel("IMU Common Attitude (rad)")
    mp.grid(True)
    mp.legend()

    mp.subplot(2, 1, 2)
    mp.plot(common_ts, vicon_euler[:, 2], label='Vicon Attitude X')
    mp.plot(common_ts, vicon_euler[:, 1], label='Vicon Attitude Y')
    mp.plot(common_ts, vicon_euler[:, 0], label='Vicon Attitude Z')
    mp.xlabel("Time [s]")
    mp.ylabel("Vicon Attitude (rad)")
    mp.grid(True)
    mp.legend()


    mp.show()



    fig = mp.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, common_orient.shape[1], 100):  
        ax.cla()  
        euler_angles = common_orient[:, i]
        rot_matrix = R.from_euler('xyz', euler_angles).as_matrix()
        rotplot(rot_matrix, currentAxes=ax)
        mp.title(f'Frame {i+1}/{common_orient.shape[1]}')
        mp.pause(0.01)

    mp.show()

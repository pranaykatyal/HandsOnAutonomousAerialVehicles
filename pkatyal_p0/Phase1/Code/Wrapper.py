import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy import io
import matplotlib.pyplot as plt
import os
import importlib.util
import sys


rotplot_path = '../rotplot.py'
spec = importlib.util.spec_from_file_location('rotplot', rotplot_path)
rotplot = importlib.util.module_from_spec(spec)
sys.modules['rotplot'] = rotplot
spec.loader.exec_module(rotplot)

base_imu_path = "../Data/Train/IMU/"
base_vicon_path = "../Data/Train/Vicon/"
IMUParams = io.loadmat("../IMUParams.mat")

def validate_rotation_matrix(matrix, tol=1e-6):
    identity_check = np.dot(matrix, matrix.T)
    if not np.allclose(identity_check, np.eye(3), atol=tol):
        U, S, Vt = np.linalg.svd(matrix)
        matrix = np.dot(U, Vt)
    
    det = np.linalg.det(matrix)
    if not np.isclose(det, 1.0, atol=tol):
        matrix = matrix / (det ** (1/3))
    
    return matrix

rotplot_base_dir = '../Code/Rotplot_frames'

orientation_plot_dir = '../Code/OrientationPlots'
os.makedirs(orientation_plot_dir, exist_ok=True)

stride = 50  

for dataset_num in range(1, 7):
    print(f"\nProcessing dataset {dataset_num}...")
    
    try:
        imu_file = f"imuRaw{dataset_num}.mat"
        vicon_file = f"viconRot{dataset_num}.mat"
        
        IMUData = io.loadmat(os.path.join(base_imu_path, imu_file))
        ViconData = io.loadmat(os.path.join(base_vicon_path, vicon_file))
        
        print("Data Imported.")

        def Acc_To_PhysAcc(IMUData, IMUParams):
            PhysAcc = np.zeros((3, np.shape(IMUData['vals'])[1]))
            for i in range(3):
                # PhysAcc[i,:] = (IMUData['vals'][i,:] - IMUParams['IMUParams'][1,i])/abs(IMUParams['IMUParams'][0,i]) 
                PhysAcc[i,:] = (IMUData['vals'][i,:] * abs(IMUParams['IMUParams'][0,i])) - IMUParams['IMUParams'][1,i]
                
            return PhysAcc

        PhysAcc = Acc_To_PhysAcc(IMUData, IMUParams)

        def w_To_Physw(IMUData):
            n = np.shape(IMUData['vals'])[1]
            Physw = np.zeros((3, n))
            
            bg_wz = np.mean(IMUData['vals'][3, :200])
            bg_wx = np.mean(IMUData['vals'][4, :200])
            bg_wy = np.mean(IMUData['vals'][5, :200])
            
            Physw[0,:] = (3300/1023)*(np.pi/180)*(0.3)*(IMUData['vals'][3,:] - bg_wz)
            Physw[1,:] = (3300/1023)*(np.pi/180)*(0.3)*(IMUData['vals'][4,:] - bg_wx)
            Physw[2,:] = (3300/1023)*(np.pi/180)*(0.3)*(IMUData['vals'][5,:] - bg_wy)
            return Physw

        Physw = w_To_Physw(IMUData)

        IMU_ts = IMUData['ts'].flatten()
        Vicon_ts = ViconData['ts'].flatten()
        Vicon_Rot = ViconData['rots']
        print("Data Calculated")

        def synchronizeUsingSlerp(Vicon_Rot, Vicon_ts, IMU_ts):
            valid_rotations = []
            for i in range(Vicon_Rot.shape[2]):
                rot_matrix = Vicon_Rot[:, :, i]
                try:
                    fixed_matrix = validate_rotation_matrix(rot_matrix)
                    valid_rotations.append(fixed_matrix)
                except:
                    valid_rotations.append(np.eye(3))
            
            Vicon_Rot_valid = np.stack(valid_rotations, axis=2)
            
            Orientation_Vicon = R.from_matrix(np.moveaxis(Vicon_Rot_valid, -1, 0))
            slerp = Slerp(Vicon_ts, Orientation_Vicon)
            ValidMask = (IMU_ts >= Vicon_ts[0]) & (IMU_ts <= Vicon_ts[-1])
            ValidIMU_ts = IMU_ts[ValidMask]
            ViconNewRots = slerp(ValidIMU_ts)
            dtar = np.diff(ValidIMU_ts)
            return ViconNewRots, ValidIMU_ts, ValidMask, dtar

        ViconNewRots, ValidIMU_ts, ValidMask, dtar = synchronizeUsingSlerp(Vicon_Rot, Vicon_ts, IMU_ts)
        print("Data Synchronized.")

        dataset_folder = os.path.join(rotplot_base_dir, f'dataset_{dataset_num}')
        os.makedirs(dataset_folder, exist_ok=True)

        OrientationFromVicon = []
        for i in range(len(ViconNewRots)):
            rotation_object = ViconNewRots[i]
            euler_angles = rotation_object.as_euler('zxy', degrees=True)
            OrientationFromVicon.append(euler_angles)
        OrientationFromVicon = np.array(OrientationFromVicon)

        def getOrientationIMU_Gyro(Physw, ViconNewRots, ValidMask, dtar):
            omega = Physw[:, ValidMask][[1, 2, 0], :]    
            current_rot = ViconNewRots[0]
            orientations = [current_rot]
            
            for i, dt in enumerate(dtar):
                delta_rot = R.from_rotvec(omega[:, i] * dt)
                current_rot = current_rot * delta_rot
                orientations.append(current_rot)
            
            euler_list = []
            for rot in orientations:
                euler_angles = rot.as_euler('zxy', degrees=True)
                euler_list.append(euler_angles)
            
            OrientationFromGyro = np.array(euler_list)
            return OrientationFromGyro

        OrientationFromGyro = getOrientationIMU_Gyro(Physw, ViconNewRots, ValidMask, dtar)
        print("Orientation from Gyro calculated.")

        def getOrientationIMU_Acc(PhysAcc, ViconNewRots, ValidMask):
            acc = PhysAcc[:, ValidMask]
            n = acc.shape[1]
            OrientationFromAcc = np.zeros((n, 3))
            initial_yaw = ViconNewRots[0].as_euler('zxy', degrees=True)[0]
            
            for i in range(n):
                acc_norm = acc[:, i] # / np.linalg.norm(acc[:, i])
                
                pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2)) * 180/np.pi
                roll = np.arctan2(acc_norm[1], acc_norm[2]) * 180/np.pi
                yaw = initial_yaw
                
                OrientationFromAcc[i] = [yaw, pitch, roll]
            
            return OrientationFromAcc

        OrientationFromAcc = getOrientationIMU_Acc(PhysAcc, ViconNewRots, ValidMask)
        print("Orientation from Acceleration calculated.")

        def getOrientationFromComplementaryFilter(Physw, PhysAcc, ViconNewRots, ValidMask, dtar, alpha=0.98):
            omega = Physw[:, ValidMask][[1, 2, 0], :]  # Reorder to [wx, wy, wz]
            
            acc = PhysAcc[:, ValidMask]
            current_orientation = ViconNewRots[0].as_euler('zxy', degrees=True) 
            orientations = [current_orientation] 

            vicon_static = []
            for i in range(min(200, len(ViconNewRots))):
                vicon_euler = ViconNewRots[i].as_euler('zxy', degrees=True)
                vicon_static.append([vicon_euler[1], vicon_euler[2]])  # [pitch, roll]
            vicon_static = np.array(vicon_static)
            vicon_pitch_static = np.mean(vicon_static[:, 0])
            vicon_roll_static = np.mean(vicon_static[:, 1])
            
            acc_pitch_static = []
            acc_roll_static = []
            for i in range(min(200, acc.shape[1])):
                acc_norm = acc[:, i] / np.linalg.norm(acc[:, i])
                pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2)) * 180/np.pi
                roll = np.arctan2(acc_norm[1], acc_norm[2]) * 180/np.pi
                acc_pitch_static.append(pitch)
                acc_roll_static.append(roll)
            
            acc_pitch_bias = np.mean(acc_pitch_static) - vicon_pitch_static
            acc_roll_bias = np.mean(acc_roll_static) - vicon_roll_static
            
            for i, dt in enumerate(dtar):
                delta_rot = R.from_rotvec(omega[:, i] * dt)
                current_rot = R.from_euler('zxy', current_orientation, degrees=True)
                gyro_rot = current_rot * delta_rot
                gyro_euler = gyro_rot.as_euler('zxy', degrees=True)  

                acc_norm = acc[:, i+1] / np.linalg.norm(acc[:, i+1]) if i+1 < acc.shape[1] else acc[:, i] / np.linalg.norm(acc[:, i])

                acc_pitch_raw = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2)) * 180/np.pi
                acc_roll_raw = np.arctan2(acc_norm[1], acc_norm[2]) * 180/np.pi

                acc_pitch = acc_pitch_raw - acc_pitch_bias  
                acc_roll = acc_roll_raw - acc_roll_bias   
                acc_yaw = current_orientation[0]  
            
                acc_euler = np.array([acc_yaw, acc_pitch, acc_roll])
                complementary_euler = np.zeros(3)
                complementary_euler[0] = gyro_euler[0]  
                complementary_euler[1] = alpha * gyro_euler[1] + (1 - alpha) * acc_euler[1]  
                complementary_euler[2] = alpha * gyro_euler[2] + (1 - alpha) * acc_euler[2]  
                
                current_orientation = complementary_euler
                orientations.append(complementary_euler)
            
            OrientationFromComplementary = np.array(orientations)
            return OrientationFromComplementary

        OrientationFromComplementary = getOrientationFromComplementaryFilter(Physw, PhysAcc, ViconNewRots, ValidMask, dtar, alpha=0.9995)
        print("Orientation from Complementary Filter calculated.")

        comparison_save_path = os.path.join(orientation_plot_dir, f'orientation_comparison_{dataset_num}.png')
        plt.figure(figsize=(15, 12))
        plt.subplot(3, 1, 1)
        plt.plot(ValidIMU_ts, OrientationFromVicon[:, 0], label='Vicon Yaw', color='k', linewidth=2)
        plt.plot(ValidIMU_ts, OrientationFromGyro[:, 0], label='Gyro Yaw', color='b', linestyle='--', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromAcc[:, 0], label='Acc Yaw', color='r', linestyle=':', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromComplementary[:, 0], label='Complementary Yaw', color='g', linewidth=1.5)
        plt.ylabel('Yaw (deg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f'Orientation Comparison: Dataset {dataset_num}')
        plt.subplot(3, 1, 2)
        plt.plot(ValidIMU_ts, OrientationFromVicon[:, 1], label='Vicon Pitch', color='k', linewidth=2)
        plt.plot(ValidIMU_ts, OrientationFromGyro[:, 1], label='Gyro Pitch', color='b', linestyle='--', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromAcc[:, 1], label='Acc Pitch', color='r', linestyle=':', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromComplementary[:, 1], label='Complementary Pitch', color='g', linewidth=1.5)
        plt.ylabel('Pitch (deg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(3, 1, 3)
        plt.plot(ValidIMU_ts, OrientationFromVicon[:, 2], label='Vicon Roll', color='k', linewidth=2)
        plt.plot(ValidIMU_ts, OrientationFromGyro[:, 2], label='Gyro Roll', color='b', linestyle='--', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromAcc[:, 2], label='Acc Roll', color='r', linestyle=':', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromComplementary[:, 2], label='Complementary Roll', color='g', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Roll (deg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comparison_save_path)
        plt.close()
        print(f'Saved orientation comparison for dataset {dataset_num} in {comparison_save_path}')


        n_frames = len(ViconNewRots)
        n_select = 3
        if n_frames < n_select:
            indices_to_save = list(range(n_frames))
        else:
            indices_to_save = np.linspace(0, n_frames-1, n_select, dtype=int)
        import matplotlib.pyplot as plt

        vicon_rotmat0 = ViconNewRots[indices_to_save[0]].as_matrix()
        ax = rotplot.rotplot(vicon_rotmat0)
        for idx in indices_to_save[1:]:
            vicon_rotmat = ViconNewRots[idx].as_matrix()
            rotplot.rotplot(vicon_rotmat, ax)
        ax.set_title(f'Vicon Orientation  {dataset_num}')
        save_path = os.path.join(dataset_folder, f'rotplot_overlapped.png')
        plt.savefig(save_path)
        plt.close(ax.figure)
        print(f'Saved overlapped rotplot for dataset {dataset_num} in {save_path}')
    
        base_dir = f'../Code/VideoFrames/dataset_{dataset_num}'
        os.makedirs(base_dir, exist_ok=True)
        methods = {
            'Gyro': OrientationFromGyro,
            'Acc': OrientationFromAcc,
            'CF': OrientationFromComplementary,
            'Vicon': OrientationFromVicon
        }
        for method, orientation_array in methods.items():
            method_dir = os.path.join(base_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            n_frames = len(orientation_array)
            indices = range(0, n_frames, stride)
            for count, idx in enumerate(indices):
                rotmat = R.from_euler('zxy', orientation_array[idx], degrees=True).as_matrix()
                ax = rotplot.rotplot(rotmat)
                ax.set_title(f'{method} Frame {count}')
                plt.savefig(os.path.join(method_dir, f'frame_{count:04d}.png'))
                plt.close(ax.figure)
        print(f'Frames for dataset {dataset_num} saved.')

        rotplot.make_videos_for_dataset(base_dir)

    except Exception as e:
        print(f"Error processing dataset {dataset_num}: {str(e)}")
        print(f"Skipping dataset {dataset_num} due to error.")
        continue
plt.show()

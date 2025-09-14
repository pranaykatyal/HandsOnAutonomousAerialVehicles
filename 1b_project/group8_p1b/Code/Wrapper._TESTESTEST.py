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
base_imu_test_path = "../Data/Test/IMU/"
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

for dataset_num in range(4, 5):
    print(f"\nProcessing dataset {dataset_num}...")
    
    try:
        imu_file = f"imuRaw{dataset_num}.mat"
        vicon_file = f"viconRot{dataset_num}.mat"
        
        is_test_data = dataset_num > 6
        
        if is_test_data:
            IMUData = io.loadmat(os.path.join(base_imu_test_path, imu_file))
            ViconData = None
            print("Test data imported (no Vicon).")
        else:
            IMUData = io.loadmat(os.path.join(base_imu_path, imu_file))
            ViconData = io.loadmat(os.path.join(base_vicon_path, vicon_file))
            print("Train data imported (with Vicon).")

        def Acc_To_PhysAcc(IMUData, IMUParams):
            PhysAcc = np.zeros((3, np.shape(IMUData['vals'])[1]))
            for i in range(3):
                PhysAcc[i,:] = ((IMUData['vals'][i,:] * IMUParams['IMUParams'][0,i]) + IMUParams['IMUParams'][1,i])
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
        
        if not is_test_data:
            Vicon_ts = ViconData['ts'].flatten()
            Vicon_Rot = ViconData['rots']
            
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
            
            OrientationFromVicon = []
            for i in range(len(ViconNewRots)):
                rotation_object = ViconNewRots[i]
                euler_angles = rotation_object.as_euler('zxy', degrees=True)
                OrientationFromVicon.append(euler_angles)
            OrientationFromVicon = np.array(OrientationFromVicon)
            
        else:
            ValidMask = np.ones(len(IMU_ts), dtype=bool)
            ValidIMU_ts = IMU_ts
            dtar = np.diff(ValidIMU_ts)
            initial_orientation = np.array([0.0, 0.0, 0.0])
            ViconNewRots = None
            OrientationFromVicon = None

        print("Data Calculated and Synchronized.")

        dataset_folder = os.path.join(rotplot_base_dir, f'dataset_{dataset_num}')
        os.makedirs(dataset_folder, exist_ok=True)

        def getOrientationIMU_Gyro(Physw, ValidMask, dtar, initial_orientation=None):
            omega = Physw[:, ValidMask][[0 ,1, 2], :]  
            
            if initial_orientation is None:
                initial_euler = np.array([0.0, 0.0, 0.0]) 
            else:
                initial_euler = initial_orientation
                
            current_euler = initial_euler.copy()
            orientations = [current_euler.copy()]

            for i, dt in enumerate(dtar):
                angular_rates_deg = omega[:, i] * 180/np.pi
                current_euler += angular_rates_deg * dt
                orientations.append(current_euler.copy())
            
            return np.array(orientations)

        def getOrientationIMU_Acc(PhysAcc, ValidMask, initial_yaw=0.0):
            acc = PhysAcc[:, ValidMask]
            n = acc.shape[1]
            OrientationFromAcc = np.zeros((n, 3))
            
            for i in range(n):
                acc_norm = acc[:, i]/ np.linalg.norm(acc[:, i])
                
                pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2)) * 180/np.pi
                roll = np.arctan2(acc_norm[1], acc_norm[2]) * 180/np.pi
                yaw = initial_yaw
                
                OrientationFromAcc[i] = [yaw, roll, pitch]
            
            return OrientationFromAcc

        def getOrientationFromComplementaryFilter(Physw, PhysAcc, ValidMask, dtar, initial_orientation, alpha=0.99):
            omega = Physw[:, ValidMask][[0, 1, 2], :] 
            acc = PhysAcc[:, ValidMask]
            
            current_orientation = initial_orientation.copy()
            orientations = [current_orientation.copy()]
            
            for i, dt in enumerate(dtar):
                angular_rates_deg = omega[:, i] * 180/np.pi
                gyro_prediction = current_orientation + angular_rates_deg * dt
                
                acc_norm = acc[:, i+1] / np.linalg.norm(acc[:, i+1]) if i+1 < acc.shape[1] else acc[:, i] / np.linalg.norm(acc[:, i])
                acc_pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2)) * 180/np.pi
                acc_roll = np.arctan2(acc_norm[1], np.sqrt(acc_norm[0]**2 + acc_norm[2]**2)) * 180/np.pi
                acc_measurement = np.array([current_orientation[0], acc_roll, acc_pitch])

                complementary_euler = alpha * gyro_prediction + (1 - alpha) * acc_measurement
                current_orientation = complementary_euler
                orientations.append(complementary_euler.copy())
            
            return np.array(orientations)

        def quaternion_multiplication(q1,q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            return np.array([w, x, y, z])
        
        def getOrientatiomfromMadgwickFilter(Physw, PhysAcc, ValidMask, dtar, initial_orientation, beta=0.01):
            initial_quat = R.from_euler('zxy', initial_orientation, degrees=True).as_quat(scalar_first=True)
            q_current = initial_quat
            orientations = []
            orientations.append(initial_orientation)
            
            for i, dt in enumerate(dtar):
                omega = Physw[:, ValidMask][[1,2,0], i]
                acc = PhysAcc[:, ValidMask][:, i]
                acc_norm = acc / np.linalg.norm(acc)
                
                q_dot_gyro = 0.5 * quaternion_multiplication(q_current, [0, omega[0], omega[1], omega[2]])
                w, x, y, z = q_current
                f = np.array([
                2*(x*z - w*y) - acc_norm[0],       
                2*(w*x + y*z) - acc_norm[1],       
                2*(0.5 - x*x - y*y) - acc_norm[2]])
                
                J = np.array([
                [-2*y,  2*z, -2*w,  2*x], 
                [ 2*x,  2*w,  2*z,  2*y],  
                [ 0,   -4*x, -4*y,  0  ]])
                
                gradient = J.T @ f
                gradient_norm = np.linalg.norm(gradient)
                gradient_normalized = gradient / gradient_norm if gradient_norm != 0 else np.zeros(4)
                
                q_dot_est = q_dot_gyro - beta * gradient_normalized
                q_current = q_current + q_dot_est * dt
                q_current = q_current / np.linalg.norm(q_current)
                
                euler = R.from_quat(q_current, scalar_first=True).as_euler('zxy', degrees=True)
                orientations.append(euler)
            
            return np.array(orientations)
        
        def quaternion_conjugate(q):
            w,x,y,z = q 
            return np.array([w, -x, -y, -z])
        
        def quaternion_normalize(q):
            norm = np.linalg.norm(q)
            if norm == 0:
                return q
            return q / norm
        
        def RotVectoQuaternion(rot_vec):
            q = R.from_rotvec(rot_vec).as_quat(scalar_first=True)
            return q
        
        # def RotVectoQuaternion(rot_vec):
        #     # Paper implementation (equations 14-16) - for testing/verification
        #     angle = np.linalg.norm(rot_vec)
        #     if angle < 1e-8:
        #         return np.array([1, 0, 0, 0])
        #     axis = rot_vec / angle
        #     return np.array([np.cos(angle/2), 
        #                      axis[0]*np.sin(angle/2),
        #                      axis[1]*np.sin(angle/2), 
        #                      axis[2]*np.sin(angle/2)])
        
        def QuaterniontoRotVec(q):
            rot = R.from_quat(q, scalar_first=True)
            return rot.as_rotvec()
        
        # def QuaterniontoRotVec(q):
        #     # Paper implementation (reverse of equations 14-16) - for testing
        #     w, x, y, z = q
        #     angle = 2 * np.arccos(np.abs(w))
        #     if angle < 1e-8:
        #         return np.array([0, 0, 0])
        #     sin_half_angle = np.sin(angle/2)
        #     axis = np.array([x, y, z]) / sin_half_angle
        #     return axis * angle
        
        def generateSigmaPoints(x_mean, P, Q):
            """
            Generate sigma points for UKF
            
            Args:
                x_mean: [7] current state [q0,q1,q2,q3,wx,wy,wz]
                P: [6x6] current covariance matrix
                Q: [6x6] process noise covariance
                
            Returns:
                sigma_points: [12x7] array of sigma points
            """
            
            # Step 1: Combine covariances (equation 35)
            P_augmented = P + Q
            
            # Step 2: More robust error handling
            try:
                S = np.linalg.cholesky(P_augmented)
            except np.linalg.LinAlgError:
                # print("First Cholesky failed, trying regularization...")
                regularization = 1e-3  # Increase regularization
                P_augmented = P_augmented + regularization * np.eye(6)
                try:
                    S = np.linalg.cholesky(P_augmented)
                    # print(f"Success with regularization {regularization}")
                except np.linalg.LinAlgError:
                    # print("Cholesky still failing, using SVD decomposition instead")
                    # Fallback to SVD
                    U, s, Vt = np.linalg.svd(P_augmented)
                    s = np.maximum(s, 1e-6)  # Ensure positive eigenvalues
                    S = U @ np.diag(np.sqrt(s))
            S = S * np.sqrt(12)  # Scale by √(2n) where n=6
            
            # Step 3: Create 6D noise vectors (12 total: +/- each column)
            sigma_points_6d = []
            for i in range(6):
                sigma_points_6d.append(S[:, i])   # +column_i
                sigma_points_6d.append(-S[:, i])  # -column_i
            
            # Step 4: Convert each 6D vector to 7D state vector
            sigma_points_7d = []
            
            # Extract current quaternion and angular velocity from x_mean
            q_mean = x_mean[0:4]
            w_mean = x_mean[4:7]
            
            for noise_6d in sigma_points_6d:
                # Split 6D noise into rotation (3D) and angular velocity (3D)
                rotation_noise = noise_6d[0:3]  # First 3 components
                w_noise = noise_6d[3:6]         # Last 3 components
                
                # Convert rotation noise to quaternion (equation 34)
                q_noise = RotVectoQuaternion(rotation_noise)
                
                # Apply noise to quaternion: q_new = q_mean * q_noise
                q_new = quaternion_multiplication(q_mean, q_noise)
                q_new = quaternion_normalize(q_new) 
                
                # Apply noise to angular velocity: w_new = w_mean + w_noise  
                w_new = w_mean + w_noise
                
                # Combine into 7D state vector
                x_new = np.concatenate([q_new, w_new])
                sigma_points_7d.append(x_new)
            
            return np.array(sigma_points_7d)  # Shape: [12, 7]
        
        
        def ukfPredict(sigma_points, dt):
            """
            Transform sigma points through process model
            
            Args:
                sigma_points: [12x7] array from generateSigmaPoints
                dt: time step
                
            Returns:
                predicted_sigma_points: [12x7] array after process model
            """
            
            predicted_points = []
            
            for i in range(12):
                # Extract current state
                x_current = sigma_points[i]
                q_current = x_current[0:4]  # quaternion
                w_current = x_current[4:7]  # angular velocity
                
                # Process model (equations 20-21):
                
                # 1. Angular velocity (trivial - stays same)
                w_new = w_current  # equation (8): ωk+1 = ωk
                
                # 2. Orientation (quaternion integration)
                # From equations (9-12):
                alpha = np.linalg.norm(w_current) * dt      # total rotation angle
                
                if alpha < 1e-8:  # Handle near-zero rotation
                    q_delta = np.array([1, 0, 0, 0])
                else:
                    axis = w_current / np.linalg.norm(w_current)  # rotation axis
                    q_delta = np.array([np.cos(alpha/2), 
                                    axis[0]*np.sin(alpha/2),
                                    axis[1]*np.sin(alpha/2), 
                                    axis[2]*np.sin(alpha/2)])
                
                # Apply rotation: q_new = q_current * q_delta
                q_new = quaternion_multiplication(q_current, q_delta)
                # q_new = quaternion_multiplication(q_delta, q_current)
                q_new = quaternion_normalize(q_new)
                
                # Combine new state
                x_new = np.concatenate([q_new, w_new])
                predicted_points.append(x_new)
            
            return np.array(predicted_points)
        
        
        def computeQuaternionMean(quaternions, max_iterations=10, tolerance=1e-6):
            """
            Compute mean of quaternions using iterative algorithm (Section 3.4)
            
            Args:
                quaternions: [12x4] array of quaternions from sigma points
                max_iterations: maximum number of iterations
                tolerance: convergence threshold
                
            Returns:
                q_mean: [4] mean quaternion
                error_vectors: [12x3] final error vectors (needed for covariance)
            """
            
            # Step 1: Initial guess (use first quaternion)
            q_mean = quaternions[0].copy()
            
            for _ in range(max_iterations):
                error_vectors = []
                
                # Step 2: Compute error vectors for each quaternion (equation 52)
                for qi in quaternions:
                    # Relative rotation: ei = qi * q_mean^(-1)
                    q_mean_conj = quaternion_conjugate(q_mean)
                    q_relative = quaternion_multiplication(qi, q_mean_conj)
                    
                    if q_relative[0] < 0:  # Ensure positive scalar part
                        q_relative = -q_relative
                    
                    # Convert to rotation vector
                    error_vector = QuaterniontoRotVec(q_relative)
                    error_vectors.append(error_vector)
                
                # Step 3: Average the error vectors (equation 54)
                mean_error = np.mean(error_vectors, axis=0)
                
                # Step 4: Check convergence
                if np.linalg.norm(mean_error) < tolerance:
                    break
                    
                # Step 5: Update mean estimate (equation 55)
                adjustment_quat = RotVectoQuaternion(mean_error)

                q_mean = quaternion_multiplication(adjustment_quat, q_mean)
                q_mean = quaternion_normalize(q_mean)
                if q_mean[0] < 0:
                    q_mean = -q_mean
            
            return q_mean, np.array(error_vectors)
        
        def computeCovariance6D(sigma_points_7d, mean_state_7d, error_vectors):
            """
            Compute 6D covariance from 7D sigma points (Section 3.5.1)
            
            Args:
                sigma_points_7d: [12x7] predicted sigma points
                mean_state_7d: [7] mean state [q_mean, w_mean]
                error_vectors: [12x3] from computeQuaternionMean
                
            Returns:
                P_6d: [6x6] covariance matrix in 6D space
                W_vectors: [12x6] deviation vectors (needed for cross-covariance)
            """
            
            w_mean = mean_state_7d[4:7]
            
            # Convert each 7D sigma point to 6D representation
            W_vectors = []
            
            for i in range(12):
                wi = sigma_points_7d[i, 4:7]
                w_diff = wi - w_mean
                rotation_vector = error_vectors[i]
                Wi = np.concatenate([rotation_vector, w_diff])
                W_vectors.append(Wi)
            
            # Compute 6D covariance matrix (equation 64)
            W_matrix = np.array(W_vectors)  # [12x6]
            P_6d = (1/12) * W_matrix.T @ W_matrix
            
            return P_6d, W_matrix  
        
        
        def transformSigmaPointsGyro(predicted_sigma_points):
            """
            Transform sigma points through gyroscope measurement model
            
            Args:
                predicted_sigma_points: [12x7] from ukfPredict
                
            Returns:
                measurement_sigma_points: [12x3] predicted gyro measurements
            """
            
            measurement_points = []
            
            for i in range(12):
                # Extract state
                x = predicted_sigma_points[i]
                omega = x[4:7]  # Angular velocity part
                
                # Gyro measurement model: z = omega (direct measurement)
                z_gyro = omega.copy()
                measurement_points.append(z_gyro)
            
            return np.array(measurement_points)

        def transformSigmaPointsAccel(predicted_sigma_points):
            """
            Transform sigma points through accelerometer measurement model
            
            Args:
                predicted_sigma_points: [12x7] from ukfPredict
                
            Returns:
                measurement_sigma_points: [12x3] predicted accel measurements
            """
            
            measurement_points = []
            
            for i in range(12):
                # Extract quaternion
                x = predicted_sigma_points[i]
                q = x[0:4]
                q = quaternion_normalize(q)
                
                # Accelerometer measurement model (equations 27, 29)
                # Transform gravity from global to body frame
                # g_global = np.array([0, 0, 0, 1])  # Gravity as quaternion
                g_global = np.array([0, 0, 0, 9.81])
                
                # g_body = q * g_global * q^(-1)
                q_conj = quaternion_conjugate(q)
                temp = quaternion_multiplication(q, g_global)
                g_body_quat = quaternion_multiplication(temp, q_conj)
                
                # Extract vector part [x, y, z]
                z_accel = g_body_quat[1:4]
                measurement_points.append(z_accel)
            
            return np.array(measurement_points)

        
        def computeMeasurementStatistics(measurement_sigma_points):
            """
            Compute mean and covariance of predicted measurements
            
            Args:
                measurement_sigma_points: [12x3] from transformSigmaPoints
                
            Returns:
                z_mean: [3] mean predicted measurement
                P_zz: [3x3] measurement covariance
            """
            # Simple mean (no quaternion complications here)
            z_mean = np.mean(measurement_sigma_points, axis=0)
            
            # Compute covariance
            deviations = measurement_sigma_points - z_mean
            P_zz = (1/12) * deviations.T @ deviations
            
            return z_mean, P_zz
        
        def computeCrossCovariance(W_vectors_6d, measurement_deviations):
            """
            Compute cross-covariance between state and measurements (Section 3.5.3)
            
            Args:
                W_vectors_6d: [12x6] state deviations from computeCovariance6D
                measurement_deviations: [12x3] measurement deviations from z_mean
                
            Returns:
                P_xz: [6x3] cross-covariance matrix
            """
            P_xz = (1/12) * W_vectors_6d.T @ measurement_deviations
            return P_xz
        
        
        def computeKalmanGain(P_xz, P_zz, R):
            """
            Compute Kalman gain (Equation 72)
            
            Args:
                P_xz: [6x3] cross-covariance from computeCrossCovariance
                P_zz: [3x3] measurement covariance from computeMeasurementStatistics  
                R: [3x3] measurement noise covariance
                
            Returns:
                K: [6x3] Kalman gain matrix
            """
            # Innovation covariance (Equation 45/69)
            P_vv = P_zz + R
            
            # Kalman gain (Equation 72)
            K = P_xz @ np.linalg.inv(P_vv)
            
            return K

        def ukfUpdate(x_mean_7d, P_6d, z_actual, z_predicted, K, P_zz, R):
            """
            Update state estimate using Kalman gain (Equations 74-75)
            
            Args:
                x_mean_7d: [7] current state estimate 
                P_6d: [6x6] current covariance estimate
                z_actual: [3] actual sensor measurement
                z_predicted: [3] predicted measurement 
                K: [6x3] Kalman gain from computeKalmanGain
                P_zz: [3x3] measurement covariance 
                R: [3x3] measurement noise covariance
                
            Returns:
                x_updated_7d: [7] updated state estimate
                P_updated_6d: [6x6] updated covariance estimate
            """
            # Innovation (Equation 44)
            innovation = z_actual - z_predicted
            
            # Update in 6D space
            state_update_6d = K @ innovation
            
            # Apply update to 7D state (similar to sigma point conversion)
            q_mean = x_mean_7d[0:4]
            w_mean = x_mean_7d[4:7]
            
            # Split 6D update
            rotation_update = state_update_6d[0:3]
            w_update = state_update_6d[3:6]
            
            # Apply quaternion update
            q_adjustment = RotVectoQuaternion(rotation_update)
            q_updated = quaternion_multiplication(q_adjustment, q_mean)
            q_updated = quaternion_normalize(q_updated)
            
            # Apply angular velocity update
            w_updated = w_mean + w_update
            
            # Combine updated state
            x_updated_7d = np.concatenate([q_updated, w_updated])
            
            # Update covariance (Equation 75)
            P_updated_6d = P_6d - K @ (P_zz + R) @ K.T
            
            return x_updated_7d, P_updated_6d
        
        def getOrientationFromUKF(Physw, PhysAcc, ValidMask, dtar, initial_orientation):
            """
            Complete UKF implementation for orientation estimation
            """
            # Initialize UKF state and parameters
            omega = Physw[:, ValidMask] # Zxy
            omega = omega[[0, 1, 2], :]  # xyz
            acc = PhysAcc[:, ValidMask] # xyz
            # acc = PhysAcc[[2, 0, 1], :][:, ValidMask] # Zxy
            # print(f"Madgwick uses: {Physw[:, ValidMask][[1,2,0], 0]}")  # First timestep

            # print(f"Initial orientation input: {initial_orientation}")

            initial_quat = R.from_euler('zxy', initial_orientation, degrees=True).as_quat(scalar_first=True)
            # initial_quat = R.from_euler('xyz', initial_orientation, degrees=True).as_quat(scalar_first=True)
            # state shud be [q0,q1,q2,q3,wx,wy,wz]
            # x_current = np.array([initial_quat[0], initial_quat[1], initial_quat[2], initial_quat[3], 
                        #  omega[1,0], omega[2,0], omega[0,0]])
            x_current = np.array([initial_quat[0], initial_quat[1], initial_quat[2], initial_quat[3], 
                     omega[0,0], omega[1,0], omega[2,0]])

            
            P_current = np.eye(6) * 1.0 # Initial uncertainty
            
            # print(f"Initial quaternion: {initial_quat}")
            # print(f"UKF state angular velocity: {x_current[4:7]}")      # Should match

            # Q = np.diag([15.0, 15.0, 15.0, 7.0, 7.0, 7.0])  # Process noise [rotation, angular_vel]
            # #compute sensor's measured covariance
            # gyro_sample = omega[:,0:400]
            # accel_sample = acc[:,0:400]

            # gyro_sample = gyro_sample - gyro_sample.mean(axis=1, keepdims=True)
            # accel_sample = accel_sample - accel_sample.mean(axis=1, keepdims=True)

            # R_gyro = np.diag(np.var(gyro_sample, axis=1, ddof=1))
            # R_accel = np.diag(np.var(accel_sample, axis=1, ddof=1))
            # Tune for Dataset 1
            # Q = np.diag([10, 10, 10, 0.5, 0.5, 0.5])
            # R_gyro = np.diag([0.05, 0.05, 0.05])
            # R_accel = np.diag([5.0, 5.0, 5.0])
            # Tune for Dataset 2
            # Q = np.diag([9, 9, 9, 0.5, 0.5, 0.5])
            # R_gyro = np.diag([0.05, 0.05, 0.05])
            # R_accel = np.diag([6.0, 6.0, 6.0])   
            # Tune for Dataset 3
            # Q = np.diag([1, 1, 1, 0.0005, 0.0005, 0.0005])
            # R_gyro = np.diag([0.05, 0.05, 0.05])
            # R_accel = np.diag([1000, 1000, 1000])
            # Tune for Dataset 4            
            Q = np.diag([10,10,10, 9,9,9])
            R_gyro = np.diag([0.1, 0.1, 0.1])
            R_accel = np.diag([0.001, 0.001, 0.001])
            # Tune for Dataset 5            
            # Q = np.diag([32,32,32, 0.15,0.15,15])
            # R_gyro = np.diag([0.05, 0.05, 0.05])
            # R_accel = np.diag([90, 90, 90])
            # Q = np.diag([15, 10, 10, 0.15, 0.15, 0.15])
            # R_gyro = np.diag([1, 1, 1])
            # R_accel = np.diag([350, 350, 350])
            # Tune for Dataset 6
            # Q = np.diag([32,32,32, 0.15,0.15,15])
            # R_gyro = np.diag([0.005, 0.005, 0.005])
            # R_accel = np.diag([900, 900, 900])
            
                  
            orientations = [initial_orientation.copy()]
            # Main UKF loop calling all your functions in sequence
            
            for i, dt in enumerate(dtar):
                # 1. Generate sigma points
                sigma_points = generateSigmaPoints(x_current, P_current, Q)
                # 2. Predict step
                predicted_sigma_points = ukfPredict(sigma_points, dt)
                # 3. Compute predicted state mean and covariance
                q_predicted, error_vectors = computeQuaternionMean(predicted_sigma_points[:, 0:4])
                w_predicted = np.mean(predicted_sigma_points[:, 4:7], axis=0)
                
                x_predicted = np.concatenate([q_predicted, w_predicted])
                
                P_predicted,W_vectors_6d = computeCovariance6D(predicted_sigma_points, x_predicted, error_vectors)
                
                # 4. Update with gyro measurement
                z_gyro_actual = omega[[1,2,0], i] 
                z_gyro_sigma = transformSigmaPointsGyro(predicted_sigma_points)
                z_gyro_mean, P_gyro_zz = computeMeasurementStatistics(z_gyro_sigma)
                
                # if i == 4:
                #     # print(f"UKF quaternion at step 4: {x_current[0:4]}")
                #     expected_quat = R.from_euler('zxy', OrientationFromVicon[4], degrees=True).as_quat(scalar_first=True)
                    # print(f"Expected quaternion: {expected_quat}")

                # if i < 5:  # Only print first few steps
                #     print(f"Step {i}: z_gyro_actual = {z_gyro_actual}")
                #     print(f"Step {i}: z_gyro_predicted = {z_gyro_mean}")
                #     print(f"Step {i}: Innovation gyro = {z_gyro_actual - z_gyro_mean}")
                
                # Compute gyro measurement deviations and cross-covariance
                gyro_deviations = z_gyro_sigma - z_gyro_mean
                P_gyro_xz = computeCrossCovariance(W_vectors_6d, gyro_deviations)
                
                # Apply gyro update
                K_gyro = computeKalmanGain(P_gyro_xz, P_gyro_zz, R_gyro)
                x_current, P_current = ukfUpdate(x_predicted, P_predicted, z_gyro_actual, z_gyro_mean, K_gyro, P_gyro_zz, R_gyro)
                
                # 5. Update with accel measurement  
                z_accel_actual = acc[:, i]
                z_accel_sigma = transformSigmaPointsAccel(predicted_sigma_points)
                z_accel_mean, P_accel_zz = computeMeasurementStatistics(z_accel_sigma)
                
                # Compute accel measurement deviations and cross-covariance
                accel_deviations = z_accel_sigma - z_accel_mean
                P_accel_xz = computeCrossCovariance(W_vectors_6d, accel_deviations)
                
                # Apply accel update
                K_accel = computeKalmanGain(P_accel_xz, P_accel_zz, R_accel)
                x_current, P_current = ukfUpdate(x_current, P_current, z_accel_actual, z_accel_mean, K_accel, P_accel_zz, R_accel)
                
                # 6. Store result and convert to Euler angles
                quat = x_current[0:4] / np.linalg.norm(x_current[0:4])  # Normalize
                euler = R.from_quat(quat, scalar_first=True).as_euler('zxy', degrees=True)
                euler[1] = -euler[1]  # Comment out for Dataset3, Dataset5, Dataset6
                euler[2] = -euler[2]  # Comment out for Dataset3, Dataset5, Dataset6
                orientations.append(euler)

            return np.array(orientations)
        

            
        
        if not is_test_data:
            vicon_raw = ViconNewRots[0].as_euler('zxy', degrees=True)
            # Apply coordinate frame correction at input instead of output
            initial_orientation = vicon_raw.copy()
            initial_orientation[1] = -vicon_raw[1]  # Pitch correction
            initial_orientation[2] = -vicon_raw[2]  # Roll correction
            initial_yaw = initial_orientation[0]
        else:
            initial_orientation = np.array([0.0, 0.0, 0.0])
            initial_yaw = 0.0

        OrientationFromGyro = getOrientationIMU_Gyro(Physw, ValidMask, dtar, initial_orientation)
        print("Orientation from Gyro calculated.")

        OrientationFromAcc = getOrientationIMU_Acc(PhysAcc, ValidMask, initial_yaw)
        print("Orientation from Acceleration calculated.")

        OrientationFromComplementary = getOrientationFromComplementaryFilter(Physw, PhysAcc, ValidMask, dtar, initial_orientation, alpha=0.99)
        print("Orientation from Complementary Filter calculated.")
        
        OrientationFromMadgwick = getOrientatiomfromMadgwickFilter(Physw, PhysAcc, ValidMask, dtar, initial_orientation, beta=0.01)
        print("Orientation from Madgwick Filter calculated.")
        
        OrientationFromUKF = getOrientationFromUKF(Physw, PhysAcc, ValidMask, dtar, initial_orientation)
        print("Orientation from UKF calculated.")
        
        stride = 150  # or 50, adjust as needed
        # print("Frame | Vicon Yaw | UKF Yaw | Vicon Pitch | UKF Pitch | Vicon Roll | UKF Roll")
        # for i in range(0, min(len(OrientationFromVicon), len(OrientationFromUKF)), stride):
        #     v_yaw = OrientationFromVicon[i, 0]
        #     ukf_yaw = OrientationFromUKF[i, 0]
        #     v_pitch = OrientationFromVicon[i, 1]
        #     ukf_pitch = OrientationFromUKF[i, 1]
        #     v_roll = OrientationFromVicon[i, 2]
        #     ukf_roll = OrientationFromUKF[i, 2]
        #     print(f"{i:5d} | {v_yaw:9.4f} | {ukf_yaw:7.4f} | {v_pitch:11.4f} | {ukf_pitch:9.4f} | {v_roll:10.4f} | {ukf_roll:8.4f}")
        # print(f"Gyro orientations: {len(OrientationFromGyro)}")
        # print(f"Vicon orientations: {len(OrientationFromVicon)}")
        # print(f"Madgwick orientations: {len(OrientationFromMadgwick)}")
        # print(f"UKF orientations: {len(OrientationFromUKF)}")

        comparison_save_path = os.path.join(orientation_plot_dir, f'orientation_comparison_{dataset_num}.png')
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        if not is_test_data:
            plt.plot(ValidIMU_ts, OrientationFromVicon[:, 0], label='Vicon Yaw', color='k', linewidth=2)
        plt.plot(ValidIMU_ts, OrientationFromGyro[:, 0], label='Gyro Yaw', color='b', linestyle='--', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromAcc[:, 0], label='Acc Yaw', color='r', linestyle=':', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromComplementary[:, 0], label='Complementary Yaw', color='g', linewidth=1.5)
        plt.plot(ValidIMU_ts, OrientationFromMadgwick[:, 0], label='Madgwick Yaw', color='m', linewidth=1.5)
        plt.plot(ValidIMU_ts, OrientationFromUKF[:, 0], label='UKF Yaw', color='c', linewidth=1.5)
        plt.ylabel('Yaw (deg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        data_type = "Test" if is_test_data else "Train"
        plt.title(f'Orientation Comparison: Dataset {dataset_num} ({data_type})')
        
        # Pitch subplot
        plt.subplot(3, 1, 2)
        if not is_test_data:
            plt.plot(ValidIMU_ts, OrientationFromVicon[:, 1], label='Vicon Pitch', color='k', linewidth=2)
        plt.plot(ValidIMU_ts, OrientationFromGyro[:, 1], label='Gyro Pitch', color='b', linestyle='--', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromAcc[:, 1], label='Acc Pitch', color='r', linestyle=':', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromComplementary[:, 1], label='Complementary Pitch', color='g', linewidth=1.5)
        plt.plot(ValidIMU_ts, OrientationFromMadgwick[:, 1], label='Madgwick Pitch', color='m', linewidth=1.5)
        plt.plot(ValidIMU_ts, OrientationFromUKF[:, 1], label='UKF Pitch', color='c', linewidth=1.5)
        plt.ylabel('Pitch (deg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Roll subplot
        plt.subplot(3, 1, 3)
        if not is_test_data:
            plt.plot(ValidIMU_ts, OrientationFromVicon[:, 2], label='Vicon Roll', color='k', linewidth=2)
        plt.plot(ValidIMU_ts, OrientationFromGyro[:, 2], label='Gyro Roll', color='b', linestyle='--', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromAcc[:, 2], label='Acc Roll', color='r', linestyle=':', alpha=0.7)
        plt.plot(ValidIMU_ts, OrientationFromComplementary[:, 2], label='Complementary Roll', color='g', linewidth=1.5)
        plt.plot(ValidIMU_ts, OrientationFromMadgwick[:, 2], label='Madgwick Roll', color='m', linewidth=1.5)
        plt.plot(ValidIMU_ts, OrientationFromUKF[:, 2], label='UKF Roll', color='c', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Roll (deg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(comparison_save_path)
        plt.close()
        print(f'Saved orientation comparison for dataset {dataset_num}')


        if not is_test_data:
            n_frames = len(ViconNewRots)
            n_select = 3
            if n_frames < n_select:
                indices_to_save = list(range(n_frames))
            else:
                indices_to_save = np.linspace(0, n_frames-1, n_select, dtype=int)

            vicon_rotmat0 = ViconNewRots[indices_to_save[0]].as_matrix()
            ax = rotplot.rotplot(vicon_rotmat0)
            for idx in indices_to_save[1:]:
                vicon_rotmat = ViconNewRots[idx].as_matrix()
                rotplot.rotplot(vicon_rotmat, ax)
            ax.set_title(f'Vicon Orientation Dataset {dataset_num}')
            save_path = os.path.join(dataset_folder, f'rotplot_overlapped.png')
            plt.savefig(save_path)
            plt.close(ax.figure)
            print(f'Saved overlapped rotplot for dataset {dataset_num}')

        base_dir = f'../Code/VideoFrames/dataset_{dataset_num}'
        os.makedirs(base_dir, exist_ok=True)
        
        methods = {
            'Gyro': OrientationFromGyro,
            'Acc': OrientationFromAcc,
            'CF': OrientationFromComplementary,
            'Madgwick': OrientationFromMadgwick,
            'UKF' : OrientationFromUKF,
        }
        
        if not is_test_data:
            methods['Vicon'] = OrientationFromVicon
            
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

        rotplot.make_videos_for_dataset(base_dir, list(methods.keys()))

    except Exception as e:
        print(f"Error processing dataset {dataset_num}: {str(e)}")
        print(f"Skipping dataset {dataset_num} due to error.")
        continue

plt.show() 
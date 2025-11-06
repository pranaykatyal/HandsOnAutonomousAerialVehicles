import numpy as np
import math
from pyquaternion import Quaternion
from numpy.linalg import norm
import scipy

class pid:
    def __init__(self, kp, ki, kd, filter_tau, dt, dim = 1, minVal = -1, maxVal = 1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.minVal = minVal
        self.maxVal = maxVal
        self.filter_tau = filter_tau
        self.dt = dt

        self.minVal = minVal
        self.maxVal = maxVal

        if dim == 1:
            self.prev_filter_val = 0.0
            self.prev_err = 0.0
            self.prev_integral = 0.0
        else:
            self.prev_err = np.zeros(dim, dtype="double")
            self.prev_filter_val = np.zeros(dim, dtype="double")
            self.prev_integral = np.zeros(dim, dtype="double")
    
    def step(self, dsOrErr, current_state = None):

        # Error
        if current_state is None:
            err = dsOrErr
        else:
            desired_state = dsOrErr
            err = desired_state - current_state

        # Error Derivative and filtering
        err_der = (err-self.prev_err)/self.dt

        # Forward euler discretization of first order LP filter
        alpha = self.dt/self.filter_tau
        err_der_filtered = err_der*alpha + self.prev_filter_val*(1-alpha)

        # Integral
        err_integral = err*self.dt + self.prev_integral

        # Raw Output
        out = self.kp*err + self.kd*err_der_filtered + self.ki*err_integral

        # NaN check
        if math.isnan(out):
            print('err', err)
            print(err_integral)
            print(err_der)
            print('Make sure waypoints are not nan. If you still get this error, contact your TA.')
            if current_state is None:
                print('Error is directly provided to the PID')
            else:
                print('desired - ', desired_state)
                print('current - ', current_state)
            raise Exception('PID blew up :( out is nan')

        # Update the internal states
        self.prev_err = err
        self.prev_filter_val = err_der_filtered
        self.prev_integral = err_integral

        # Integral anti-windup. Clamp values
        self.prev_integral = np.clip(self.prev_integral, self.minVal, self.maxVal)

        # Clip the final output
        out = np.clip(out, self.minVal, self.maxVal)

        # Inf check
        if math.isinf(out):
            raise Exception('PID output is inf')
        
        return out

class quad_control:
    def __init__(self):

        # CONTROLLER PROPERTIES AND GAINS
        dt = 0.010
        filter_tau = 0.04
        self.dt = dt

        # tello params
        self.param_mass = 0.08
        self.linearThrustToU = self.param_mass*9.81*2/4

        maxAcc = 5.
        maxVel = 10.
        # maxAng = 30.*3.14159/180.
        self.maxRate = 1.5
        maxAct = 0.3

        minAcc = -maxAcc
        minVel = -maxVel
        # minAng = -maxAng
        self.minRate = -self.maxRate
        minAct = -maxAct
        
        
        ##################### SET YOUR GAINS FROM P2 #################################################
        # NED position controller. EDIT GAINS HERE
        self.x_pid = pid(1.0, 1.0, 1.0, filter_tau, dt, minVal = minVel, maxVal=maxVel)
        self.y_pid = pid(1.0, 1.0, 1.0, filter_tau, dt, minVal = minVel, maxVal=maxVel)
        self.z_pid = pid(1.0, 1.0, 1.0, filter_tau, dt, minVal = minVel, maxVal=maxVel)

        # NED velocity controller. EDIT GAINS HERE
        self.vx_pid = pid(1.0, 1.0, 1.0, filter_tau, dt, minVal = minAcc, maxVal=maxAcc)
        self.vy_pid = pid(1.0, 1.0, 1.0, filter_tau, dt, minVal = minAcc, maxVal=maxAcc)
        self.vz_pid = pid(1.0, 1.0, 1.0, filter_tau, dt, minVal = minAcc, maxVal = maxAcc)

        # Quaternion based P Controller. Output is desired angular rate. tau is time constant of closed loop
        self.tau_angle = 0
        self.angle_sf = np.array((1, 1, 0.4)) # deprioritize yaw control using this scale factor

        # Angular velocity controller
        kp_angvel = 1.0
        self.p_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal = minAct, maxVal = maxAct)
        self.q_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal = minAct, maxVal = maxAct)
        self.r_pid = pid(kp_angvel, 0, kp_angvel/15, filter_tau, dt, minVal = minAct, maxVal = maxAct)

        # For logging
        self.current_time = 0.
        self.timeArray = 0
        self.controlArray = np.array([0., 0, 0, 0])

    def step(self, X, WP, VEL_SP, ACC_SP):
        """
        Quadrotor position controller
        """
        # EXTRACT STATES
        xyz = X[0:3]
        vxyz = X[3:6]
        quat_list = X[6:10]
        pqr = X[10:13]

        quat = Quaternion(quat_list)
        ypr = quat.yaw_pitch_roll
        yaw = ypr[0]
        pitch = ypr[1]
        roll = ypr[2]

        DCM_EB = quat.rotation_matrix
        DCM_BE = DCM_EB.T

        # NED POSITION CONTROLLER
        vx_ned_sp = VEL_SP[0] + self.x_pid.step(WP[0], xyz[0])
        vy_ned_sp = VEL_SP[1] + self.y_pid.step(WP[1], xyz[1])
        vz_ned_sp = VEL_SP[2] + self.z_pid.step(WP[2], xyz[2])

        vxyz_sp = np.array([vx_ned_sp, vy_ned_sp, vz_ned_sp])

        # # NED VELOCITY CONTROLLER x, y, z
        acc_x_sp = ACC_SP[0] + self.vx_pid.step(vxyz_sp[0], vxyz[0])
        acc_y_sp = ACC_SP[1] + self.vy_pid.step(vxyz_sp[1], vxyz[1])
        acc_z_sp = ACC_SP[2] + self.vz_pid.step(vxyz_sp[2], vxyz[2]) 

        # ACCELERATION SETPOINT TO QUATERNION SETPOINT

        # mass specific force to be applied by the actuation system
        f_inertial = np.array((acc_x_sp, acc_y_sp, acc_z_sp)) - np.array((0., 0., 9.81))

        rotationAxis = np.cross(np.array((0., 0., -1.)), f_inertial/norm(f_inertial))
        rotationAxis += np.array((1e-3, 1e-3, 1e-3)) # Avoid numerical issue

        sinAngle = norm(rotationAxis)
        rotationAxis = rotationAxis / norm(rotationAxis)

        cosAngle = np.dot( np.array((0., 0., -1.)), f_inertial/norm(f_inertial) )

        angle = math.atan2(sinAngle, cosAngle)

        quat_wo_yaw = Quaternion(axis=rotationAxis, radians=angle)

        quat_yaw = Quaternion(axis=np.array((0., 0., 1)), radians=WP[3])

        # I dont think the order of multiplication matters in this special case. check later TODO
        # Also multiplication can even be done in the first place as they share the same NED basis
        quat_sp = quat_wo_yaw * quat_yaw

        # QUATERNION P CONTROLLER (simplified version of px4 implementation) 8)
        # https://github.com/PX4/PX4-Autopilot/blob/1c1f8da7d9cc416aaa53d76254fe08c2e9fa65e6/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp#L91

        # quat_sp = Quaternion(-0.183, 0.113, 0.108, 0.971)
        err_quat = quat.inverse*quat_sp
        
        pqr_sp = 2./self.tau_angle*np.sign(err_quat.w)*np.array((err_quat.x, err_quat.y, err_quat.z))
        # pqr_sp = np.array((0.1, 0.2, -0.2))

        pqr_sp = np.multiply(pqr_sp, self.angle_sf)
        pqr_sp = pqr_sp.clip(self.minRate, self.maxRate)

        # ANGULAR VELOCITY
        tau_x = self.p_pid.step(pqr_sp[0], pqr[0])
        tau_y = self.q_pid.step(pqr_sp[1], pqr[1])
        tau_z = self.r_pid.step(pqr_sp[2], pqr[2])

        # ROTOR THRUST. Cheating a bit here making use of tello parameters
        netSpecificThrustFromRotors = norm(f_inertial) # N/kg
        netThrust = netSpecificThrustFromRotors * self.param_mass

        thrustPerRotor = netThrust/4.
        throttle = thrustPerRotor/self.linearThrustToU

        # MIXER
        u1 = throttle - tau_x + tau_y + tau_z
        u2 = throttle + tau_x - tau_y + tau_z
        u3 = throttle + tau_x + tau_y - tau_z
        u4 = throttle - tau_x - tau_y - tau_z

        U = np.array([u1, u2, u3, u4])
        U = U.clip(0.0, 1.0)

        # Logger
        self.controlArray = np.vstack((self.controlArray, np.array((throttle, tau_x, tau_y, tau_z))))
        self.timeArray = np.append(self.timeArray, self.current_time)
        self.current_time+=self.dt

        loggedDict = {'control_time': self.timeArray,
                  'control_premix': self.controlArray}
        
        scipy.io.savemat('./log/control.mat', loggedDict)

        return U

class QuadrotorController:
    """
    Enhanced interface for the quad_control class with trajectory tracking
    """
    
    def __init__(self, drone_params):
        self.params = drone_params
        self.controller = quad_control()
        
        # Trajectory tracking
        self.trajectory_points = None
        self.trajectory_velocities = None
        self.trajectory_accelerations = None
        self.time_points = None
        
        # Performance metrics
        self.position_errors = []
        self.velocity_errors = []
        
    def set_trajectory(self, trajectory_points, time_points, velocities, accelerations):
        """Set the reference trajectory"""
        self.trajectory_points = trajectory_points
        self.time_points = time_points
        self.trajectory_velocities = velocities
        self.trajectory_accelerations = accelerations

    
    def get_desired_state(self, t):
        """Get desired position, velocity, and acceleration at time t"""
        if (self.trajectory_points is None or len(self.trajectory_points) == 0 or 
            self.time_points is None):
            return np.zeros(3), np.zeros(3), np.zeros(3)
        
        # Handle time bounds
        if t <= self.time_points[0]:
            idx = 0
        elif t >= self.time_points[-1]:
            idx = len(self.trajectory_points) - 1
        else:
            # Find closest time point with interpolation
            idx = np.searchsorted(self.time_points, t)
            if idx > 0:
                t1, t2 = self.time_points[idx-1], self.time_points[idx]
                alpha = (t - t1) / (t2 - t1) if t2 != t1 else 0
                
                pos_des = ((1 - alpha) * self.trajectory_points[idx-1] + 
                          alpha * self.trajectory_points[idx])
                vel_des = ((1 - alpha) * self.trajectory_velocities[idx-1] + 
                          alpha * self.trajectory_velocities[idx])
                acc_des = ((1 - alpha) * self.trajectory_accelerations[idx-1] + 
                          alpha * self.trajectory_accelerations[idx])
                
                return pos_des, vel_des, acc_des
            else:
                idx = 0
        
        pos_des = self.trajectory_points[idx]
        vel_des = self.trajectory_velocities[idx]
        acc_des = self.trajectory_accelerations[idx]
        
        return pos_des, vel_des, acc_des
    
    def compute_control(self, current_state, t):
        """Main control computation"""
        # Get desired trajectory state
        pos_des, vel_des, acc_des = self.get_desired_state(t)
        
        # Create waypoint [x, y, z, yaw]
        waypoint = np.append(pos_des, 0.0)  # Zero yaw
        
        # Use the controller
        try:
            control_input = self.controller.step(current_state, waypoint, vel_des, acc_des)
        except Exception as e:
            print(f"Controller error: {e}")
            # Emergency hover
            hover_thrust = self.params.mass * 9.81 / 4.0 / self.params.linearThrustToU
            control_input = np.array([hover_thrust] * 4)
        
        # Track performance
        current_pos = current_state[0:3]
        current_vel = current_state[3:6]
        
        pos_error = np.linalg.norm(current_pos - pos_des)
        vel_error = np.linalg.norm(current_vel - vel_des)
        
        self.position_errors.append(pos_error)
        self.velocity_errors.append(vel_error)
        
        return control_input
    
    def reset_metrics(self):
        """Reset performance tracking"""
        self.position_errors = []
        self.velocity_errors = []
    
    def get_performance_summary(self):
        """Get performance metrics"""
        if not self.position_errors:
            return "No performance data"
        
        return f"""
Performance Summary:
  Mean Position Error: {np.mean(self.position_errors):.3f} m
  Max Position Error:  {np.max(self.position_errors):.3f} m
  Mean Velocity Error: {np.mean(self.velocity_errors):.3f} m/s
  Max Velocity Error:  {np.max(self.velocity_errors):.3f} m/s
"""
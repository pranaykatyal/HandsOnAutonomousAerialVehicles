# DJI Tello properties

# SI units unless specified otherwise

# Benotsmane, R.; VÃ¡sÃ¡rhelyi, J. Towards Optimization of Energy Consumption of Tello Quad-Rotor with Mpc Model Implementation. Energies 2022, 15, 9207. https://doi.org/10.3390/en15239207 

import numpy as np
mass = 0.08
Ixx = 0.0097
Iyy = 0.0097
Izz = 0.017
# inertia matrix as given in https://in.mathworks.com/help/aeroblks/6dofeulerangles.html
inertiaMat = np.diag([Ixx, Iyy, Izz])

rotorDragCoeff = 0.08
rotorLiftCoeff = 1
halfDiag = 0.06

gravity = 9.81

# Rotor position
rpos = np.array([
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0]
])*halfDiag/np.sqrt(2.0)
'''
Weight: Approximately 80 g (Propellers and Battery Included)

Dimensions: 98Ã—92.5Ã—41 mm - as found on : https://www.ryzerobotics.com/tello/specs
'''
# Robot dimensions (cuboid) - tall vertical profile
robot_width = 0.1   # meters (X direction)
robot_length = 0.1  # meters (Y direction)
robot_height = 0.2  # meters (Z direction) -  made it taller as coz of assignment 

# Safety buffer beyond physical dimensions
safety_buffer = 0.2  # Additional clearance

# Total margins for obstacle bloating
margin_xy = (robot_width / 2) + safety_buffer  
margin_z = (robot_height / 2) + safety_buffer  


# linear control input to thrust mapping
#  This is totally guess work for now
#  Assuming, the drone runs at half throttle during hover 
linearThrustToU = mass*gravity*2/4
linearTorqToU = linearThrustToU/rotorLiftCoeff*rotorDragCoeff
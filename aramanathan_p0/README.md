This project implements IMU data processing and orientation estimation techniques using Python. The system fuses accelerometer and gyroscope measurements with a complementary filter and aligns the results with ground-truth motion capture data from the Vicon system.

The workflow includes:

- Converting raw IMU sensor outputs into physical units.

- Estimating orientation from gyroscope integration and accelerometer tilt calculations.

- Fusing results using a complementary filter.

- Converting the IMU reference frame into the Vicon reference frame for comparison.

- Plotting and analyzing results for multiple datasets.

A detailed explanation of the methodology and results is provided in `Report.pdf`.

aramanathan_p0/

│── Phase1

│    └── Code   

│              └── Report.pdf                # Project report with methodology and analysis  
│              └── Wrapper.py                # Main program to run the processing pipeline  
│── IMUParams.mat             # Calibration parameters for IMU conversion  
│
└── Data/
    └── Train/
        │── IMU/              # Directory containing IMU datasets  
        │    ├── imuRaw1.mat  
        │    ├── imuRaw2.mat  
        │    └── ...  
        │
        └── Vicon/            # Directory containing Vicon datasets  
             ├── viconRot1.mat  
             ├── viconRot2.mat  
             └── ...

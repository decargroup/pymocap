# %%
from pymocap import MocapTrajectory
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
filename = "data/random2.bag"
agent = "ifo003"


# Extract data
mocap = MocapTrajectory.from_bag(filename, agent)

# Now you can sample at any time you want (in seconds from start
# Uses smoothing spline interpolation internally
query_stamps = np.arange(0, mocap.stamps[-1], 0.1)

pos = mocap.position(query_stamps)
quat = mocap.quaternion(query_stamps)
vel = mocap.velocity(query_stamps)
accel = mocap.acceleration(query_stamps)
imu_accel = mocap.accelerometer(query_stamps)
omega = mocap.angular_velocity(query_stamps)
is_static = mocap.is_static(query_stamps)

# %% Plotting
################################################################################
### POSITION
fig, axs = plt.subplots(3, 1, sharex=True)
axs: List[plt.Axes] = axs
axs[0].plot(query_stamps, pos[:, 0])
axs[1].plot(query_stamps, pos[:, 1])
axs[2].plot(query_stamps, pos[:, 2])
axs[2].plot(query_stamps, is_static.astype(int), label="Static")
axs[0].set_title("Mocap Position Trajectory")
axs[2].legend()

################################################################################
### VELOCITY
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(query_stamps, vel[:, 0])
axs[1].plot(query_stamps, vel[:, 1])
axs[2].plot(query_stamps, vel[:, 2])
axs[0].set_title("Mocap Velocity Trajectory")

################################################################################
### ACCELERATION
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(query_stamps, accel[:, 0])
axs[1].plot(query_stamps, accel[:, 1])
axs[2].plot(query_stamps, accel[:, 2])
axs[0].set_title("Mocap Acceleration Trajectory")


################################################################################
### QUATERNION
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(query_stamps, quat[:, 0])
axs[1].plot(query_stamps, quat[:, 1])
axs[2].plot(query_stamps, quat[:, 2])
axs[3].plot(query_stamps, quat[:, 3])
axs[0].set_title("Mocap Quaternion Trajectory")
axs[0].set_ylim(-1.05, 1.05)

################################################################################
### ANGULAR VELOCITY
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(query_stamps, omega[:, 0])
axs[1].plot(query_stamps, omega[:, 1])
axs[2].plot(query_stamps, omega[:, 2])
axs[0].set_title("Mocap Angular Velocity Trajectory")
axs[0].set_ylim(-1.05, 1.05)


plt.show()

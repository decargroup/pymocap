# %%
from pymocap import MocapTrajectory, HeightData
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import rosbag

sns.set_theme(style="whitegrid")
filename = "data/random2.bag"
agent = "ifo001"

# Extract data
mocap = MocapTrajectory.from_bag(filename, agent)
height = HeightData.from_bag(filename, f"/{agent}/mavros/distance_sensor/hrlv_ez4_pub")

# %% Plotting
height.plot(mocap)
plt.show()

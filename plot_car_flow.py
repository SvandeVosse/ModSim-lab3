#
# Plot the car flow versus initial car density for each simulated height
#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# indicate number of runs per simulation
R = 10

# read data and indicate data type
car_flow_df = pd.read_csv(f"0_Car_flow_R{R}_0.csv")
car_flow_df = car_flow_df.astype(
    {"density": float, "height": int, "rep_num": int, "car_flow": float}
)

# set up figure with subplots for each height
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=[15, 10])

# save axes as list
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# plot car flow versus car density for each height
i = 0
for height in np.unique(car_flow_df["height"]):

    # initialize dataframe for the mean data
    mean_data = pd.DataFrame([], columns=["density", "mean_car_flow", "err_car_flow"])
    height_data = car_flow_df[car_flow_df["height"] == height]

    # calculate the mean car flow for each density and store in mean data
    for density in np.unique(car_flow_df["density"]):
        density_data = height_data[height_data["density"] == density]
        mean_data.loc[len(mean_data)] = [
            density,
            density_data["car_flow"].mean(),
            density_data["car_flow"].std() / np.sqrt(len(density_data)),
        ]

    # plot the mean car flow versus density
    axes[i].errorbar(
        mean_data["density"],
        mean_data["mean_car_flow"],
        yerr=mean_data["err_car_flow"],
        fmt="o",
        ms=4,
        capsize=3,
        label=f"T = {height}",
    )

    # figure layout
    axes[i].set_xlabel("car density $[L^{-1}]$")
    axes[i].set_ylabel("car flow $[t^{-1}]$")
    axes[i].legend()

    i += 1

plt.savefig(f"Car_flow_{R}.png")

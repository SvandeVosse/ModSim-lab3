#
# Plot the car flow versus initial car density
#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

R = 10
car_flow_df = pd.read_csv(f"0_Car_flow_R{R}_0.csv")
car_flow_df = car_flow_df.astype(
    {"density": float, "height": int, "rep_num": int, "car_flow": float}
)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=[15, 10])

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

i = 0

for height in np.unique(car_flow_df["height"]):

    mean_data = pd.DataFrame([], columns=["density", "mean_car_flow", "err_car_flow"])

    height_data = car_flow_df[car_flow_df["height"] == height]

    for density in np.unique(car_flow_df["density"]):

        density_data = height_data[height_data["density"] == density]
        mean_data.loc[len(mean_data)] = [
            density,
            density_data["car_flow"].mean(),
            density_data["car_flow"].std() / np.sqrt(len(density_data)),
        ]

    axes[i].errorbar(
        mean_data["density"],
        mean_data["mean_car_flow"],
        yerr=mean_data["err_car_flow"],
        fmt="o",
        ms=4,
        capsize=3,
        label=f"T = {height}",
    )
    axes[i].set_xlabel("car density $[L^{-1}]$")
    axes[i].set_ylabel("car flow $[t^{-1}]$")
    axes[i].legend()

    i += 1

plt.savefig(f"Car_flow_{R}.png")

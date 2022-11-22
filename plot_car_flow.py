#
# Plot the car flow versus initial car density
#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

N_sim = 3
car_flow_df = pd.read_csv(f"Car_flow_{N_sim}_0.csv")
car_flow_df = car_flow_df.astype({"density": float, "rep_num": int, "car_flow": float})

mean_data = pd.DataFrame([], columns=["density", "mean_car_flow", "err_car_flow"])

for density in np.unique(car_flow_df["density"]):
    density_data = car_flow_df[car_flow_df["density"] == density]
    mean_data.loc[len(mean_data)] = [
        density,
        density_data["car_flow"].mean(),
        density_data["car_flow"].std() / np.sqrt(len(density_data)),
    ]

plt.errorbar(
    mean_data["density"],
    mean_data["mean_car_flow"],
    yerr=mean_data["err_car_flow"],
    fmt="o",
    ms=4,
    capsize=3,
    color="black",
)
plt.xlabel("car density $[L^{-1}]$")
plt.ylabel("car flow $[t^{-1}]$")
plt.show()

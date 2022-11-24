#
# Determine the minimal amount of time steps (system height) for which
# the critical density is estimated correctly 90% of the time
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model

# determine the critical density for each system height by fitting a offset triangle through the

R = 10
number_heights = 6

# define gaussian function
def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-((x - center) ** 2) / width)


mod = Model(gaussian)

# set initial parameters
params = mod.make_params(center=0, amplitude=0.5, width=0.5)

# create dictionary for all heights to put in all critical densities
heights = [5, 20, 100, 500, 1000, 2000]
dic = {}
for item in heights:
    dic["height_list_" + str(item)] = []


for dummy_var in range(10):
    # Read the csv files
    df = pd.read_csv(f"{dummy_var}_Car_flow_R{R}_0.csv")
    df = df.astype({"density": float, "height": int, "rep_num": int, "car_flow": float})

    # calculate critical density from gaussian fit (Based on plot_car_flow.py)
    for height in np.unique(df["height"]):
        mean_data = pd.DataFrame(
            [], columns=["density", "mean_car_flow", "err_car_flow"]
        )

        height_data = df[df["height"] == height]

        for density in np.unique(df["density"]):

            density_data = height_data[height_data["density"] == density]
            mean_data.loc[len(mean_data)] = [
                density,
                density_data["car_flow"].mean(),
                density_data["car_flow"].std() / np.sqrt(len(density_data)),
            ]
        # Perform a gaussian fit on the data to find the center of the distribution
        fit = mod.fit(mean_data["mean_car_flow"], params, x=mean_data["density"])

        # plt.plot(mean_data["density"], mean_data["mean_car_flow"], "o")
        # plt.plot(mean_data["density"], fit.init_fit, "--", label="initial fit")
        # plt.plot(mean_data["density"], fit.best_fit, "-", label="best fit")
        # plt.legend()
        # plt.show()

        print(fit.params["center"].value)

        if 0.45 <= fit.params["center"].value <= 0.55:
            dic["height_list_" + str(height)].append("correct")
        else:
            dic["height_list_" + str(height)].append("incorrect")

# print(dic)

for item in heights:
    # print(dic["height_list_" + str(item)].count("correct"))

    probability = dic["height_list_" + str(item)].count("correct") / len(
        dic["height_list_" + str(item)]
    )
    probability = probability * 100
    print(
        f"The probability of finding the correct critical density for {item} time steps is {probability} %"
    )

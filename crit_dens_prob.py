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


def triangle(x, l_slope, r_slope, center, amplitude):
    y = [
        amplitude - (l_slope * (center - i))
        if i <= center
        else amplitude - (r_slope * (center - i))
        for i in x
    ]
    return y


mod_gauss = Model(gaussian)
mod_tri = Model(triangle)

# set initial parameters
params_gauss = mod_gauss.make_params(center=0.5, amplitude=0.5, width=0.5)
mod_gauss.set_param_hint("width", min=0, max=1)

params_tri = mod_tri.make_params(l_slope=0.5, r_slope=-0.5, center=0.5, amplitude=0.25)


# create dictionary for all heights to put in all critical densities
heights = [5, 20, 100, 500, 1000, 2000]
dic_gauss = {}
dic_tri = {}
for item in heights:
    dic_gauss["height_list_" + str(item)] = []
    dic_tri["height_list_" + str(item)] = []

dics = (dic_gauss, dic_tri)

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
        fit_gauss = mod_gauss.fit(
            mean_data["mean_car_flow"],
            params_gauss,
            x=mean_data["density"],
            weights=1 / mean_data["err_car_flow"],
            nan_policy="omit",
        )

        # Perform a triangle fit on the data to find the center of the distribution
        fit_tri = mod_tri.fit(
            mean_data["mean_car_flow"],
            params_tri,
            x=mean_data["density"],
            weights=1 / mean_data["err_car_flow"],
            nan_policy="omit",
        )

        fits = (fit_gauss, fit_tri)

        # plot once for every height
        if dummy_var == 0:

            fig, axes = plt.subplots(1, 2, figsize=[12, 7])

            i = 0

            for ax in axes:

                ax.errorbar(
                    mean_data["density"],
                    mean_data["mean_car_flow"],
                    yerr=mean_data["err_car_flow"],
                    capsize=4,
                    fmt="o",
                )
                ax.plot(
                    mean_data["density"], fits[i].init_fit, "--", label="initial fit"
                )
                ax.plot(mean_data["density"], fits[i].best_fit, "-", label="best fit")
                ax.set_title(f"Fit for height {height}")
                ax.legend()

                i += 1

            plt.show()

        # print(fit.params["center"].value)

        for i in range(len(dics)):
            if 0.45 <= fits[i].params["center"].value <= 0.55:
                dics[i]["height_list_" + str(height)].append("correct")
            else:
                dics[i]["height_list_" + str(height)].append("incorrect")

# print(dic)

for item in heights:
    # print(dic["height_list_" + str(item)].count("correct"))

    probability = dic_gauss["height_list_" + str(item)].count("correct") / len(
        dic_gauss["height_list_" + str(item)]
    )
    probability = probability * 100
    print(
        f"The probability of finding the correct critical density based on the Gauss fit for {item} time steps is {probability} %"
    )

    probability = dic_tri["height_list_" + str(item)].count("correct") / len(
        dic_tri["height_list_" + str(item)]
    )
    probability = probability * 100
    print(
        f"The probability of finding the correct critical density based on the Triangle fit for {item} time steps is {probability} %"
    )

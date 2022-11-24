#
# Determine the probabilities of inferring the correct critical density
# as a function of the number of time steps (system height).
# This is done using two fit functions through the car flow versus car density:
# - A Gaussian fit where "center" is estimated to be the critical density.
# - A Triangle fit where the the baseline of the triangle is the x-axis between densities 0 and 1
#   and the top has a coordinates "center" and "amplitude".
#   Also here, "center" is estimated to be the critical density.
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model


# define gaussian function
def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-((x - center) ** 2) / width)


# define triangle function
def triangle(x, center, amplitude):
    l_slope = amplitude / center
    r_slope = -1 * (amplitude / (1 - center))
    y = [
        amplitude - (l_slope * (center - i))
        if i <= center
        else amplitude - (r_slope * (center - i))
        for i in x
    ]
    return y


# create Gauss and Triangle model
mod_gauss = Model(gaussian)
mod_tri = Model(triangle)

# set initial parameters and constraints
params_gauss = mod_gauss.make_params(center=0.1, amplitude=0.25, width=0.5)
mod_gauss.set_param_hint("width", min=0, max=1)

params_tri = mod_tri.make_params(center=0.1, amplitude=0.25)
mod_tri.set_param_hint("center", min=0, max=1)

# create dictionary for all heights to put in all critical densities
heights = [5, 20, 100, 500, 1000, 2000]
dic_gauss = {}
dic_tri = {}
for item in heights:
    dic_gauss["height_list_" + str(item)] = []
    dic_tri["height_list_" + str(item)] = []

# save the two dictionaries as tuple
dics = (dic_gauss, dic_tri)

# indicate number of runs per simulation
R = 10

# for all the simulation runs
for dummy_var in range(10):

    # Read the csv files
    df = pd.read_csv(f"{dummy_var}_Car_flow_R{R}_0.csv")
    df = df.astype({"density": float, "height": int, "rep_num": int, "car_flow": float})

    # calculate critical density from gaussian fit (Based on plot_car_flow.py) for each height
    for height in np.unique(df["height"]):

        # initialize dataframe for the mean data
        mean_data = pd.DataFrame(
            [], columns=["density", "mean_car_flow", "err_car_flow"]
        )
        height_data = df[df["height"] == height]

        # calculate the mean car flow for each density and store in mean data
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

        # save fits as tuple
        fits = (fit_gauss, fit_tri)

        # plot once (every second simulation run) for every height
        if dummy_var == 1:

            # set up figure
            fig, axes = plt.subplots(1, 2, figsize=[12, 7])

            # plot separately the Gaussian and Triangle fits
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
                    mean_data["density"],
                    fits[i].init_fit,
                    "--",
                    label="initial fit",
                    color="black",
                )
                ax.plot(
                    mean_data["density"],
                    fits[i].best_fit,
                    "-",
                    label="best fit",
                    color="black",
                )

                # figure layout
                ax.set_title(f"Fit for height {height}")
                ax.grid()
                ax.set_xlabel(r"Initial car density $\rho$ [$L^{-1}$]")
                ax.set_ylabel("Car flow [$t^{-1}$]")
                ax.legend()

                i += 1

            plt.show()

        # indicate for every height if the estimated critical density is correct or incorrect
        # for both the Gauss and Triangle fits
        for i in range(len(dics)):
            if 0.45 <= fits[i].params["center"].value <= 0.55:
                dics[i]["height_list_" + str(height)].append("correct")
            else:
                dics[i]["height_list_" + str(height)].append("incorrect")


# lists of the probabilities for finding the correct critical density per height
prob_gauss = []
prob_tri = []


for item in heights:
    # calculate the probability as the fraction of correct critical densities

    # for the Gauss fit method
    probability = dic_gauss["height_list_" + str(item)].count("correct") / len(
        dic_gauss["height_list_" + str(item)]
    )
    probability = probability * 100
    prob_gauss.append(probability)
    print(
        f"The probability of finding the correct critical density based on the Gauss fit for {item} time steps is {probability} %"
    )

    # for the Triangle fit method
    probability = dic_tri["height_list_" + str(item)].count("correct") / len(
        dic_tri["height_list_" + str(item)]
    )
    probability = probability * 100
    prob_tri.append(probability)
    print(
        f"The probability of finding the correct critical density based on the Triangle fit for {item} time steps is {probability} %"
    )

# set up figure for the probability plots versus height
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5])

# plot the probabilities versus heights
ax1.plot(np.unique(heights), prob_gauss, "black", marker="o")
ax2.plot(np.unique(heights), prob_tri, "black", marker="o")
ax1.set_title("Gaussian fit")
ax2.set_title("Triangle fit")

# figure layout
for ax in (ax1, ax2):
    ax.set_xlabel("Number of time steps T")
    ax.set_ylabel(r"Probability of finding the correct $\rho_{crit}$")
    ax.grid()
    ax.set_ylim(bottom=0, top=105)
    ax.set_xscale("log")

plt.suptitle(
    r"Probabilities of finding the correct critical density $\rho_{crit}$ as a function of number of time steps T"
)

plt.savefig("critical_density_prob.png")

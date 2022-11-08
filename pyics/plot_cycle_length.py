#
# Plot the cycle length for different rule sets of the CA,
# with each Wolfram class in a different color.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wolfram_class(measurements, height=None):
    """Based on measurements of the cycle lengths and homogeneity for different rule sets,
    the Wolfram class is added to the measurements array.

    Args:
        measurements (array): Measurements of rule sets that have been simulated.
    Array consists of three columns:
    - rule: Rule set simulated (int).
    - cycle_length: Cycle length corresponding to the simulation (int).
    - homogeneous: Whether the simulation ended in a homogeneous state (True/False.
        height (int): Height of the simulation. If heigt is None, the maximum cycle length
        will be assumed to be the simulation height. Defaults to None.
    """

    # append column at the end for class
    zero_column = np.zeros([measurements.shape[0], 1], dtype=int)
    data = np.append(measurements, zero_column, axis=1)

    # determine height
    if height == None:
        height = np.max(data[:, 1])

    # determine class for each rule
    for rule in set(data[:, 0]):
        # create separate array containing only measurements of specific rule set
        rule_array = data[data[:, 0] == rule]
        rule_cl = rule_array[:, 1]
        rule_homogeneous = rule_array[:, 2]

        if rule_cl.mean() == height:
            wolfram_class = 3
        elif height in rule_cl:
            wolfram_class = 4
        elif rule_cl.mean() != 1:
            wolfram_class = 2
        elif all(i == True for i in rule_homogeneous):
            wolfram_class = 1
        else:
            wolfram_class = 2

        # change value of class column to wolfram class for the rows corresponding to the given rule
        data[:, 3] = np.where(data[:, 0] == rule, wolfram_class, data[:, 3])

    return data


def plot_CL(measurements, method="errorbar"):

    fig, ax = plt.subplots(1, 1, figsize=[10, 10])

    measurements = wolfram_class(measurements)

    mean_measurements = []
    std_measurements = []
    for rule in set(measurements[:, 0]):
        rule_array = measurements[measurements[:, 0] == rule]
        mean_measurements.append(np.mean(rule_array, axis=0))
        std_measurements.append(np.std(rule_array, axis=0))
    mean_measurements = np.array(mean_measurements)
    std_measurements = np.array(std_measurements)

    if method == "scatter":
        for w_class in set(measurements[:, 3]):
            # select data from wolfram_class
            class_data = measurements[measurements[:, 3] == w_class]
            ax.scatter(
                class_data[:, 0],
                class_data[:, 1],
                s=5,
                label="class = " + str(w_class),
            )

    if method == "errorbar":
        for w_class in set(mean_measurements[:, 3]):
            # select data from wolfram_class
            mean_class_data = mean_measurements[mean_measurements[:, 3] == w_class]
            std_class_data = std_measurements[mean_measurements[:, 3] == w_class]
            ax.errorbar(
                mean_class_data[:, 0],
                mean_class_data[:, 1],
                yerr=std_class_data[:, 1],
                fmt="o",
                capsize=2,
                label="class = " + str(w_class),
            )

    ax.set_xlabel("rule")
    ax.set_ylabel("cycle length")
    ax.set_yscale("log")
    ax.set_ylim(9e-1, 1e2)
    ax.legend()

    return fig, ax


# import data
data_CL = pd.read_csv("classes_10_0.csv")
data_hom = pd.read_csv("classes_10_1.csv")

# create measurements array in the right format
measurements = np.zeros([len(data_CL["rule"]), 3], dtype=int)
measurements[:, 0] = data_CL["rule"]
measurements[:, 1] = data_CL["cycle_length"]
measurements[:, 2] = data_hom["homogeneous"]

# plot cycle lengths against rule number for each class
fig, ax = plot_CL(measurements, "errorbar")

plt.savefig("cycle_lengths_iter_10.png")
plt.show()

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

    fig, ax = plt.subplots(1, 1, figsize=[10, 30])

    measurements = wolfram_class(measurements)

    mean_measurements = []
    std_measurements = []

    for rule in set(measurements[:, 0]):
        rule_array = measurements[measurements[:, 0] == rule]
        mean_measurements.append(np.mean(rule_array, axis=0))
        std_measurements.append(
            np.std(rule_array, axis=0) / np.sqrt(rule_array.shape[0])
        )

    mean_measurements = np.array(mean_measurements)
    std_measurements = np.array(std_measurements)

    # if method == "scatter":
    #     class_position = 3
    #     ticks = np.array([], dtype=int)
    #     labels = np.array([], dtype=int)
    #     for w_class in set(measurements[:, 3]):
    #         # select data from wolfram_class
    #         class_data = measurements[measurements[:, 3] == w_class]

    #         positions = np.arange(
    #             class_position, class_position + len(class_data[:, 0])
    #         )
    #         ticks = np.append(ticks, positions)
    #         labels = np.append(labels, class_data[:, 0])

    #         class_position = 3 + ticks[-1]

    #         ax.scatter(
    #             y=positions,
    #             x=class_data[:, 1],
    #             marker="D",
    #             label="class = " + str(int(w_class)),
    #         )

    #         print(f"Wolfram class {w_class}, rules: {set(class_data[:, 0])}")

    if method == "errorbar":
        class_position = 3
        ticks = np.array([], dtype=int)
        labels = np.array([], dtype=int)
        for w_class in set(mean_measurements[:, 3]):
            # select data from wolfram_class
            mean_class_data = mean_measurements[mean_measurements[:, 3] == w_class]
            std_class_data = std_measurements[mean_measurements[:, 3] == w_class]

            # determine positions within the plot and the corresponding ticks and labels
            positions = np.arange(
                class_position, class_position + len(mean_class_data[:, 0])
            )
            ticks = np.append(ticks, positions)
            labels = np.append(labels, mean_class_data[:, 0])

            # set starting position for the next class
            class_position = 3 + ticks[-1]

            # plot class
            ax.errorbar(
                y=positions,
                x=mean_class_data[:, 1],
                xerr=std_class_data[:, 1],
                fmt="D",
                ms=4,
                capsize=2,
                label="class = " + str(int(w_class)),
            )

            # print rules belonging to class
            print(f"Wolfram class {int(w_class)}, rules: {set(mean_class_data[:, 0])}")

    ax.set_yticks(ticks)
    ax.set_yticklabels(np.array(labels, dtype=int), rotation=0, size=5.7)
    ax.grid()
    ax.set_ylabel("rule")
    ax.set_xlabel("cycle length")
    ax.set_xscale("log")
    ax.set_xlim(9e-1, 2 * np.max(mean_measurements[:, 1]))
    ax.set_ylim(np.min(ticks) - 5, np.max(ticks) + 5)
    ax.invert_yaxis()
    ax.legend()

    return fig, ax


def plot_CL_wolfram(cl_data, wolfram_classes, method="errorbar"):

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=[10, 30])

    # determine mean and standarddeviation per rule
    mean_cl = []
    std_cl = []

    for rule in set(cl_data["rule"]):
        rule_array = cl_data[cl_data["rule"] == rule]
        mean_cl.append(np.mean(rule_array, axis=0))
        std_cl.append(np.std(rule_array, axis=0) / np.sqrt(rule_array.shape[0]))

    mean_cl = np.array(mean_cl)
    std_cl = np.array(std_cl)

    if method == "errorbar":

        # position in plot for first class
        class_position = 3

        # store ticks and labels
        ticks = np.array([], dtype=int)
        labels = np.array([], dtype=int)

        #
        for w_class in set(wolfram_classes["class"]):

            # select data for rules corresponding to the wolfram class
            mean_class_data = mean_cl[wolfram_classes["class"] == w_class]
            std_class_data = std_cl[wolfram_classes["class"] == w_class]

            # determine positions within the plot and the corresponding ticks and labels
            positions = np.arange(
                class_position, class_position + len(mean_class_data[:, 0])
            )
            ticks = np.append(ticks, positions)

            # add rule numbers to the labels
            labels = np.append(labels, mean_class_data[:, 0])

            # set starting position for the next class
            class_position = 3 + ticks[-1]

            # plot class
            ax.errorbar(
                y=positions,
                x=mean_class_data[:, 2],
                xerr=std_class_data[:, 2],
                fmt="D",
                ms=4,
                capsize=2,
                label="class = " + str(int(w_class)),
            )

            # print rules belonging to class
            print(f"Wolfram class {int(w_class)}, rules: {set(mean_class_data[:, 0])}")

    ax.set_yticks(ticks)
    ax.set_yticklabels(np.array(labels, dtype=int), rotation=0, size=5.7)
    ax.grid()
    ax.set_ylabel("rule")
    ax.set_xlabel("cycle length")
    ax.set_xscale("log")
    ax.set_xlim(9e-1, 2 * np.max(mean_cl))
    ax.set_ylim(np.min(ticks) - 5, np.max(ticks) + 5)
    ax.invert_yaxis()
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

# import wolfram classes per rule
rule_wolfram_class = pd.read_csv("rule_class_wolfram.csv")
test = rule_wolfram_class["class"]
print(test)

fig, ax = plot_CL_wolfram(
    data_CL,
    rule_wolfram_class,
)

plt.savefig("cl_wclass_iter_10.png")
plt.show()

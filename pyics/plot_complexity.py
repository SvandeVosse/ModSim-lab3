#
# Plot the complexity versus the Langton parameter for each rule of the CA
# with each Wolfram Class in a different color.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_Shannon(
    rules, langton, cell_Shannon, row_Shannon, local_Shannon, wolfram_classes
):
    """plots the shannon entropy of a rule as a function of the langton parameter."""

    fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=[20, 8])

    for wclass in set(wolfram_classes):

        # looks up rules belonging to a certain wolfram class
        class_rules = np.where(wolfram_classes == wclass)[0]

        indeces = []
        # creates a list with indices for rules of a wolfram class
        for class_rule in class_rules:
            rule_indeces = np.where(rules == class_rule)[0]
            indeces = indeces + (list(rule_indeces))

        # makes the lists for langton parameter values coresponding to the rules of that wolfram class
        class_langton = langton[indeces]
        # makes the lists for entropies coresponding to the rules of that wolfram class
        class_cell_Shannon = cell_Shannon[indeces]
        class_row_Shannon = row_Shannon[indeces]
        class_local_Shannon = local_Shannon[indeces]

        # scatter plot of the entropies on the y-axis and the langton parameter on the x-axis
        ax1.scatter(
            class_langton,
            class_cell_Shannon,
            s=3,
            label="class " + str(wclass),
        )
        ax2.scatter(class_langton, class_row_Shannon, s=3, label="class " + str(wclass))

        ax4.scatter(
            class_langton, class_local_Shannon, s=3, label="class " + str(wclass)
        )

    ax1.set_xlabel("Langton parameter $\lambda$")
    ax1.set_ylabel(r"Shannon entropy for cells $\bar{H}_c$")
    ax1.legend()

    ax2.set_xlabel("Langton parameter $\lambda$")
    ax2.set_ylabel(r"Shannon entropy for rows $\bar{H}_r$")
    ax2.legend()

    ax4.set_xlabel("Langton parameter $\lambda$")
    ax4.set_ylabel(r"Shannon entropy for local configurations $\bar{H}_{loc}$")
    ax4.legend()

    return fig, (ax1, ax2, ax4)


rules_df = pd.read_csv("ShannonLangton_100_0.csv")
# reads out the values for the shannon entropy calculated with different methods
langton_df = pd.read_csv("ShannonLangton_100_1.csv")
cell_df = pd.read_csv("ShannonLangton_100_2.csv")
row_df = pd.read_csv("ShannonLangton_100_3.csv")
local_df = pd.read_csv("ShannonLangton_100_4.csv")

wolfram_class = pd.read_csv("rule_class_wolfram.csv")

rules = rules_df["rule"]
langton = langton_df["langton"]
cell_Shannon = cell_df["cell_Shannon"]
row_Shannon = row_df["row_Shannon"]
local_Shannon = local_df["local_config_Shannon"]

# plot of rules selected on their langton parameter plotted against their shannon entropy,
# wolfram classes are distinguished by colour.
fig, axes = plot_Shannon(
    rules, langton, cell_Shannon, row_Shannon, local_Shannon, wolfram_class["class"]
)
plt.show()

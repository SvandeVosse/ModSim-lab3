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

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[15, 15])

    for wclass in set(wolfram_classes):

        class_rules = np.where(wolfram_classes == wclass)[0]
        print(class_rules)

        indeces = []
        for class_rule in class_rules:
            rule_indeces = np.where(rules == class_rule)[0]
            indeces = indeces + (list(rule_indeces))

        print(f"{wclass=}")
        print(f"{indeces=}")

        class_langton = langton[indeces]
        class_cell_Shannon = cell_Shannon[indeces]
        class_row_Shannon = row_Shannon[indeces]
        class_local_Shannon = local_Shannon[indeces]

        ax1.scatter(
            class_langton,
            class_cell_Shannon,
            s=3,
            label="class " + str(wclass),
        )
        ax2.scatter(class_langton, class_row_Shannon, s=3, label="class " + str(wclass))
        ax3.scatter(
            class_langton,
            class_cell_Shannon + class_row_Shannon,
            s=3,
            label="class " + str(wclass),
        )
        ax4.scatter(
            class_langton, class_local_Shannon, s=3, label="class " + str(wclass)
        )

    ax1.set_xlabel("Langton parameter $\lambda$")
    ax1.set_ylabel(r"Shannon entropy for cells $\bar{H}_c$")
    ax1.legend()

    ax2.set_xlabel("Langton parameter $\lambda$")
    ax2.set_ylabel(r"Shannon entropy for rows $\bar{H}_r$")
    ax2.legend()

    ax3.set_xlabel("Langton parameter $\lambda$")
    ax3.set_ylabel(r"Shannon entropy for cells and rows $\bar{H}_{cr}$")
    ax3.legend()

    ax4.set_xlabel("Langton parameter $\lambda$")
    ax4.set_ylabel(r"Shannon entropy for local configurations $\bar{H}_{loc}$")
    ax4.legend()

    return fig, ((ax1, ax2), (ax3, ax4))


rules_df = pd.read_csv("ShannonLangton_1_0.csv")
langton_df = pd.read_csv("ShannonLangton_1_1.csv")
cell_df = pd.read_csv("ShannonLangton_1_2.csv")
row_df = pd.read_csv("ShannonLangton_1_3.csv")
local_df = pd.read_csv("ShannonLangton_1_4.csv")

wolfram_class = pd.read_csv("rule_class_wolfram.csv")

rules = rules_df["rule"]
langton = langton_df["langton"]
cell_Shannon = cell_df["cell_Shannon"]
row_Shannon = row_df["row_Shannon"]
local_Shannon = local_df["local_config_Shannon"]


fig, axes = plot_Shannon(
    rules, langton, cell_Shannon, row_Shannon, local_Shannon, wolfram_class["class"]
)
plt.show()

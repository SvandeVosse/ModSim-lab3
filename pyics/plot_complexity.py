#
# Plot the complexity versus the Langton parameter for each rule of the CA
# with each Wolfram Class in a different color.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_Shannon(rules, langton, cell_Shannon, row_Shannon, wolfram_classes):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[15, 5])

    ax1.scatter(langton, cell_Shannon)

    ax2.scatter(langton, row_Shannon)

    ax3.scatter(langton, cell_Shannon + row_Shannon)

    return fig, (ax1, ax2, ax3)


cell_df = pd.read_csv("ShannonLangton_1_0.csv")
row_df = pd.read_csv("ShannonLangton_1_1.csv")
langton_df = pd.read_csv("ShannonLangton_1_2.csv")
wolfram_class = pd.read_csv("rule_class_wolfram.csv")

rules = langton_df["rule"]
cell_Shannon = cell_df["cell_Shannon"]
row_Shannon = row_df["row_Shannon"]
langton = langton_df["langton"]

fig, axes = plot_Shannon(
    rules, langton, cell_Shannon, row_Shannon, wolfram_class["class"]
)
plt.show()

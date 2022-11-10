#
# Code for running CA simulations.
# In order to analyse and show results, plot_cycle_length.py should be run.
#

import numpy as np

from pyics import Model


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    k_base_string = np.base_repr(n, k)
    return np.array([*k_base_string], dtype=int)


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param("r", 1)
        self.make_param("k", 2)
        self.make_param("width", 50)
        self.make_param("height", 50)
        self.make_param("rule", 30, setter=self.setter_rule)

        self.cycle_length = self.height  # minimal cycle length (for aperiodic patterns)
        self.homogeneous = False

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""

        # determine the rule in k-base
        size = self.k ** (2 * self.r + 1)
        k_base_n = decimal_to_base_k(self.rule, self.k)

        # set rule_set as array according to k-base rule
        rule = np.zeros([size], dtype=int)
        rule[-len(k_base_n) :] = k_base_n
        self.rule_set = rule

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        # transform input array to string
        string_inp = ""
        for i in inp:
            string_inp += str(i)

        # transform input configuration to decimal base
        deci_base_inp = int(string_inp, self.k)

        # determine next state according to the rule_set
        new_state = np.flip(self.rule_set)[deci_base_inp]

        return new_state

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""

        single_seed = False

        # create initial row as a single seed of state k in the middle or as a random configuration
        if single_seed == True:
            init_row = np.zeros(self.width, dtype=int)
            init_row[int(len(init_row) / 2)] = self.k - 1
        else:
            init_row = np.random.randint(0, self.k, self.width, dtype=int)

        return init_row

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width], dtype=int)
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(
            self.config,
            interpolation="none",
            vmin=0,
            vmax=self.k - 1,
            cmap=matplotlib.cm.binary,
        )
        plt.axis("image")
        plt.title("t = %d" % self.t)

    def find_cycle_length(self):
        """Find cycle length by checking if last rule occurs before in the configuration."""

        # if no periodicity is found, the height of the configuration is defined to be the cycle length.
        self.cycle_length = self.height

        # start comparing the last row to the second to last row.
        i = self.height - 2

        # periodicity is False until proven True
        periodic = False
        while periodic == False and i >= 0:
            # check if rows are the same
            if list(self.config[-1]) == list(self.config[i]):
                self.cycle_length = self.height - 1 - i
                periodic = True
            # update row number to check
            i -= 1

    def check_homogeneity(self):
        """Check homogeneity of the configuration by checking if the last two rows consist of all the same values."""

        # check if the configuration is homogeneous.
        self.homogeneous = all(
            x == self.config[-1, 0]
            for x in np.reshape(self.config[-2:], [2 * self.config.shape[1], 1])
        )

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            # update cycle length and homogeneity
            self.find_cycle_length()
            self.check_homogeneity()
            if self.homogeneous == False:
                homo_string = "and it is not homogeneous."
            else:
                homo_string = "and it is homogeneous."
            # print(
            #     f"cycle length of rule {self.rule} is {self.cycle_length}", homo_string
            # )
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [
                i % self.width for i in range(patch - self.r, patch + self.r + 1)
            ]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)


if __name__ == "__main__":

    # make CA simulation instance
    sim = CASim()

    # import relevant modules
    from pyics import GUI
    from pyics import paramsweep

    # indicate if parameter measurements should be run
    paramsimulate = False

    if paramsimulate == True:

        # determine parameter space to simulate
        param_space = {
            "rule": list(
                range(
                    getattr(sim, "k")
                    ** getattr(sim, "k")
                    ** (2 * getattr(sim, "r") + 1),
                )
            )
        }

        # set the simulation parameters
        sim.width = 50
        sim.height = 4 * sim.width
        N_sim = 10

        # perform simulations and save measurements on given csv base filename
        measurements = paramsweep(
            sim,
            N_sim,
            param_space=param_space,
            measure_attrs=["cycle_length", "homogeneous"],
            measure_interval=0,
            csv_base_filename="classes_" + str(N_sim),
        )

    # start up GUI
    cx = GUI(sim)
    cx.start()

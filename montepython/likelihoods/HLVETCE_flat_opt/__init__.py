import os
import numpy as np
from montepython.likelihood_class import Likelihood


class HLVETCE_flat_opt(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.measdL = np.array([], 'float64')
        self.errdL = np.array([], 'float64')

        # read redshifts and data points
        for line in open(os.path.join(
                self.data_directory, self.data_file), 'r'):
            if (line.find('#') == -1):
                self.z = np.append(self.z, float(line.split()[0]))
                self.measdL = np.append(self.measdL, float(line.split()[1]))
                self.errdL = np.append(self.errdL, float(line.split()[3]))

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.

        # for each point infer
        # theoretical prediction and difference with observation

        for i in range(self.num_points):

            theodL = cosmo.luminosity_distance_gw(self.z[i])

            chi2 += ((theodL -self.measdL[i]) / (self.errdL[i])) ** 2

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl

# ---------------------------------------------------------------------------- #
"""
WavelengthFit is a class describing the functional form for transforming pixel
position to wavelength.  The inputs for this task are the given pixel position
and the corresponding wavelength, as well as an input functional form and order
for that form.  The class then calculates the coefficients for that form.
Options for the wavelength fit are polynomial, spline, chebyshev and legendre.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import math
# numpy import
import numpy as np
# Application imports
# - saltreduce.functions
from ...functions import Fit1D

# ---------------------------------------------------------------------------- #
class WavelengthFit:
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
    def __init__(self, p, w, function='spline', order=5, niter=5, thresh=3):
# ---------------------------------------------------------------------------- #

        self.p = p
        self.w = w

        self.thresh = thresh
        self.niter = niter
        self.order = order
        self.function = function

        # Set function
        self.func = Fit1D(p, w, function=function, order=order, niter=niter,
                          low_reject=thresh, high_reject=thresh)

        # Set coefficients
        if self.func.func is None:
            self.coef = self.func.coef

        else:
            self.coef = self.func.func.parameters

# ---------------------------------------------------------------------------- #
    def set_coef(self, coef):
# ---------------------------------------------------------------------------- #

        if self.func.func is None:
            self.func.coef = coef
            self.coef = self.func.coef

        else:
            self.func.func.parameters = coef
            self.coef = self.func.func.parameters

# ---------------------------------------------------------------------------- #
    def value(self, x):
# ---------------------------------------------------------------------------- #

        return self.func(x)

# ---------------------------------------------------------------------------- #
    def res(self, x, y):
# ---------------------------------------------------------------------------- #

        """Return the residuals of the fit"""

        # Calculate residuals
        residuals = np.subtract(y, self.value(x))

        return residuals

# ---------------------------------------------------------------------------- #
    def rms(self, x, y):
# ---------------------------------------------------------------------------- #

        """Return the RMS of the fit"""

        # Calculate residuals
        residuals = self.res(x, y)
        # Calculate the mean of the squared residuals (Mean Square Error)
        mse = np.square(residuals).mean()
        # calculate the square root of mse (Root Mean Square Error)
        rmse = math.sqrt(mse)

        return rmse

# ---------------------------------------------------------------------------- #
    def sigma(self, x, y):
# ---------------------------------------------------------------------------- #

        """Return the SIGMA, RMS and RESIDUALS of the fit"""

        # Calculate residuals
        residuals = y - self.value(x)
        # Calculate rms
        rms = (((residuals) ** 2).mean()) ** 0.5

        # If there many data points...
        if len(x) >= 4:
            # Get average distance between 16th and 84th percentiles of the
            # residuals divided by 2 - should be less sensitive to outliers:
            # - sort residuals
            residuals = np.sort(residuals)
            # - get correct indices and take difference
            sig = (residuals[int(0.84 * len(residuals))] -
                   residuals[int(0.16 * len(residuals))]) / 2.

        else:
            # Else return the RMS
            sig = rms

        return sig, rms, residuals

# ---------------------------------------------------------------------------- #
    def chisq(self, x, y, err):
# ---------------------------------------------------------------------------- #

        """Return the chi^2 of the fit"""

        return (((y - self.value(x)) / err) ** 2).sum()

# ---------------------------------------------------------------------------- #
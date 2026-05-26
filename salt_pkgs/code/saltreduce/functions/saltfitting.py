# ---------------------------------------------------------------------------- #
"""
SALT general utilities:
- saltfitting provides general fitting of arrays for SALT tasks. It provides
  non-interactiving fitting of 1- and 2-D arrays and allows fitting of general
  functions to the parameters.
"""
# ---------------------------------------------------------------------------- #

# numpy import
import numpy as np
# scipy imports
from scipy.interpolate import splrep
from scipy.interpolate import splev
# astropy imports
from astropy.modeling.models import Polynomial1D
from astropy.modeling.models import Chebyshev1D
from astropy.modeling.models import Legendre1D
from astropy.modeling.fitting import LinearLSQFitter

# ---------------------------------------------------------------------------- #

MYNAME = 'saltfitting'

# ---------------------------------------------------------------------------- #
class Fit1D:
# ---------------------------------------------------------------------------- #

    """
    Given x and y data arrays, find best fitting curve. After initial fit,
    iterate on solution to reject any points outside threshold for solution.

    * x - list or array of x data
    * y - list or array of y data
    * yerr - error on y data
    * coef - initial coefficients for fit
    * function - function to fit to the data:
                 polynomial (poly), chebyshev, or legendre
    * order - order of the function that is fit
    * low_reject - lower rejection threshold (units=sigma)
    * high_reject - upper rejection threshold (units=sugma)
    * niter - number of times to iterate
    """

    def __init__(self, x, y, yerr=None, function='poly', coef=None, order=3,
                 low_reject=3, high_reject=2, niter=5, debug=False):
        """Set up the variables"""

        self.x_orig = x
        self.y_orig = y
        self.npts = len(self.x_orig)
        self.debug = debug

        if yerr is None:
            self.yerr_orig = np.ones(self.npts)

        else:
            self.yerr_orig = yerr

        self.order = order
        self.niter = niter
        self.low_reject = low_reject
        self.high_reject = high_reject

        self.set_func(function)
        self.set_coef(coef)
        self.mask = np.ones(self.npts, dtype=bool)

        if len(x) > 0 and len(y) > 0 and coef is None:
            self.set_arrays(self.x_orig, self.y_orig, err=self.yerr_orig)
            self.fit()

        return

    def set_func(self, function):
        """Set the function that will be used"""

        self.function = function

        if self.function in ['poly', 'polynomial']:
            self.func = Polynomial1D(self.order)

        elif self.function == 'chebyshev':
            self.func = Chebyshev1D(self.order)

        elif self.function == 'legendre':
            self.func = Legendre1D(self.order)

        else:
            err = '{0} is not a valid function.'.format(self.function)
            raise SALTError(err)

        return

    def set_coef(self, coef=None):
        """Set the coefficients for fits of poly, chebyshev and legendre"""

        if coef is None:
            coef = np.ones(self.order + 1)

        self.coef = coef

        if self.func is not None:
            self.func.parameters = coef

        return

    def set_mask(self):
        """Set the mask according to the values for rejecting points"""

        # Check high and low reject values
        if self.high_reject == 0 and self.low_reject == 0:
            return

        # Get residuals for y and fit arrays
        res = self.y - self(self.x)
        # Calculate sigma of residuals for current x and y arrays
        sigma = self.sigma(self.x, self.y)
        # Set high reject masK
        high_reject_mask = res < self.high_reject * sigma
        # Set low reject mask
        low_reject_mask = res > -self.low_reject * sigma
        # Combine masks
        self.mask = (high_reject_mask) * (low_reject_mask)

        return

    def set_arrays(self, x, y, err=None):
        """Set the arrays using a mask"""
        self.x = x[self.mask]
        self.y = y[self.mask]
        if err is not None:
            self.yerr = err[self.mask]

        return

    def sigma(self, x, y):
        """Return the RMS of the fit """

        return (((y - self(x))**2).mean())**0.5

    def chisq(self, x, y, err):
        """Return the chi^2 of the fit"""

        return (((y - self(x)) / err)**2).sum()

    def fit(self, task=0, s=None, t=None):
        """Fit the function to the data"""

        # Use Linear Least Square fitter
        fitter = LinearLSQFitter()

        # Do iterations
        for i in range(self.niter):

            try:
                # Do linear least squares fit
                self.func = fitter(self.func, self.x, self.y)

            except Exception as error:
                # Check if at least 1 iteration was completed
                if i > 0:
                    # Get out!
                    break
                else:
                    raise SALTError(error)

            # Check if coefficients is same-same: Break
            if self.coef.tolist() == self.func.parameters.tolist(): break

            self.set_mask()
            self.set_arrays(self.x, self.y)
            # Update coefficients
            self.coef = self.func.parameters

        return

    def __call__(self, x):
        """Return the value of the function evaluated at x"""

        return self.func(x)

# ---------------------------------------------------------------------------- #
class CurFit:
# ---------------------------------------------------------------------------- #

    """Given x and y data arrays, find the best fitting curve

    * x - list or array of x data
    * y - list or array of y data
    * yerr - error on y data
    * coef - initial coefficients for fit
    * function - function to be fit to the data:
                 options include polynomial, chebyshev, legendre, or spline
    * order - order of the function that is fit
    """

    def __init__(self, x, y, yerr=None, function='poly', coef=None, order=3):
        """Set up the variables"""

        self.x = x
        self.y = y

        if yerr is None:
            self.yerr = np.ones(len(y))

        else:
            self.yerr = yerr

        self.order = order
        self.set_func(function)
        self.set_coef(coef)

        return

    def set_func(self, function):
        """Set the function that will be used"""

        self.function = function

        if self.function in ['poly', 'polynomial']:
            self.func = Polynomial1D(self.order)

        elif self.function == 'chebyshev':
            self.func = Chebyshev1D(self.order)

        elif self.function == 'legendre':
            self.func = Legendre1D(self.order)

        elif self.function == 'spline':
            self.func = None

        else:
            err = '{0} is not a valid function.'.format(self.function)
            raise SALTError(err)

        return

    def set_coef(self, coef=None):
        """Set the coefficients for fits of poly, chebyshev and legendre"""

        if coef is None:
            coef = np.ones(self.order + 1)

        self.coef = coef

        if self.func is not None:
            self.func.parameters = coef

        return

    def set_weight(self, err):
        """Set the weighting for spline fitting """

        self.weight = None
        if isinstance(err, np.ndarray):
            if err.any() != 0:
                self.weight = 1 / err

        return

    def sigma(self, x, y):
        """Return the RMS of the fit """

        return (((y - self(x))**2).mean())**0.5

    def chisq(self, x, y, err):
        """Return the chi^2 of the fit"""

        return (((y - self(x)) / err)**2).sum()

    def fit(self, task=0, s=None, t=None, full_output=1):
        """Fit the function to the data"""

        if self.function == 'spline':
            self.set_weight(self.yerr)
            self.results = splrep(self.x, self.y, k=self.order, w=self.weight,
                                  task=task, s=s, t=t, full_output=full_output)
            self.set_coef(self.results[0])

        else:
            fitter = LinearLSQFitter()
            self.func = fitter(self.func, self.x, self.y)
            self.coef = self.func.parameters

        return

    def __call__(self, x):
        """Return the value of the function evaluated at x"""

        if self.function == 'spline': return splev(x, self.coef, der=0)

        return self.func(x)

# ---------------------------------------------------------------------------- #
class InterFit(CurFit):
# ---------------------------------------------------------------------------- #

    """
    Given x and y data arrays, find best fitting curve. After initial fit,
    iterate on solution to reject any points outside threshold for solution.

    * x - list or array of x data
    * y - list or array of y data
    * yerr - error on y data
    * coef - initial coefficients for fit
    * function - function to be fit to the data:
                 polynomial, chebyshev, legendre, or spline
    * order - order of the function that is fit
    * niter - number of times to iterate
    * thresh - threshold for rejection
    """

    def __init__(self, x, y, yerr=None, coef=None, function='poly', order=3,
                 niter=5, thresh=3, full_output=1):
        """Set up the variables"""

        self.x = x
        self.y = y

        if yerr is None and function != 'spline':
            self.yerr = np.ones(len(y))

        else:
            self.yerr = yerr

        self.x_orig = self.x
        self.y_orig = self.y
        self.yerr_orig = self.yerr

        self.thresh = thresh
        self.niter = niter
        self.order = order
        self.set_func(function)
        self.set_coef(coef)

        if coef is None: self.interfit(full_output=full_output)

        return

    def set_mask(self):
        """Set the mask according to the values for rejecting points"""

        self.mask = np.ones(len(self.x), dtype=bool)
        # difference the arrays
        diff = self.y_orig - self.func(self.x_orig)
        sigma = self.sigma(self.x, self.y)
        self.mask = (abs(diff) < self.thresh * sigma)

        return

    def interfit(self, full_output=1):
        """Fit a function and then iterate it to reject possible outliers"""

        if self.function == 'spline':
            self.set_weight(self.yerr)
            self.results = splrep(self.x, self.y, k=self.order, w=self.weight,
                                  full_output=full_output)
            if full_output == 1:
                self.set_coef(self.results[0])

            else:
                self.set_coef(self.results)

        else:
            # Use Iterated Reweighted Least Squares Method: robust to outliers
            fitter = LinearLSQFitter()
            # Initialise weights to 1's
            weights = np.ones(len(self.x))

            # Do iterations
            for _ in range(self.niter):

                # Do linear least squares fit
                self.func = fitter(self.func, self.x, self.y, weights=weights)

                # Check if coefficients is same-same: Break
                if self.coef.tolist() == self.func.parameters.tolist(): break

                # Calculate residuals
                r = (self.y - self.func(self.x)) / self.yerr
                # Calculate median absolute deviation and normalise to 50%
                # confidence level (0.6745 sigma for gaussian)
                s = np.median(abs(r - np.median(r))) / 0.6745

                # Check if s is zero: Break
                if s == 0: break

                # Set biweights function
                biweights = lambda x: ((abs(x) < self.thresh) * 1.0)
                # Recalculate weights (using biweights)
                weights = biweights(r / s)

                # Update mask
                self.mask = (weights > 0)
                # Update coefficients
                self.coef = self.func.parameters

        return

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #
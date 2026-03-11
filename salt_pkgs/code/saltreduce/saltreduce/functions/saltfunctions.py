# ---------------------------------------------------------------------------- #
"""
SALT general utilities:
- saltfunctions provides general functions for SALT tasks.
"""
# ---------------------------------------------------------------------------- #

# numpy import
import numpy as np

# ---------------------------------------------------------------------------- #

MYNAME = 'saltfunctions'

# ---------------------------------------------------------------------------- #
def normal_distribution(arr, mean, sigma, scale=1):
# ---------------------------------------------------------------------------- #

    """
    Return the normal distribution of N dimensions

    arr: input array of N dimensions
    mean: mean of the distribution - scalar or N-dimensional array
    sigma: std deviation of the distribution - scalar or N-dimensional array
    scale: scale of the distrubion
    """

    arr = arr
    mean = mean
    sigma = sigma
    scale = scale

    # Set dimensions
    dim = np.ndim(arr) - 1

    # Check if mean is an array
    if isinstance(mean, np.ndarray):
        # Check if mean array dimensions match input array
        if len(mean) == dim:
            # Reshape mean array to calculate the distribution
            mean = np.reshape(mean, (dim, 1, 1))

        else:
            msg = 'Mean and input array are different number of dimensions.'
            raise SALTError(msg)

    # Check if sigma is an array
    if isinstance(sigma, np.ndarray):
        # Check if sigma array dimensions match input array
        if len(sigma) == dim:
            # Reshape sigma array to calculate the distribution
            sigma = np.reshape(sigma, (dim, 1, 1))

        else:
            msg = 'Sigma and input array are different number of dimensions.'
            raise SALTError(msg)

    # Calculate the gaussian
    z = scale * np.exp(-0.5 * (arr - mean) ** 2 / sigma)

    return z

# ---------------------------------------------------------------------------- #
def get_evenly_spaced_array(start, stop, size, precision=8):
# ---------------------------------------------------------------------------- #

    """
    Calculate an evenly spaced array.
    """

    a, da = np.linspace(start, stop, num=size, retstep=True, dtype=float)
    a = np.around(a, decimals=precision)
    a1 = a.min()
    da = np.around(da, precision)

    return a, a1, da

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# """
# Here's the full text from Morton 1991 ApJS 77, 119 and it is a good question
# whether the data from Peck and Reeder must be adopted as IR is important.

# The IAU standard for conversion between air and vacuum wavelengths according
# to Edlen (1953) and Oosterhoff (1957), is

# \begin{equation}
# \frac{\lambda_{vac}-\lambda_{air}}{\lambda_{air}}=(n-1) =
#    6.4328\times10^{-5}+ \frac{2.94981 times10^{-2}}{146-\sigma^2}
#    +\frac{2.5540\times10^{-4}}{41-\sigma^2}
# \end{equation}
# where $\sigma=10^4/$, wiht $\lambda$ in angstroms.

#  Edlen (1966) and Peck & Reeder (1972) have proposed improvements that
#  primarily affect infrared wavelengths, but equation (3) was used here.

# More recently, this has been updated in IDLASTRO with a formula from
# Ciddor 1996, Applied Optics 62, 958

# See http://idlastro.gsfc.nasa.gov/ftp/pro/astro/airtovac.pro
# """
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
def air_to_vac(w_air, mode='Morton'):
# ---------------------------------------------------------------------------- #

    """
    Convert 'in air' wavelength to 'in vacuum' wavelength

    w -- wavelength in air in units of angstrom
    mode -- method to use for conversion
       Morton -- Morton 1991 ApJS 77, 119
       Ciddor -- Ciddor 1996, Applied Optics 62, 958
    """

    if mode == 'Morton':
        sigmasq = (1e4 / w_air) ** 2
        w_vac = w_air * (1 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigmasq) + 2.5540e-4 / (41.0 - sigmasq))

    elif mode == 'Ciddor':
        sigmasq = (1e4 / w_air) ** 2
        w_vac = w_air * (1 + 5.792105e-2 / (238.0185 - sigmasq) + 1.67917e-3 / (57.362 - sigmasq))

    return w_vac

# ---------------------------------------------------------------------------- #
def vac_to_air(w_vac, mode='Morton'):
# ---------------------------------------------------------------------------- #

    """
    Convert 'in vacuum' wavelength to 'in air' wavelength

    w -- wavelength in vacuum in units of angstrom
    mode -- method to use for conversion
       Morton -- Morton 1991 ApJS 77, 119
       Ciddor -- Ciddor 1996, Applied Optics 62, 958
    """

    if mode == 'Morton':
        sigmasq = (1e4 / w_vac) ** 2
        w_air = w_vac / (1 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigmasq) + 2.5540e-4 / (41.0 - sigmasq))

    elif mode == 'Ciddor':
        sigmasq = (1e4 / w_vac) ** 2
        w_air = w_vac / (1 + 5.792105e-2 / (238.0185 - sigmasq) + 1.67917e-3 / (57.362 - sigmasq))

    return w_air

# ---------------------------------------------------------------------------- #
def fnu_to_flamda(warr, farr):
# ---------------------------------------------------------------------------- #

    """
    Convert farr in ergs/s/cm2/Hz to ergs/s/cm2/A
    """

    # Speed of light in Angstroms/s
    c = 2.99792458e18
    return farr * c / warr ** 2

# ---------------------------------------------------------------------------- #
def mag_to_flux(marr, fzero):
# ---------------------------------------------------------------------------- #

    """
    Convert from magnitude to flux.
    marr--input array in magnitude
    fzero--zero point for the conversion
    """
    return fzero * 10 ** (-0.4 * marr)

# ---------------------------------------------------------------------------- #
def flux_to_mag(farr, fzero):
# ---------------------------------------------------------------------------- #

    """"
    Convert from flux to magnitude.
    farr--input array in flux units
    fzero--zero point for the converion
    """
    return -2.5 * np.log10(farr / fzero)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #
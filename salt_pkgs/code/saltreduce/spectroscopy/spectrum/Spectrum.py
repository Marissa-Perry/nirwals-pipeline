# ---------------------------------------------------------------------------- #
"""
Spectrum is a class to describe and generate a spectrum. It can either be
generated from a line list or a continuum flux.
"""
# ---------------------------------------------------------------------------- #

# numpy import
import numpy as np

# Application imports
# - saltreduce.functions
from ...functions import normal_distribution

# ---------------------------------------------------------------------------- #
class Spectrum:
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
    def __init__(self, wavelength, flux, wrange=None, res=1., dres=0.1,
                 stype='line'):
# ---------------------------------------------------------------------------- #

        """
        wavelength: <numpy array> 1D wavelength array
        flux: <numpy array> 1D flux array
        wrange: <list> wavelength range
        res: <float> ?
        dres: <float> sampling of the spectrum
        stype: <str> spectrum type:
                     line - input spectrum is a list of lines
                     continuum - input spectrum is continuum values
        """

        # Set variables
        self.wrange = wrange
        self.res = res
        self.dres = dres
        self.stype = stype

        # Set wavelength
        self.set_wavelength(wavelength)
        # Set flux
        self.set_flux(wavelength, flux)

        return

# ---------------------------------------------------------------------------- #
    def set_wavelength(self, wavelength):
# ---------------------------------------------------------------------------- #

        """Set the wavelength"""

        if self.wrange is None:
            self.wrange = [wavelength.min(), wavelength.max()]

        if self.stype == 'line':
            self.wavelength = np.arange(
                self.wrange[0], self.wrange[1], self.dres)

        else:
            self.wavelength = wavelength

# ---------------------------------------------------------------------------- #
    def set_flux(self, wavelength, flux):
# ---------------------------------------------------------------------------- #

        """Set the flux"""

        if flux is not None and len(flux) == len(self.wavelength):
            # Set flux: as is
            self.flux = flux

        elif flux is not None and wavelength is not None:
            # Check spectrum type
            if self.stype == 'line':
                # Initialise flux
                self.flux = np.zeros(len(self.wavelength), dtype=float)

                # Loop for wavelength and flux points...
                for w, f in zip(wavelength, flux):

                    self.flux += normal_distribution(
                        self.wavelength, w, self.res * self.dres, f)

            else:
                # Set flux: interpolated
                self.flux = np.interp(self.wavelength, wavelength, flux)

        else:
            # Set flux: zero
            self.flux = np.zeros(len(wavelength), dtype=float)

# ---------------------------------------------------------------------------- #
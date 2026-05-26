"""
LongSlit is a class that describes a typical long-slit spectrograph, e.g., the
optical arm of the Robert Stobie Spectrograph and the fibre-fed Near InfraRed
Spectrograph (pseudo long-slit). LongSlit inherits from Spectrograph.
"""

# numpy import
import numpy as np

# Application imports
from .Spectrograph import Equations
from .Spectrograph import Optics
from .Spectrograph import Slit
from .Spectrograph import Grating
from .Spectrograph import Detector, Chip
from .Spectrograph import Spectrograph

class LongSlit(Spectrograph):

    """
    A class that describes the long-slit spectrograph in terms of its detector,
    camera, collimator, slit and grating as well as functions related to it.
    All angles are in degrees.
    """

    def __init__(self, telescope, slit, collimator, grating, camera, detector, factors):

        # Initialise the equations
        self.equations = Equations()
        # Initialise the telescope
        self.telescope = Optics(**telescope)
        # Initialise the slit
        self.slit = Slit(**slit)
        # Set width (in mm)
        self.slit.width = self.slit.calc_width(self.telescope.focal_length)
        # Initialise the collimator
        self.collimator = Optics(**collimator)
        # Initialise the grating
        self.grating = Grating(**grating)
        # Initialise the camera
        self.camera = Optics(**camera)
        # Initialise the detector chips
        chips = []
        for chip in detector['chip']:
            chips.append(Chip(**chip))
        # 'Replace' detector dictionary chip entry with initialised chips
        detector['chip'] = chips
        # Initialise the detector
        self.detector = Detector(**detector)
        # Set alpha and beta factors
        self.da = factors['da']
        self.mF = factors['mF']

    def alpha(self):
        """Return the value of alpha for the spectrograph"""
        return self.grating.gr_ang + self.da

    def beta(self):
        """Return the value of beta for the spectrograph"""
        return (1 + self.mF) * self.grating.ar_ang - self.alpha()

    def get_wavelength(self, p, j=1, axis=1):
        """
        For a given spectrograph configuration, return the wavelength
        coordinate associated with a pixel coordinate.
        p: 1-D Array of pixel coordinates
        j: The row (column) being analyzed, default=1
        axis: Dispersion axis: vertical=0, horizontal=1, default=1
        returns array of wavelengths in mm
        """
        # Calculate 'shift' of pixel coordinates from centre row (column):
        # - dispersion axis: vertical
        if axis == 0:
            d = (self.detector.ybin * 
                    self.detector.pix_size * 
                        (p - self.detector.ypix_centre()))
        # - dispersion axis: horizontal
        elif axis == 1:
            d = (self.detector.xbin * 
                    self.detector.pix_size * 
                        (p - self.detector.xpix_centre()))
        # Delta beta
        dbeta = -np.degrees(np.arctan(d / self.camera.focal_length))

        # Set column (row) k adjusted for detector position:
        # NOTE: This does not take individual chip positions into account!
        # - dispersion axis: vertical
        if axis == 0:
            k = j - int(self.detector.xpos / self.detector.pix_size / 
                          self.detector.xbin)
        # - dispersion axis: horizontal
        elif axis == 1:
            k = j - int(self.detector.ypos / self.detector.pix_size /
                          self.detector.ybin)

        # Set fixed factors for calculating the gamma of a column (row):
        # - dispersion axis: vertical
        if axis == 0:
            f1 = (self.detector.pix_size * self.detector.xbin / 
                    self.camera.focal_length)
            f2 = 0.5 * self.detector.width / self.camera.focal_length
        # - dispersion axis: horizontal
        elif axis == 1:
            f1 = (self.detector.pix_size * self.detector.ybin / 
                    self.camera.focal_length)
            f2 = 0.5 * self.detector.height / self.camera.focal_length
        # Calculate gamma of column (row) k
        gamma = np.arctan(k * f1 - f2) * 180. / np.pi
        # Wavelength
        w = self.wavelength(self.alpha(), -self.beta() + dbeta, gamma=gamma)

        return w

    def get_info(self):

        # Initialise info dictionary for the current configuration
        info = {}

        # Grating information:
        # - add grating name to info dictionary
        info['grating'] = self.grating.name
        # - add articulation angle to info dictionary
        info['ar_ang'] = self.grating.ar_ang
        # - add grating angle to info dictionary
        info['gr_ang'] = self.grating.gr_ang

        # Slit information:
        # - add slit name to info dictionary
        info['slit'] = self.slit.name
        # - add slit width in arcsec (i.e., phi) to info dictionary
        info['width'] = self.slit.phi

        # Wavelength information:
        # - get the central wavelength
        w_cen = self.get_central_wavelength()
        # - add to info dictionary
        info['central_wavelength'] = w_cen
        # - get the wavelength edges
        w_blue = self.get_blue_wavelength()
        w_red = self.get_red_wavelength()
        # - add to info dictionary
        info['blue_wavelength_edge'] = w_blue
        info['red_wavelength_edge'] = w_red

        # Resolution information:
        # - get the resolution element (angstroms)
        dw = self.get_resolution_element()
        # - add to info dictionary
        info['resolution_element'] = dw
        # - get the resolution
        R = self.get_resolution(w_cen)
        # - add to info dictionary
        info['resolution'] = R

        return info

    def get_central_wavelength(self):
        """Return central wavelength in angstroms"""
        return 1e7 * self.central_wavelength(self.alpha(), -self.beta())

    def get_blue_wavelength(self):
        """Return blue 'edge' wavelength in angstroms"""
        return 1e7 * self.blue_wavelength(self.alpha(), -self.beta())

    def get_red_wavelength(self):
        """Return red 'edge' wavelength in angstroms"""
        return 1e7 * self.red_wavelength(self.alpha(), -self.beta())

    def get_resolution_element(self):
        """Return resolution element in angstroms"""
        return 1e7 * self.resolution_element(self.alpha(), -self.beta())

    def get_resolution(self, w):
        """Return resolution at wavelength w"""
        return w / self.get_resolution_element()
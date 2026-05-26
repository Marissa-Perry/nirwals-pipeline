# Standard library imports
import math

# numpy import
import numpy as np

class Equations:

    def sind(self, x):
        """Return sin of x where x is in degrees"""
        if isinstance(x, np.ndarray):
            return np.sin(math.pi * x / 180.0)
        return math.sin(math.radians(x))

    def cosd(self, x):
        """Return cos of x where x is in degrees"""
        if isinstance(x, np.ndarray):
            return np.cos(math.pi * x / 180.0)
        return math.cos(math.radians(x))

    def tand(self, x):
        """Return tan of x where x is in degrees"""
        if isinstance(x, np.ndarray):
            return np.tan(math.pi * x / 180.0)
        # return math.cos(math.radians(x))  #### must be a typo ? modified on 03/15/2026 to be tan instead of cos
        return math.tan(math.radians(x))

    def n_index(self):
        return 1.

    def grating_equation(self, sigma, order, sign, alpha, beta, gamma=0., nd=None):
        """Apply the grating equation to determine the wavelength
        w = sigma/m * cos (gamma) * nd * (sin alpha +- sin beta)
        Return wavelength in mm"""
        if nd is None: nd = self.n_index()
        angle = self.cosd(gamma) * nd * (self.sind(alpha) + sign * self.sind(beta))
        w = sigma / order * angle
        return w

    def anamorphic_magnification(self, alpha, beta):
        """Calculate the anamorphic magnification
        Return anamorpic magnification"""
        return self.cosd(alpha) / self.cosd(beta)

    def angular_dispersion(self, sigma, order, beta, gamma=0.):
        """Calculate the angular dispersion according to m/sigma/cos(beta)/cos(gamma)
        Return angular dispersion in 1/mm"""
        return order / sigma / self.cosd(beta) / self.cosd(gamma)

    def resolution_element(self, slitw, fcol, sigma, order, alpha, beta, gamma=0.):
        """Calculate the resolution element using dw=r*slitw/A/fcol
        Return resolution element in mm"""
        r = self.anamorphic_magnification(alpha, beta)
        A = self.angular_dispersion(sigma, order, beta, gamma=gamma)
        return r * slitw / A / fcol

    def resolution(self, w, dw):
        """Calculate the resolution R=w/dw
        Return resolution"""
        return w / dw

class Optics:

    """
    A class that describes optics. This assumes all optics can be desribed by
    a focal length - focal_length is in mm.
    """

    def __init__(self, name='', focal_length=100):
        # define the variables that describe the optics
        self.name = name
        self.focal_length = focal_length

class Chip:

    """
    Defines a detector chip by x and y position, size, and pixel size.  The x
    and y positions are set such that they are 0 relative to the detector
    position. This assumes that the x and y positions are in the centre of the
    pixels and that the chip is symmetric - pix_size is in mm.
    """

    def __init__(self, name='', height=0, width=0, xpos=0, ypos=0,
                 pix_size=0.015, xpix=2048, ypix=4096):
        # set the variables
        self.xpos = xpos
        self.ypos = ypos
        self.pix_size = pix_size
        self.xpix = xpix
        self.ypix = ypix
        self.height = self.set_height(height)
        self.width = self.set_width(width)

    def set_width(self, w):
        """
        If the width is less than the number of pixels, then the width is
        given by the number of pixels
        """
        min = self.xpix * self.pix_size
        return max(w, min)

    def set_height(self, h):
        """
        If the height is less than the number of pixels, then the height is
        given by the number of pixels
        """
        min = self.ypix * self.pix_size
        return max(h, min)

    def find_corners(self):
        """
        Return the corners of the chip
        """
        x1 = self.xpos - 0.5 * self.width
        x2 = self.xpos + 0.5 * self.width
        y1 = self.ypos - 0.5 * self.height
        y2 = self.ypos + 0.5 * self.height
        return x1, x2, y1, y2

class Detector(Chip):

    """
    A class describing a detector. It inherits from the Chip class as
    there could be multiple chips at each position.

    name--Name of the detector
    chip--Chip class or list describing the Chip(s) in the detecfor
    xpos--Offset of the x centre of the chip from the central ray in mm
    ypos--Offset of the y centre of the chip from the central ray in mm
    xbin--chip binning in x-direction
    ybin--chip binning in y-direction
    """

    def __init__(self, name='', chip=Chip(), xpos=0, ypos=0, xbin=2, ybin=2):

        # Set the detector up as a list of Chips.
        self.detector = []
        self.pix_size = None

        if isinstance(chip, Chip):
            self.detector = [chip]
            self.pix_size = chip.pix_size

        elif isinstance(chip, list):
            for c in chip:
                if isinstance(c, Chip):
                    self.detector.append(c)
                    if self.pix_size:
                        self.pix_size = min(self.pix_size, c.pix_size)
                    else:
                        self.pix_size = c.pix_size
        else:
            return

        self.nchip = len(self.detector)

        # set up the zero points for the detector
        self.name = name
        self.xpos = xpos
        self.ypos = ypos
        self.xbin = xbin
        self.ybin = ybin

        # check to make sure that the chips don't overlap
        self.real = self.check_chips()

        # determine the max width and height for the detector
        self.width = self.set_width()
        # determine the max width and height for the detector
        self.height = self.set_height()

    def check_chips(self):
        """Check to make sure none of the chips overlap"""
        if self.nchip <= 1:
            return True

        # loop over each chip and check to see if any of the chip
        # overlaps with the coordinates of another chip
        for i in range(self.nchip):
            ax1, ax2, ay1, ay2 = self.detector[i].find_corners()
            for j in range(i + 1, self.nchip):
                bx1, bx2, by1, by2 = self.detector[j].find_corners()
                if ax1 <= bx1 < ax2 or ax1 < bx2 < ax2:
                    if ay1 <= by1 < ay2 or ay1 < by2 < ay2:
                        return False

        return True

    def xpix_centre(self):
        """Return the xpixel centre based on the x and y position"""
        return int((0.5 * self.width - self.xpos) / self.pix_size / self.xbin)

    def ypix_centre(self):
        """Return the xpixel centre based on the x and y position"""
        return int((0.5 * self.height - self.ypos) / self.pix_size / self.ybin)

    def set_width(self):
        """Loop over all the chips in detector and find the width"""
        width = 0
        # return zero if no detector
        if self.nchip < 1:
            return width
        # handle a single detector
        width = self.detector[0].width
        if self.nchip == 1:
            return width
        # Loop over multipe chips to find the width
        ax1, ax2, _, _ = self.detector[0].find_corners()
        xmin = min(ax1, ax2)
        xmax = max(ax1, ax2)
        for chip in self.detector[1:]:
            ax1, ax2, _, _ = chip.find_corners()
            xmin = min(xmin, ax1, ax2)
            xmax = max(xmax, ax1, ax2)
        width = xmax - xmin
        return width

    def set_height(self):
        """Loop over all the chips in detector and find the height"""
        height = 0
        # return zero if no detector
        if self.nchip < 1:
            return height
        # handle a single detector
        height = self.detector[0].height
        if self.nchip == 1:
            return height
        # Loop over multipe chips to find the height
        _, _, ay1, ay2 = self.detector[0].find_corners()
        ymin = min(ay1, ay2)
        ymax = max(ay1, ay2)
        for chip in self.detector[1:]:
            _, _, ay1, ay2 = chip.find_corners()
            ymin = min(ymin, ay1, ay2)
            ymax = max(ymax, ay1, ay2)
        height = ymax - ymin
        return height

class Grating:

    """
    A class describing a grating. Spacing should be in lines/mm and the units
    of the dimensions should be mm.
    """

    def __init__(self, name='', spacing=600, order=1, gr_type='transmission',
                 gr_ang=45, ar_ang=45, da=0):
        # define the variables that describe the grating
        self.name = name
        self.sigma = 1. / spacing
        self.order = order
        self.type = gr_type
        # set the sign for the grating equation
        self.sign = 1
        if self.type == 'transmission':
            self.sign = -1
        # set grating and articulation (camera) angles
        self.gr_ang = gr_ang
        self.ar_ang = ar_ang

class Slit:

    """
    A class that describing the slit. A single slit is assumed - phi is in
    arcseconds and width is in mm.
    """

    def __init__(self, name='', phi=1, width=1):

        # define the variables that describe the slit
        self.name = name
        self.phi = phi
        self.width = width

    def calc_phi(self, ftel):
        """
        Calculate phi(angle on sky) assuming w/ftel
        Return phi in arcseconds
        """
        return 3600. * math.degrees(self.width / ftel)

    def calc_width(self, ftel):
        """
        Calculate width assuming ftel*phi(rad)
        Return slit width in mm
        """
        return ftel * math.radians(self.phi / 3600.)

class Spectrograph:

    """
    A class that describes a spectrograph in terms of its detector, camera,
    collimator, slit and grating as well as functions related to it.
    All angles are in degrees.
    """

    def __init__(self, equations=Equations(), telescope=Optics(),
                 slit=Slit(), collimator=Optics(), grating=Grating(),
                 camera=Optics(), detector=Detector()):

        # Initialise the equations
        self.equations = equations
        # Initialise the telescope
        self.telescope = telescope
        # Initialise the detector
        self.slit = slit
        # Initialise the collimator
        self.collimator = collimator
        # Initialise the grating
        self.grating = grating
        # Initialise the camera
        self.camera = camera
        # Initialise the detector
        self.detector = detector

    def alpha(self):
        return self.grating.gr_ang

    def beta(self):
        return self.grating.ar_ang - self.grating.gr_ang

    def wavelength(self, alpha, beta, gamma=0., nd=None):
        """Apply the grating equation to determine the wavelength (in mm)"""
        w = self.equations.grating_equation(self.grating.sigma,
                                            self.grating.order,
                                            self.grating.sign,
                                            alpha, beta,
                                            gamma=gamma, nd=nd)
        return w

    def central_wavelength(self, alpha, beta):
        """Calculate the central wavelength
        Return wavelength in mm"""
        return self.wavelength(alpha, beta)

    def blue_wavelength(self, alpha, beta):
        """Calculate the blue wavelength 'edge' for the detector
        Return wavelength in mm"""
        dbeta = math.degrees(math.atan(0.5 * 
                                self.detector.width /
                                    self.camera.focal_length))
        return self.wavelength(alpha, beta + dbeta)

    def red_wavelength(self, alpha, beta):
        """Calculate the red wavelength 'edge' for the detector
        Return wavelength in mm"""
        dbeta = math.degrees(math.atan(0.5 * 
                                self.detector.width /
                                    self.camera.focal_length))
        return self.wavelength(alpha, beta - dbeta)

    def resolution_element(self, alpha, beta):
        """Calculate the resolution of a single element for a filled slit
        Return wavelength resolution in mm"""
        return self.equations.resolution_element(self.slit.width,
                                                 self.collimator.focal_length,
                                                 self.grating.sigma,
                                                 self.grating.order,
                                                 alpha, beta)

    def resolution(self, w, alpha, beta):
        """Calculate the resolution at a given wavelength: w/dw
        Return resolution"""
        dw = self.resolution_element(alpha, beta)
        return self.equations.resolution(w, dw)
# ---------------------------------------------------------------------------- #
"""
SALT spectrograph utilities:
- saltspectrograph provides general utilities for initialising a specified
  type of spectrograph class to be used in the reduction of spectral data:
  - ...
"""
# ---------------------------------------------------------------------------- #

# Application imports:
# - LongSlit.LongSlit (generic long-slit spectrograph)
from  .LongSlit import LongSlit

# ---------------------------------------------------------------------------- #

MYNAME = 'saltspectrograph'

# ---------------------------------------------------------------------------- #

# Primary fits extension nr
PRIMARY = 0

# ---------------------------------------------------------------------------- #
def set_spectrograph(hdulist, config):
# ---------------------------------------------------------------------------- #

    """
    hdulist: <astropy.io> opened FITS HDU list
    config: <dictionary> spectrograph config

    return: <class> spectrograph
    """

    # Check spectrograph type is longslit
    if 'longslit' in config['type']:
        # Set longslit spectrograph
        spectrograph = set_longslit_spectrograph(hdulist, config)

    else:
        spectrograph = None
        error = 'Spectrograph type {0} does not exist.'.format(config['type'])
        raise SALTError(error)

    return spectrograph

# ---------------------------------------------------------------------------- #
def set_longslit_spectrograph(hdulist, config):
# ---------------------------------------------------------------------------- #

    # Set the spectrograph detector
    detector = config['detector']

    # Get exposure binning from fits header
    sum = hdulist[PRIMARY].header[config['binning_key']]
    xbin = int(sum.split()[0])
    ybin = int(sum.split()[1])
    # Set the detector binning
    detector['xbin'] = xbin
    detector['ybin'] = ybin

    # Set the spectrograph camera
    camera = config['camera']

    # Get grating name from fits header
    gr_name = hdulist[PRIMARY].header['GRATING']
    # Get grating tilt and camera angle from fits header
    grtilt = hdulist[PRIMARY].header['GRTILT']
    camang = hdulist[PRIMARY].header['CAMANG']
    # Check GRTILT and CAMANG data type
    if isinstance(grtilt, str): grtilt = float(grtilt)
    if isinstance(camang, str): camang = float(camang)
    # Get grating and articulation encoder angles from fits header
    gr_ang = hdulist[PRIMARY].header['GR-ANGLE']
    ar_ang = hdulist[PRIMARY].header['AR-ANGLE']
    # Check GR-ANGLE and AR-ANGLE data type
    if isinstance(gr_ang, str): gr_ang = float(gr_ang)
    if isinstance(ar_ang, str): ar_ang = float(ar_ang)
    # Override grating and articulation encoder angles if necessary
    if abs(gr_ang - grtilt) > 0.15: gr_ang = grtilt
    if abs(ar_ang - camang) > 0.15: ar_ang = camang

    # Set the spectrograph grating
    grating = next(gr for gr in config['gratings'] if gr['name'] == gr_name)
    grating['gr_ang'], grating['ar_ang'] = gr_ang, ar_ang

    # Set the spectrograph collimator
    collimator = config['collimator']

    # Set the spectrograph slit
    slit = config['slit']
    # Check if slit mask details are in config
    if slit['name'] and slit['phi']:
        # Let's go with it!
        pass

    else:
        # Get slit mask details from fits header
        try:
            slit_name = hdulist[PRIMARY].header['MASKID']
            slit_phi = float(slit_name[2:6]) / 100.

        except:
            slit_name = 'PSEUDO'
            slit_phi = 1.5

        # Update slit mask details in config
        slit['name'] = slit_name
        slit['phi'] = slit_phi

    # Set the spectrograph telescope
    telescope = config['telescope']

    # Set alpha and beta factors
    factors = config['factors']

    # Set the spectrograph model
    spectrograph = LongSlit(telescope=telescope, slit=slit,
                            collimator=collimator, grating=grating,
                            camera=camera, detector=detector, factors=factors)
####>
    # # Dynamic import
    # mod = __import__('my_package.my_module', fromlist=['my_class'])
    # spectrograph = getattr(mod, 'my_class')
####<
    return spectrograph

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #